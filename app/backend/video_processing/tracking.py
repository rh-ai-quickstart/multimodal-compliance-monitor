from __future__ import annotations

import json
import queue as queue_mod
import threading
from datetime import datetime
from multiprocessing import Event, Process, Queue

import numpy as np

from logger import get_logger

log = get_logger(__name__)

_RESET = "_RESET"
_CONFIGURE = "_CONFIGURE"

# ----- SQL / op constants (ported from db_write.py) -----

_INSERT_TRACK_SQL = (
    "INSERT INTO detection_tracks (track_id, detection_classes_id, first_seen, last_seen) "
    "VALUES (%s, %s, %s, %s) "
    "ON CONFLICT (track_id) DO UPDATE SET last_seen = EXCLUDED.last_seen"
)
_UPDATE_LAST_SEEN_SQL = "UPDATE detection_tracks SET last_seen = %s WHERE track_id = %s"
_INSERT_OBSERVATION_SQL = (
    "INSERT INTO detection_observations (track_id, timestamp, attributes) "
    "VALUES (%s, %s, %s)"
)

_OP_INSERT_TRACK = "insert_track"
_OP_UPDATE_LAST_SEEN = "update_last_seen"
_OP_INSERT_OBSERVATION = "insert_observation"

_SQL_BY_OP = {
    _OP_INSERT_TRACK: _INSERT_TRACK_SQL,
    _OP_UPDATE_LAST_SEEN: _UPDATE_LAST_SEEN_SQL,
    _OP_INSERT_OBSERVATION: _INSERT_OBSERVATION_SQL,
}

_OP_ORDER = (_OP_INSERT_TRACK, _OP_UPDATE_LAST_SEEN, _OP_INSERT_OBSERVATION)


class TrackerProcess:
    """Fire-and-forget background process: DeepSORT tracking + batched DB writes.

    The main thread pushes ``(tracker_input_dets, detections, frame)`` tuples
    via :meth:`submit`.  The frame is pickled through the multiprocessing Queue
    so every item the child receives owns its own copy — no shared-memory
    races.  The child runs DeepSORT, associates PPE items to tracked persons,
    and writes tracks/observations to PostgreSQL in batches.
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        last_seen_update_interval: int = 30,
    ) -> None:
        self._max_age = max_age
        self._n_init = n_init
        self._last_seen_update_interval = last_seen_update_interval

        self._in_queue: Queue = Queue(maxsize=100)
        self._stop = Event()
        self._process: Process | None = None

    def start(self) -> None:
        self._stop.clear()
        self._process = Process(
            target=_tracker_process_target,
            args=(
                self._in_queue,
                self._stop,
                self._max_age,
                self._n_init,
                self._last_seen_update_interval,
            ),
            daemon=True,
        )
        self._process.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._process is not None:
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
            self._process = None

    def configure(
        self,
        trackable_by_class_id: dict[int, bool],
        detection_class_name_to_id: dict[str, int],
    ) -> None:
        """Send new pipeline config to the process (called on stream start/switch)."""
        try:
            self._in_queue.put(
                {
                    "kind": _CONFIGURE,
                    "trackable_by_class_id": trackable_by_class_id,
                    "detection_class_name_to_id": detection_class_name_to_id,
                },
                timeout=5.0,
            )
        except queue_mod.Full:
            log.error("Tracker: failed to send CONFIGURE (queue full)")

    def submit(
        self,
        tracker_input_dets: list,
        frame: np.ndarray,
        detections: list[dict],
    ) -> None:
        """Fire-and-forget: enqueue frame + detection metadata for the child process."""
        try:
            self._in_queue.put_nowait((tracker_input_dets, detections, frame))
        except queue_mod.Full:
            log.warning("Tracker input queue full, dropping frame")

    def reset(self) -> None:
        _drain(self._in_queue)
        try:
            self._in_queue.put_nowait(_RESET)
        except queue_mod.Full:
            log.warning("Tracker input queue full, reset message dropped")


# ----- Batch DB writer (runs inside the tracker process) -----


class _BatchDbWriter:
    """Offloads DB writes to a dedicated background thread.

    The tracker loop builds a batch of ``(op, args)`` tuples per frame and
    hands the whole list to :meth:`submit_batch`.  A single writer thread
    drains *all* pending batches from a bounded queue, merges them into one
    combined transaction, and flushes to PostgreSQL -- so multiple frames
    that pile up while the DB is busy collapse into a single round-trip.
    """

    _QUEUE_MAXSIZE = 64

    def __init__(self) -> None:
        self._queue: queue_mod.Queue[list[tuple[str, tuple]] | None] = queue_mod.Queue(
            maxsize=self._QUEUE_MAXSIZE
        )
        self._conn = None
        self._thread = threading.Thread(
            target=self._writer_loop, name="db-writer", daemon=True
        )
        self._thread.start()

    def submit_batch(self, batch: list[tuple[str, tuple]]) -> None:
        """Hand off a frame's DB ops to the writer thread (blocks if queue full)."""
        if not batch:
            return
        self._queue.put(batch)

    def close(self) -> None:
        """Signal the writer thread to drain remaining work and exit."""
        self._queue.put(None)
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            log.warning("DB writer: thread did not exit in time")

    def _writer_loop(self) -> None:
        log.info("DB writer thread started")
        try:
            while True:
                first = self._queue.get()
                if first is None:
                    break

                batches = [first]
                sentinel_seen = False
                while True:
                    try:
                        item = self._queue.get_nowait()
                    except queue_mod.Empty:
                        break
                    if item is None:
                        sentinel_seen = True
                        break
                    batches.append(item)

                combined: list[tuple[str, tuple]] = []
                for b in batches:
                    combined.extend(b)

                if combined:
                    log.info(
                        "DB writer: flushing %d ops from %d batches",
                        len(combined),
                        len(batches),
                    )
                    self._flush(combined)

                if sentinel_seen:
                    break
        except Exception:
            log.exception("DB writer thread crashed")
        finally:
            self._close_conn()
            log.info("DB writer thread exited")

    def _flush(self, batch: list[tuple[str, tuple]]) -> None:
        groups: dict[str, list[tuple]] = {}
        for op, args in batch:
            groups.setdefault(op, []).append(args)

        conn = self._ensure_conn()
        if conn is None:
            log.warning(f"DB writer: no connection, dropping {len(batch)} items")
            return

        cursor = conn.cursor()
        try:
            for op in _OP_ORDER:
                rows = groups.get(op)
                if rows:
                    cursor.executemany(_SQL_BY_OP[op], rows)
            conn.commit()
        except Exception:
            log.exception(f"DB writer: batch failed ({len(batch)} items), retrying")
            self._close_conn()
            try:
                conn = self._ensure_conn()
                if conn is not None:
                    cursor = conn.cursor()
                    for op in _OP_ORDER:
                        rows = groups.get(op)
                        if rows:
                            cursor.executemany(_SQL_BY_OP[op], rows)
                    conn.commit()
            except Exception:
                log.exception(f"DB writer: retry failed, dropping {len(batch)} items")
                self._close_conn()

    def _ensure_conn(self):
        import psycopg2
        from database import get_connection_string

        if self._conn is None or self._conn.closed:
            try:
                self._conn = psycopg2.connect(get_connection_string())
            except Exception:
                log.exception("DB writer: failed to connect")
                self._conn = None
        return self._conn

    def _close_conn(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


# ----- PPE association -----

_PPE_MAPPING = {
    "Hardhat": ("hardhat", True),
    "NO-Hardhat": ("hardhat", False),
    "Safety Vest": ("vest", True),
    "NO-Safety Vest": ("vest", False),
    "Mask": ("mask", True),
    "NO-Mask": ("mask", False),
}


def _associate_ppe_to_person(
    person_bbox: tuple[int, int, int, int],
    all_detections: list[dict],
) -> dict[str, bool | None]:
    """Determine PPE status for a person based on bounding box overlap."""
    status: dict[str, bool | None] = {"hardhat": None, "vest": None, "mask": None}
    for det in all_detections:
        mapping = _PPE_MAPPING.get(det["class_name"])
        if mapping is None:
            continue
        if _boxes_overlap(person_bbox, det["bbox"]):
            ppe_type, ppe_value = mapping
            if status[ppe_type] is None:
                status[ppe_type] = ppe_value
    return status


# ----- Process target -----


def _tracker_process_target(
    in_q: Queue,
    stop_event: Event,
    max_age: int,
    n_init: int,
    last_seen_update_interval: int,
) -> None:
    from logger import get_logger

    log = get_logger(__name__)

    tracker = _Tracker(
        max_age=max_age,
        n_init=n_init,
        last_seen_update_interval=last_seen_update_interval,
    )
    db = _BatchDbWriter()

    trackable_by_class_id: dict[int, bool] = {}
    detection_class_name_to_id: dict[str, int] = {}
    person_last_state: dict[int, tuple] = {}

    log.info("Tracker process started")
    try:
        while not stop_event.is_set():
            try:
                item = in_q.get(timeout=0.05)
            except queue_mod.Empty:
                continue

            if item == _RESET:
                tracker.reset()
                person_last_state.clear()
                continue

            if isinstance(item, dict):
                kind = item.get("kind")
                if kind == _CONFIGURE:
                    trackable_by_class_id = item["trackable_by_class_id"]
                    detection_class_name_to_id = item["detection_class_name_to_id"]
                    person_last_state.clear()
                    log.info(
                        "Tracker: configured with %d trackable classes, %d name-to-id mappings",
                        sum(1 for v in trackable_by_class_id.values() if v),
                        len(detection_class_name_to_id),
                    )
                    continue

            tracker_input_dets, detections, frame = item

            result = tracker.update(
                tracker_input_dets, frame, detections, trackable_by_class_id
            )

            now = datetime.now()

            frame_batch: list[tuple[str, tuple]] = []
            for track_id, person_bbox in result.tracked_boxes.items():
                if track_id in result.new_track_ids:
                    tname = result.track_det_class.get(track_id)
                    dcid = detection_class_name_to_id.get(tname) if tname else None
                    if dcid is not None:
                        frame_batch.append(
                            (_OP_INSERT_TRACK, (track_id, dcid, now, now))
                        )
                elif track_id in result.updated_track_ids:
                    frame_batch.append((_OP_UPDATE_LAST_SEEN, (now, track_id)))

                ppe_status = _associate_ppe_to_person(person_bbox, detections)
                current_state = (
                    ppe_status["hardhat"],
                    ppe_status["vest"],
                    ppe_status["mask"],
                )
                last_state = person_last_state.get(track_id)

                if last_state is None or last_state != current_state:
                    attributes = {
                        k: v
                        for k, v in [
                            ("hardhat", ppe_status["hardhat"]),
                            ("vest", ppe_status["vest"]),
                            ("mask", ppe_status["mask"]),
                        ]
                        if v is not None
                    }
                    frame_batch.append(
                        (
                            _OP_INSERT_OBSERVATION,
                            (track_id, now, json.dumps(attributes)),
                        )
                    )
                    person_last_state[track_id] = current_state

            db.submit_batch(frame_batch)

    except Exception:
        log.exception("Tracker process crashed")
    finally:
        db.close()
        log.info("Tracker process exited")


# ----- Helpers -----


def _drain(q: Queue) -> None:
    while True:
        try:
            q.get_nowait()
        except queue_mod.Empty:
            break


class _Tracker:
    """Internal DeepSORT wrapper -- only called from within the tracker process."""

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        last_seen_update_interval: int = 30,
    ) -> None:
        from deep_sort_realtime.deepsort_tracker import DeepSort

        self._max_age = max_age
        self._n_init = n_init
        self._last_seen_update_interval = last_seen_update_interval

        self._tracker = DeepSort(max_age=max_age, n_init=n_init)
        self._track_history: dict[int, dict[str, datetime]] = {}
        self._frames_since_last_seen_update = 0

    def reset(self) -> None:
        from deep_sort_realtime.deepsort_tracker import DeepSort

        self._tracker = DeepSort(max_age=self._max_age, n_init=self._n_init)
        self._track_history = {}
        self._frames_since_last_seen_update = 0

    def update(
        self,
        tracker_input_dets: list,
        frame: np.ndarray,
        detections: list[dict],
        trackable_by_class_id: dict[int, bool],
    ) -> _TrackerResult:
        tracked_boxes: dict[int, tuple[int, int, int, int]] = {}
        track_det_class: dict[int, str] = {}

        if tracker_input_dets:
            tracks = []
            try:
                tracks = self._tracker.update_tracks(tracker_input_dets, frame=frame)
            except IndexError as e:
                log.error(f"Tracker IndexError: {e}", exc_info=True)

            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = int(track.track_id)
                ltrb = track.to_ltrb()
                if ltrb is not None:
                    x1, y1, x2, y2 = map(int, ltrb)
                    tracked_boxes[track_id] = (x1, y1, x2, y2)
                dc = track.get_det_class()
                if dc:
                    track_det_class[track_id] = dc

        _assign_track_ids(
            detections, trackable_by_class_id, tracked_boxes, track_det_class
        )

        new_track_ids, updated_track_ids = self._update_history(tracked_boxes)

        return _TrackerResult(
            tracked_boxes=tracked_boxes,
            track_det_class=track_det_class,
            new_track_ids=new_track_ids,
            updated_track_ids=updated_track_ids,
        )

    def _update_history(
        self,
        tracked_boxes: dict[int, tuple[int, int, int, int]],
    ) -> tuple[list[int], list[int]]:
        now = datetime.now()
        self._frames_since_last_seen_update += 1
        do_last_seen_update = (
            self._frames_since_last_seen_update >= self._last_seen_update_interval
        )

        new_track_ids: list[int] = []
        updated_track_ids: list[int] = []

        for track_id in tracked_boxes:
            if track_id not in self._track_history:
                self._track_history[track_id] = {
                    "first_seen": now,
                    "last_seen": now,
                }
                new_track_ids.append(track_id)
            else:
                self._track_history[track_id]["last_seen"] = now
                if do_last_seen_update:
                    updated_track_ids.append(track_id)

        if do_last_seen_update:
            self._frames_since_last_seen_update = 0

        return new_track_ids, updated_track_ids


class _TrackerResult:
    """Lightweight result container (process-internal, not pickled)."""

    __slots__ = (
        "tracked_boxes",
        "track_det_class",
        "new_track_ids",
        "updated_track_ids",
    )

    def __init__(
        self,
        tracked_boxes: dict[int, tuple[int, int, int, int]],
        track_det_class: dict[int, str],
        new_track_ids: list[int],
        updated_track_ids: list[int],
    ) -> None:
        self.tracked_boxes = tracked_boxes
        self.track_det_class = track_det_class
        self.new_track_ids = new_track_ids
        self.updated_track_ids = updated_track_ids


def _assign_track_ids(
    detections: list[dict],
    trackable_by_class_id: dict[int, bool],
    tracked_boxes: dict[int, tuple[int, int, int, int]],
    track_det_class: dict[int, str],
) -> None:
    for det in detections:
        if not trackable_by_class_id.get(det["class_id"], False):
            continue
        dname = det["class_name"]
        for tid, pbox in tracked_boxes.items():
            tcn = track_det_class.get(tid)
            if tcn is not None and tcn != dname:
                continue
            if _boxes_overlap(det["bbox"], pbox):
                det["track_id"] = tid
                break


def _boxes_overlap(
    box1: tuple[int, int, int, int],
    box2: tuple[int, int, int, int],
) -> bool:
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    if x2_1 < x1_2 or x2_2 < x1_1:
        return False
    if y2_1 < y1_2 or y2_2 < y1_1:
        return False
    return True
