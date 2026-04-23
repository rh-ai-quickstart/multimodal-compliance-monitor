from __future__ import annotations

import json
import queue as queue_mod
import threading
from datetime import datetime
from multiprocessing import Event, Process, Queue

import numpy as np
import supervision as sv

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

_MAX_DRAIN = 8


class TrackerProcess:
    """Fire-and-forget background process: ByteTrack tracking + batched DB writes.

    The main thread pushes detection dicts via :meth:`submit`.  No frame pixels
    are sent through the queue (ByteTrack is motion-only, no embedding CNN).
    The child runs ByteTrack, associates PPE items to tracked persons via
    vectorized numpy overlap, and writes tracks/observations to PostgreSQL in
    batches.
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

    def submit(self, detections: list[dict]) -> None:
        """Fire-and-forget: enqueue detection dicts for the child process."""
        try:
            self._in_queue.put_nowait(detections)
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
                    log.debug(
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


# ----- PPE association (vectorized) -----

_PPE_MAPPING = {
    "Hardhat": ("hardhat", True),
    "NO-Hardhat": ("hardhat", False),
    "Safety Vest": ("vest", True),
    "NO-Safety Vest": ("vest", False),
    "Mask": ("mask", True),
    "NO-Mask": ("mask", False),
}

_EMPTY_PPE = {"hardhat": None, "vest": None, "mask": None}


def _batch_associate_ppe(
    person_xyxy: np.ndarray,
    all_det_xyxy: np.ndarray,
    all_det_classes: list[str],
) -> list[dict[str, bool | None]]:
    """Vectorized PPE association for all persons against all detections.

    Computes a (P, K) overlap matrix in one numpy broadcast, then walks
    only the sparse hits to assign first-match-wins PPE status.
    """
    n_persons = len(person_xyxy)
    if n_persons == 0:
        return []

    ppe_indices = [i for i, cn in enumerate(all_det_classes) if cn in _PPE_MAPPING]
    if not ppe_indices:
        return [dict(_EMPTY_PPE) for _ in range(n_persons)]

    ppe_idx_arr = np.array(ppe_indices)
    ppe_xyxy = all_det_xyxy[ppe_idx_arr]

    # (P, K) boolean overlap matrix via broadcast
    overlap = (
        (person_xyxy[:, 0:1] <= ppe_xyxy[:, 2])
        & (person_xyxy[:, 2:3] >= ppe_xyxy[:, 0])
        & (person_xyxy[:, 1:2] <= ppe_xyxy[:, 3])
        & (person_xyxy[:, 3:4] >= ppe_xyxy[:, 1])
    )

    results: list[dict[str, bool | None]] = [dict(_EMPTY_PPE) for _ in range(n_persons)]
    for k_idx, det_idx in enumerate(ppe_indices):
        ppe_type, ppe_val = _PPE_MAPPING[all_det_classes[det_idx]]
        for p in np.where(overlap[:, k_idx])[0]:
            if results[p][ppe_type] is None:
                results[p][ppe_type] = ppe_val
    return results


def _dicts_to_sv_detections(detections: list[dict]) -> sv.Detections:
    """Convert the app's detection dicts to a supervision Detections object."""
    if not detections:
        return sv.Detections.empty()

    xyxy = np.array([d["bbox"] for d in detections], dtype=np.float32)
    confidence = np.array([d["confidence"] for d in detections], dtype=np.float32)
    class_id = np.array([d["class_id"] for d in detections], dtype=int)
    class_names = np.array([d["class_name"] for d in detections])

    return sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        data={"class_name": class_names},
    )


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

    from database import get_max_track_id

    track_id_offset = get_max_track_id()
    log.info("Tracker: resuming with track_id_offset=%d", track_id_offset)

    tracker = _Tracker(
        max_age=max_age,
        n_init=n_init,
        last_seen_update_interval=last_seen_update_interval,
        track_id_offset=track_id_offset,
    )
    db = _BatchDbWriter()

    trackable_by_class_id: dict[int, bool] = {}
    detection_class_name_to_id: dict[str, int] = {}
    person_last_state: dict[int, tuple] = {}

    log.info("Tracker process started")
    try:
        while not stop_event.is_set():
            # --- Drain up to _MAX_DRAIN items from the queue ---
            try:
                first = in_q.get(timeout=0.1)
            except queue_mod.Empty:
                continue

            raw_items = [first]
            while len(raw_items) < _MAX_DRAIN:
                try:
                    raw_items.append(in_q.get_nowait())
                except queue_mod.Empty:
                    break

            det_frames: list[list[dict]] = []
            for item in raw_items:
                if item == _RESET:
                    tracker.reset()
                    person_last_state.clear()
                    det_frames.clear()
                    continue

                if isinstance(item, dict):
                    kind = item.get("kind")
                    if kind == _CONFIGURE:
                        trackable_by_class_id = item["trackable_by_class_id"]
                        detection_class_name_to_id = item["detection_class_name_to_id"]
                        person_last_state.clear()
                        log.info(
                            "Tracker: configured with %d trackable classes, "
                            "%d name-to-id mappings",
                            sum(1 for v in trackable_by_class_id.values() if v),
                            len(detection_class_name_to_id),
                        )
                        det_frames.clear()
                        continue

                det_frames.append(item)

            if not det_frames:
                continue

            # --- Phase 1: Sequential ByteTrack updates ---
            tracker_results: list[tuple[_TrackerResult, list[dict]]] = []
            for detections in det_frames:
                sv_dets = _dicts_to_sv_detections(detections)
                result = tracker.update(sv_dets, trackable_by_class_id)
                tracker_results.append((result, detections))

            # --- Phase 2: Vectorized PPE association across all frames ---
            all_person_xyxy: list[np.ndarray] = []
            all_det_xyxy: list[np.ndarray] = []
            all_det_classes: list[list[str]] = []
            person_counts: list[int] = []

            for result, detections in tracker_results:
                boxes = result.tracked_boxes
                n_p = len(boxes)
                person_counts.append(n_p)
                if n_p > 0:
                    all_person_xyxy.append(
                        np.array(list(boxes.values()), dtype=np.float32)
                    )
                det_xyxy = (
                    np.array([d["bbox"] for d in detections], dtype=np.float32)
                    if detections
                    else np.empty((0, 4), dtype=np.float32)
                )
                all_det_xyxy.append(det_xyxy)
                all_det_classes.append([d["class_name"] for d in detections])

            ppe_results_by_frame: list[list[dict[str, bool | None]]] = []
            for i, (result, detections) in enumerate(tracker_results):
                if person_counts[i] > 0:
                    ppe_results_by_frame.append(
                        _batch_associate_ppe(
                            all_person_xyxy.pop(0),
                            all_det_xyxy[i],
                            all_det_classes[i],
                        )
                    )
                else:
                    ppe_results_by_frame.append([])

            # --- Phase 3: Sequential state diff + combined DB ops ---
            now = datetime.now()
            combined_batch: list[tuple[str, tuple]] = []

            for frame_idx, (result, detections) in enumerate(tracker_results):
                person_idx = 0
                for track_id, person_bbox in result.tracked_boxes.items():
                    if track_id in result.new_track_ids:
                        tname = result.track_det_class.get(track_id)
                        dcid = detection_class_name_to_id.get(tname) if tname else None
                        if dcid is not None:
                            combined_batch.append(
                                (_OP_INSERT_TRACK, (track_id, dcid, now, now))
                            )
                    elif track_id in result.updated_track_ids:
                        combined_batch.append((_OP_UPDATE_LAST_SEEN, (now, track_id)))

                    ppe_status = ppe_results_by_frame[frame_idx][person_idx]
                    person_idx += 1

                    current_state = (
                        ppe_status["hardhat"],
                        ppe_status["vest"],
                        ppe_status["mask"],
                    )
                    last_state = person_last_state.get(track_id)

                    if last_state is None or last_state != current_state:
                        attributes = {
                            k: v for k, v in ppe_status.items() if v is not None
                        }
                        combined_batch.append(
                            (
                                _OP_INSERT_OBSERVATION,
                                (track_id, now, json.dumps(attributes)),
                            )
                        )
                        person_last_state[track_id] = current_state

                _assign_track_ids(
                    detections,
                    trackable_by_class_id,
                    result.tracked_boxes,
                    result.track_det_class,
                )

            db.submit_batch(combined_batch)

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
    """Internal ByteTrack wrapper -- only called from within the tracker process."""

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        last_seen_update_interval: int = 30,
        track_id_offset: int = 0,
    ) -> None:
        self._max_age = max_age
        self._n_init = n_init
        self._last_seen_update_interval = last_seen_update_interval
        self._track_id_offset = track_id_offset

        self._tracker = sv.ByteTrack(
            lost_track_buffer=max_age,
            minimum_consecutive_frames=n_init,
        )
        self._track_history: dict[int, dict[str, datetime]] = {}
        self._frames_since_last_seen_update = 0

    def reset(self) -> None:
        self._tracker = sv.ByteTrack(
            lost_track_buffer=self._max_age,
            minimum_consecutive_frames=self._n_init,
        )
        self._track_history = {}
        self._frames_since_last_seen_update = 0

    def update(
        self,
        sv_dets: sv.Detections,
        trackable_by_class_id: dict[int, bool],
    ) -> _TrackerResult:
        tracked_boxes: dict[int, tuple[int, int, int, int]] = {}
        track_det_class: dict[int, str] = {}

        if len(sv_dets) > 0:
            mask = (
                np.array(
                    [
                        trackable_by_class_id.get(int(cid), False)
                        for cid in sv_dets.class_id
                    ]
                )
                if sv_dets.class_id is not None
                else np.zeros(len(sv_dets), dtype=bool)
            )

            trackable_dets = sv_dets[mask]

            if len(trackable_dets) > 0:
                tracked = self._tracker.update_with_detections(trackable_dets)

                for i in range(len(tracked)):
                    tid = int(tracked.tracker_id[i]) + self._track_id_offset
                    x1, y1, x2, y2 = tracked.xyxy[i].astype(int)
                    tracked_boxes[tid] = (int(x1), int(y1), int(x2), int(y2))
                    if (
                        tracked.data.get("class_name") is not None
                        and len(tracked.data["class_name"]) > i
                    ):
                        track_det_class[tid] = tracked.data["class_name"][i]
            else:
                self._tracker.update_with_detections(sv.Detections.empty())

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
