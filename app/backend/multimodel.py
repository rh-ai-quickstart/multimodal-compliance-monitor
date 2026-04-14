import cv2
import json
import numpy as np
import os
import tempfile
import threading
import time
import atexit
from collections import defaultdict
from datetime import datetime
import queue
from multiprocessing import Process, Queue, Event
from multiprocessing.shared_memory import SharedMemory

from minio_client import download_file
from database import (
    init_database,
    get_detection_classes_pipeline_maps,
    get_all_configs,
    get_config_by_id,
    DbWriterThread,
    OP_INSERT_TRACK,
    OP_UPDATE_LAST_SEEN,
    OP_INSERT_OBSERVATION,
)
from logger import get_logger
from response import process_detections
from runtime import Runtime

log = get_logger(__name__)

# config_queue message kinds (multiprocessing — must be picklable dicts)
CONFIG_MSG_INIT_SHM = "INIT_SHM"
CONFIG_MSG_RELOAD_CONFIG = "RELOAD_CONFIG"


def _resolve_video_source_to_path(video_source: str) -> tuple[str, str | None]:
    """
    Resolve video_source to a path cv2.VideoCapture can open.
    Returns (path_to_use, temp_path_or_none). If temp_path is set, caller must delete it.
    S3 URIs (s3://bucket/key) are downloaded to a temp file.
    """
    if not video_source or not isinstance(video_source, str):
        return video_source or "", None
    p = video_source.strip()
    if p.startswith("s3://"):
        parts = p[5:].split("/", 1)
        if len(parts) == 2:
            bucket, key = parts[0], parts[1]
            fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            try:
                download_file(bucket, key, tmp_path)
                return tmp_path, tmp_path
            except Exception as e:
                log.exception("Failed to download S3 video %s: %s", video_source, e)
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
    return video_source, None


# --- Module-level helpers (used by both main and inference process) ---


def _boxes_overlap(box1, box2):
    """Check if two bounding boxes overlap."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    if x2_1 < x1_2 or x2_2 < x1_1:
        return False
    if y2_1 < y1_2 or y2_2 < y1_1:
        return False
    return True


def _associate_ppe_to_person(person_bbox, all_detections):
    """Determine PPE status for a person based on bounding box overlap."""
    status = {"hardhat": None, "vest": None, "mask": None}
    ppe_mapping = {
        "Hardhat": ("hardhat", True),
        "NO-Hardhat": ("hardhat", False),
        "Safety Vest": ("vest", True),
        "NO-Safety Vest": ("vest", False),
        "Mask": ("mask", True),
        "NO-Mask": ("mask", False),
    }
    for det in all_detections:
        class_name = det["class_name"]
        if class_name in ppe_mapping:
            ppe_bbox = det["bbox"]
            if _boxes_overlap(person_bbox, ppe_bbox):
                ppe_type, ppe_value = ppe_mapping[class_name]
                if status[ppe_type] is None:
                    status[ppe_type] = ppe_value
    return status


def model_class_names_in_order(classes: dict[int, str]) -> list[str]:
    """Class names in model_class_index order (for Latest Detection text)."""
    return [classes[i] for i in sorted(classes.keys())]


def format_detection_description(
    detections_class_count: dict[str, int],
    class_names_in_order: list[str],
) -> str:
    """Build a short, human-readable description from counts; order follows config/model indices."""
    description = "Detected: "
    for item in class_names_in_order:
        if detections_class_count.get(item, 0) > 0:
            description += f"{item}: {detections_class_count[item]}, "
    return description.rstrip(", ")


def generate_summary(descriptions: list) -> str:
    """Summarize PPE compliance over a list of detection descriptions."""
    total_stats = defaultdict(int)
    frame_count = len(descriptions)
    for desc in descriptions:
        for item in [
            "Person",
            "Hardhat",
            "Safety Vest",
            "Mask",
            "NO-Hardhat",
            "NO-Safety Vest",
            "NO-Mask",
        ]:
            count = desc.count(item)
            total_stats[item] += count

    summary = "Safety Trends Summary:\n\n"
    summary += f"Total observations: {frame_count} frames\n\n"

    if total_stats["Person"] > 0:
        hardhat_compliance = (
            total_stats["Hardhat"]
            / (total_stats["Hardhat"] + total_stats["NO-Hardhat"])
            if (total_stats["Hardhat"] + total_stats["NO-Hardhat"]) > 0
            else 0
        )
        vest_compliance = (
            total_stats["Safety Vest"]
            / (total_stats["Safety Vest"] + total_stats["NO-Safety Vest"])
            if (total_stats["Safety Vest"] + total_stats["NO-Safety Vest"]) > 0
            else 0
        )
        mask_compliance = (
            total_stats["Mask"] / (total_stats["Mask"] + total_stats["NO-Mask"])
            if (total_stats["Mask"] + total_stats["NO-Mask"]) > 0
            else 0
        )
        summary += "Compliance rates:\n"
        summary += f"\n• Hardhat compliance: {hardhat_compliance:.2%} ({total_stats['Hardhat']} out of {total_stats['Hardhat'] + total_stats['NO-Hardhat']} detections)"
        summary += f"\n• Safety Vest compliance: {vest_compliance:.2%} ({total_stats['Safety Vest']} out of {total_stats['Safety Vest'] + total_stats['NO-Safety Vest']} detections)"
        summary += f"\n• Mask compliance: {mask_compliance:.2%} ({total_stats['Mask']} out of {total_stats['Mask'] + total_stats['NO-Mask']} detections)"
        overall_compliance = (
            hardhat_compliance + vest_compliance + mask_compliance
        ) / 3
        summary += f"\n\nOverall PPE compliance: {overall_compliance:.2%}\n"
        summary += "\nRecommendations:\n"
        if overall_compliance < 0.8:
            summary += f"\n• Critical: Immediate action required. {total_stats['NO-Hardhat'] + total_stats['NO-Safety Vest'] + total_stats['NO-Mask']} PPE violations detected."
            summary += "\n• Conduct an emergency safety briefing."
            summary += "\n• Increase on-site safety inspections."
        elif overall_compliance < 0.95:
            summary += f"\n• Warning: Improvement needed. {total_stats['NO-Hardhat'] + total_stats['NO-Safety Vest'] + total_stats['NO-Mask']} PPE violations detected."
            summary += "\n• Reinforce PPE policies through team meetings."
            summary += "\n• Consider additional PPE training sessions."
        else:
            summary += (
                "\n• Good compliance observed. Maintain current safety protocols."
            )
            summary += "\n• Continue regular safety reminders and training."
    else:
        summary += "\n• No people detected in the observed period."
        summary += "\n• Check camera positioning and functionality."

    return summary


# --- Inference process target (runs in separate process) ---


def _inference_process_target(
    config_queue: Queue,
    results_queue: Queue,
    stop_event: Event,
    frame_ready_event: Event,
    inference_ready_event: Event,
    rebuffer_ack_event: Event,
):
    """
    Long-lived inference process: attaches shared memory, runs OVMS + DeepSORT.

    Control messages on ``config_queue`` (picklable dicts):

    - INIT_SHM: attach/replace shared buffer, load pipeline for ``config_id``.
    - RELOAD_CONFIG: same buffer dimensions; rebuild Runtime + tracker for ``config_id``.
    """
    shm = None
    shm_h = shm_w = 0
    runtime = None
    tracker = None
    # Mirrors main-process _stream_epoch for results tagging (drops stale after switch).
    pipeline_epoch = -1

    description_buffer: list = []
    frame_count = 0
    person_history: dict = {}
    person_last_state: dict = {}
    person_observations: list = []
    last_seen_update_interval = 30
    frames_since_last_seen_update = 0
    latest_summary = ""
    include_in_counts_by_class_id: dict[int, bool] = {}
    trackable_by_class_id: dict[int, bool] = {}
    detection_class_name_to_id: dict[str, int] = {}

    db_writer = DbWriterThread()
    db_writer.start()

    def reset_tracking_state() -> None:
        nonlocal description_buffer, frame_count, person_history, person_last_state
        nonlocal person_observations, frames_since_last_seen_update, latest_summary
        description_buffer = []
        frame_count = 0
        person_history = {}
        person_last_state = {}
        person_observations = []
        frames_since_last_seen_update = 0
        latest_summary = ""

    def build_pipeline(active_config_id) -> None:
        """Load DB config, create Runtime + DeepSORT. Raises ValueError on bad config."""
        nonlocal runtime, tracker, include_in_counts_by_class_id
        nonlocal trackable_by_class_id, detection_class_name_to_id
        from deep_sort_realtime.deepsort_tracker import DeepSort

        config = None
        if active_config_id is not None:
            try:
                cid = int(active_config_id)
                config = get_config_by_id(cid)
                log.info("Inference process: using config id=%s", cid)
            except (ValueError, TypeError):
                pass
        if not config:
            configs = get_all_configs()
            if configs:
                config = get_config_by_id(configs[0]["id"])
                log.info("Inference process: using first config id=%s", config["id"])
        if not config:
            raise ValueError(
                "No config available. Add a config via the Config dialog first."
            )
        (
            classes,
            include_in_counts_by_class_id,
            trackable_by_class_id,
            detection_class_name_to_id,
        ) = get_detection_classes_pipeline_maps(config["id"])
        if not classes:
            raise ValueError(
                "No detection classes available. Add a config via the Config dialog first."
            )
        model_url = (config.get("model_url") or "").strip()
        if not model_url:
            raise ValueError(
                "Config has no inferencing URL (model_url). "
                "Edit the config and set the inferencing URL."
            )
        model_name = (config.get("model_name") or "").strip() or "ppe"
        log.info(
            f"Inference process: creating Runtime config_id={config['id']} "
            f"service_url={model_url} model_name={model_name} input_name=env"
        )
        runtime = Runtime(
            classes=classes,
            service_url=model_url,
            model_name=model_name,
        )
        tracker = DeepSort(max_age=30, n_init=3)

    def handle_init_shm(msg: dict) -> None:
        nonlocal shm, shm_h, shm_w, runtime, tracker, pipeline_epoch
        shm_name = msg["shm_name"]
        nh, nw = int(msg["h"]), int(msg["w"])
        cid = msg.get("config_id")
        try:
            pipeline_epoch = int(msg["epoch"])
        except (TypeError, KeyError, ValueError):
            pipeline_epoch = -1
        try:
            if shm is not None:
                try:
                    shm.close()
                except Exception:
                    pass
                shm = None
            log.info(
                "Inference: INIT_SHM attach name=%s size=%dx%d config_id=%s",
                shm_name,
                nh,
                nw,
                cid,
            )
            shm = SharedMemory(name=shm_name)
            shm_h, shm_w = nh, nw
            inference_ready_event.clear()
            runtime = None
            tracker = None
            try:
                build_pipeline(cid)
                reset_tracking_state()
            except ValueError as e:
                log.error("Inference process: config/init failed: %s", e)
                try:
                    results_queue.put(([], str(e), "", pipeline_epoch), timeout=2.0)
                except Exception:
                    pass
            except Exception as e:
                log.exception("Inference process init failed: %s", e)
                try:
                    results_queue.put(
                        ([], f"Error: {e}", "", pipeline_epoch), timeout=2.0
                    )
                except Exception:
                    pass
            else:
                inference_ready_event.set()
                log.info("Inference process: ready after INIT_SHM")
        finally:
            rebuffer_ack_event.set()

    def handle_reload_config(msg: dict) -> None:
        nonlocal runtime, tracker, pipeline_epoch
        cid = msg.get("config_id")
        try:
            pipeline_epoch = int(msg["epoch"])
        except (TypeError, KeyError, ValueError):
            pipeline_epoch = -1
        log.info("Inference: RELOAD_CONFIG config_id=%s", cid)
        inference_ready_event.clear()
        runtime = None
        tracker = None
        try:
            build_pipeline(cid)
            reset_tracking_state()
        except ValueError as e:
            log.error("Inference process: RELOAD_CONFIG failed: %s", e)
            try:
                results_queue.put(([], str(e), "", pipeline_epoch), timeout=2.0)
            except Exception:
                pass
        except Exception as e:
            log.exception("Inference process: RELOAD_CONFIG failed: %s", e)
            try:
                results_queue.put(([], f"Error: {e}", "", pipeline_epoch), timeout=2.0)
            except Exception:
                pass
        else:
            inference_ready_event.set()
            log.info("Inference process: ready after RELOAD_CONFIG")

    def process_control_message(msg: dict) -> None:
        kind = msg.get("kind")
        if kind == CONFIG_MSG_INIT_SHM:
            handle_init_shm(msg)
        elif kind == CONFIG_MSG_RELOAD_CONFIG:
            handle_reload_config(msg)
        else:
            log.warning("Inference: unknown config_queue message kind=%s", kind)

    try:
        log.info("Inference process: waiting for first INIT_SHM...")
        first = None
        while True:
            first = config_queue.get(timeout=300)
            if stop_event.is_set() or first is None:
                return
            if isinstance(first, dict) and first.get("kind") == CONFIG_MSG_INIT_SHM:
                break
            log.warning("Inference: expected INIT_SHM first, got %s", first)
            return

        handle_init_shm(first)

        stale_ready_loops = 0
        while not stop_event.is_set():
            try:
                while True:
                    msg = config_queue.get_nowait()
                    process_control_message(msg)
            except queue.Empty:
                pass

            if shm is None or runtime is None or tracker is None:
                stale_ready_loops += 1
                time.sleep(0.02)
                continue
            stale_ready_loops = 0

            if not frame_ready_event.wait(timeout=0.15):
                continue

            buf = np.ndarray((shm_h, shm_w, 3), dtype=np.uint8, buffer=shm.buf)
            frame = buf.copy()
            frame_ready_event.clear()

            runtime_detections = runtime.run(frame)
            detections, counts, tracker_input_dets = process_detections(
                runtime_detections,
                include_in_counts_by_class_id,
                trackable_by_class_id,
            )

            tracked_target_boxes: dict[int, tuple[int, int, int, int]] = {}
            track_det_class: dict[int, str] = {}
            if tracker_input_dets:
                tracks = []
                try:
                    tracks = tracker.update_tracks(tracker_input_dets, frame=frame)
                except IndexError as e:
                    log.error(f"Inference: tracker IndexError: {e}", exc_info=True)
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    track_id = int(track.track_id)
                    ltrb = track.to_ltrb()
                    if ltrb is not None:
                        x1, y1, x2, y2 = map(int, ltrb)
                        tracked_target_boxes[track_id] = (x1, y1, x2, y2)
                    dc = track.get_det_class()
                    if dc:
                        track_det_class[track_id] = dc

            description = format_detection_description(
                counts, model_class_names_in_order(runtime.CLASSES)
            )
            description_buffer.append(description)
            if len(description_buffer) > 50:
                description_buffer.pop(0)

            for det in detections:
                if not trackable_by_class_id.get(det["class_id"], False):
                    continue
                dname = det["class_name"]
                for tid, pbox in tracked_target_boxes.items():
                    tcn = track_det_class.get(tid)
                    if tcn is not None and tcn != dname:
                        continue
                    if _boxes_overlap(det["bbox"], pbox):
                        det["track_id"] = tid
                        break

            now = datetime.now()
            frames_since_last_seen_update += 1
            do_last_seen_db_update = (
                frames_since_last_seen_update >= last_seen_update_interval
            )

            for track_id, person_bbox in tracked_target_boxes.items():
                if track_id not in person_history:
                    person_history[track_id] = {
                        "first_seen": now,
                        "last_seen": now,
                    }
                    tname = track_det_class.get(track_id)
                    dcid = detection_class_name_to_id.get(tname) if tname else None
                    if dcid is not None:
                        db_writer.enqueue(OP_INSERT_TRACK, (track_id, dcid, now, now))
                else:
                    person_history[track_id]["last_seen"] = now
                    if do_last_seen_db_update:
                        db_writer.enqueue(OP_UPDATE_LAST_SEEN, (now, track_id))

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
                    person_observations.append(
                        {
                            "track_id": track_id,
                            "timestamp": now,
                            "hardhat": ppe_status["hardhat"],
                            "vest": ppe_status["vest"],
                            "mask": ppe_status["mask"],
                            "bbox": person_bbox,
                        }
                    )
                    db_writer.enqueue(
                        OP_INSERT_OBSERVATION,
                        (track_id, now, json.dumps(attributes)),
                    )
                    person_last_state[track_id] = current_state

                if len(person_observations) > 1000:
                    person_observations = person_observations[-1000:]

            if do_last_seen_db_update:
                frames_since_last_seen_update = 0

            frame_count += 1
            if frame_count % 50 == 0:
                latest_summary = generate_summary(description_buffer)

            detections_clean = []
            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                item = {
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": float(d["confidence"]),
                    "class_id": int(d["class_id"]),
                    "class_name": str(d["class_name"]),
                }
                if d.get("track_id") is not None:
                    item["track_id"] = int(d["track_id"])
                detections_clean.append(item)

            try:
                results_queue.put(
                    (
                        detections_clean,
                        description,
                        latest_summary or "",
                        pipeline_epoch,
                    ),
                    timeout=2.0,
                )
            except queue.Full:
                log.warning("Inference: results queue full, dropping result")

    except Exception as e:
        log.exception("Inference process: crashed: %s", e)
        try:
            results_queue.put(
                ([], f"Inference error: {e}", "", pipeline_epoch), timeout=2.0
            )
        except Exception:
            pass
    finally:
        log.info("Inference process: exiting")
        db_writer.stop(timeout=3.0)
        if shm is not None:
            try:
                shm.close()
            except Exception:
                pass


class MultiModalAIDemo:
    """Core video analysis pipeline for detection, summaries, and chat context.

    Inference runs in a separate process to avoid GIL contention. Main process
    reads frames, puts them in shared memory when inference is ready, and
    consumes results for display.
    """

    def __init__(self, video_source=None):
        """Initialize the demo. video_source can be None; call start_streaming() when user selects a source."""
        self.video_source = video_source
        self.cap = None
        self._streaming_started = False
        self.class_names = [
            "Hardhat",
            "Mask",
            "NO-Hardhat",
            "NO-Mask",
            "NO-Safety Vest",
            "Person",
            "Safety Cone",
            "Safety Vest",
            "machinery",
            "vehicle",
        ]
        self.latest_detection = defaultdict(int)
        self.latest_summary = ""
        self.latest_tracked_persons = []

        # Frame reader thread: generation bumps on every stop so a session that is
        # still blocked in cap.read() cannot "come back to life" after we clear a
        # shared stop Event and start a new reader (fixes duplicate/zombie readers).
        self._reader_thread = None
        self._frame_reader_generation = 0
        self._latest_frame = None
        self._latest_frame_id = None
        # Epoch of frames in _latest_frame (see _stream_epoch); inference SHM skips mismatches.
        self._latest_frame_epoch = -1
        self._latest_frame_lock = threading.Lock()
        self._stream_epoch = 0
        self._reconnect_needed = False
        self._last_reconnect_warn_time = 0.0

        # Display state (updated by results consumer from inference process)
        self._display_lock = threading.Lock()
        self._display_detections = []
        self._display_description = ""
        self._display_summary = ""
        self._results_received_count = 0

        # Multiprocessing: inference process and IPC
        self._inference_process = None
        self._config_queue = None
        self._results_queue = None
        self._stop_event = None
        self._frame_ready_event = None
        self._inference_ready_event = None
        self._shm = None
        self._shm_h = self._shm_w = 0
        self._shm_initialized = False
        self._results_consumer_thread = None
        self._active_config_id = None
        self._s3_temp_path = None  # Temp file for S3 video; cleaned on switch/shutdown

    def setup_components(self):
        """Initialize DB, queues, and results consumer. Call start_streaming() when user selects a source."""
        # Initialize PostgreSQL (main process)
        init_database()
        log.info("PostgreSQL database initialized")

        # Create IPC queues and events
        self._config_queue = Queue(maxsize=1)
        self._results_queue = Queue(maxsize=1)
        self._stop_event = Event()
        self._frame_ready_event = Event()
        # Set by inference process when it enters main loop; main can check for startup timing
        self._inference_ready_event = Event()
        # Child sets this after handling INIT_SHM so main can unlink the previous segment safely.
        self._rebuffer_ack_event = Event()

        # Start results consumer thread
        self._results_consumer_thread = threading.Thread(
            target=self._results_consumer_loop,
            daemon=True,
        )
        self._results_consumer_thread.start()
        log.info("Results consumer thread started")

        # Register shutdown
        atexit.register(self._shutdown)

    def start_streaming(self, video_source: str, config_id: int | None = None):
        """Start or switch video source. Keeps the inference process alive when possible."""
        with self._latest_frame_lock:
            self._latest_frame = None
            self._latest_frame_id = None
            self._latest_frame_epoch = -1
        self._stop_frame_reader()

        # Reset stop_event so a newly spawned child does not exit on the first queued
        # INIT_SHM (stop can remain set after a prior _stop_inference_process).
        if self._stop_event is not None:
            self._stop_event.clear()

        # New switch generation: drop stale inference results and skip SHM for old frames.
        self._stream_epoch += 1
        _epoch = self._stream_epoch

        warm = (
            self._inference_process is not None and self._inference_process.is_alive()
        )

        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self._s3_temp_path:
            try:
                os.unlink(self._s3_temp_path)
            except OSError:
                pass
            self._s3_temp_path = None

        self.video_source = video_source
        self._reconnect_needed = False

        path_to_open, self._s3_temp_path = _resolve_video_source_to_path(video_source)

        with self._display_lock:
            self._display_description = "Processing video..."
            self._display_detections = []
            self._results_received_count = 0
            self._active_config_id = config_id

        self.cap = cv2.VideoCapture(path_to_open)
        if not self.cap.isOpened():
            log.warning("Stream not ready, retrying until connected...")
            attempt = 0
            while True:
                time.sleep(2)
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                self.cap = cv2.VideoCapture(path_to_open)
                if self.cap.isOpened():
                    log.info("Stream connected (attempt %d)", attempt + 1)
                    break
                attempt += 1
                if attempt % 5 == 0 and attempt > 0:
                    log.info("Still waiting for video stream (attempt %d)...", attempt)
        log.info("Video streaming from: %s", video_source)

        ok, probe = self.cap.read()
        h, w = 0, 0
        if ok and probe is not None:
            h, w = int(probe.shape[0]), int(probe.shape[1])
            with self._latest_frame_lock:
                self._latest_frame = probe.copy()
                self._latest_frame_id = 1
                self._latest_frame_epoch = self._stream_epoch

        rebuf_failed = False
        if warm:
            while True:
                try:
                    self._results_queue.get_nowait()
                except queue.Empty:
                    break
            while True:
                try:
                    self._config_queue.get_nowait()
                except queue.Empty:
                    break
            self._inference_ready_event.clear()
            self._frame_ready_event.clear()

            same_geo = (
                self._shm_initialized
                and h > 0
                and h == self._shm_h
                and w == self._shm_w
            )
            if same_geo:
                try:
                    self._config_queue.put(
                        {
                            "kind": CONFIG_MSG_RELOAD_CONFIG,
                            "config_id": config_id,
                            "epoch": _epoch,
                        },
                        timeout=60.0,
                    )
                except Exception as e:
                    log.exception("RELOAD_CONFIG put failed: %s", e)
                    rebuf_failed = True
            elif h <= 0 or w <= 0:
                log.warning("Warm switch: invalid frame size from probe; cold restart")
                rebuf_failed = True
            else:
                self._rebuffer_ack_event.clear()
                new_shm = None
                try:
                    new_shm = SharedMemory(create=True, size=h * w * 3)
                    self._config_queue.put(
                        {
                            "kind": CONFIG_MSG_INIT_SHM,
                            "shm_name": new_shm.name,
                            "h": h,
                            "w": w,
                            "config_id": config_id,
                            "epoch": _epoch,
                        },
                        timeout=60.0,
                    )
                except Exception as e:
                    log.exception("INIT_SHM (rebuffer) failed: %s", e)
                    rebuf_failed = True
                    if new_shm is not None:
                        try:
                            new_shm.close()
                            new_shm.unlink()
                        except Exception:
                            pass
                        new_shm = None
                if not rebuf_failed and new_shm is not None:
                    if not self._rebuffer_ack_event.wait(timeout=5.0):
                        log.error(
                            "Timeout waiting for inference rebuffer ack; restarting inference"
                        )
                        self._stop_inference_process(quick=True)
                        rebuf_failed = True
                        try:
                            new_shm.close()
                            new_shm.unlink()
                        except Exception:
                            pass
                    else:
                        if self._shm is not None:
                            try:
                                self._shm.close()
                                self._shm.unlink()
                            except Exception:
                                pass
                        self._shm = new_shm
                        self._shm_h, self._shm_w = h, w
                        self._shm_initialized = True

            if rebuf_failed:
                warm = False
                log.warning(
                    "start_streaming: warm reconfigure failed, will cold-restart inference"
                )

        if not warm:
            self._stop_inference_process(quick=True)
            while True:
                try:
                    self._config_queue.get_nowait()
                except queue.Empty:
                    break
            while True:
                try:
                    self._results_queue.get_nowait()
                except queue.Empty:
                    break
            self._shm_initialized = False
            self._frame_ready_event.clear()
            self._inference_ready_event.clear()
            if self._shm is not None:
                try:
                    self._shm.close()
                    self._shm.unlink()
                except Exception:
                    pass
                self._shm = None

            self._stop_event.clear()
            self._inference_process = Process(
                target=_inference_process_target,
                args=(
                    self._config_queue,
                    self._results_queue,
                    self._stop_event,
                    self._frame_ready_event,
                    self._inference_ready_event,
                    self._rebuffer_ack_event,
                ),
                daemon=True,
            )
            self._inference_process.start()
            log.info("Inference process spawned (waiting for first INIT_SHM)")

        self._start_frame_reader()
        self._streaming_started = True

    def stop_streaming_if_active_config(self, config_id: int) -> None:
        """If this config is the active stream, stop reader and inference and clear active id."""
        if self._active_config_id != config_id:
            return
        log.info(f"Stopping stream: active config {config_id} was deleted")
        self._stream_epoch += 1
        if self._results_queue is not None:
            while True:
                try:
                    self._results_queue.get_nowait()
                except queue.Empty:
                    break
        self._stop_frame_reader()
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        if self._s3_temp_path:
            try:
                os.unlink(self._s3_temp_path)
            except OSError:
                pass
            self._s3_temp_path = None
        self.video_source = None
        self._stop_inference_process(quick=True)
        if self._config_queue is not None:
            while True:
                try:
                    self._config_queue.get_nowait()
                except queue.Empty:
                    break
        self._shm_initialized = False
        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except Exception:
                pass
            self._shm = None
        with self._display_lock:
            self._active_config_id = None
            self._display_detections = []
            self._display_description = ""
            self._display_summary = ""
            self._results_received_count = 0
        self._streaming_started = False

    def _stop_inference_process(self, quick=False):
        """Stop the inference process if running.
        quick=True: short timeout then terminate (for source switching - don't wait for inference).
        quick=False: longer graceful shutdown (for app exit).
        """
        if self._inference_process is not None and self._inference_process.is_alive():
            self._stop_event.set()
            timeout = 0.2 if quick else 5.0
            self._inference_process.join(timeout=timeout)
            if self._inference_process.is_alive():
                self._inference_process.terminate()
                self._inference_process.join(timeout=1.0)
            self._inference_process = None

    def _shutdown(self):
        """Gracefully stop inference process on exit."""
        log.info("Shutting down: stopping inference process...")
        if self._stop_event is not None:
            self._stop_event.set()
        if self._inference_process is not None and self._inference_process.is_alive():
            self._inference_process.join(timeout=5.0)
            if self._inference_process.is_alive():
                self._inference_process.terminate()
            self._inference_process = None
        if self._s3_temp_path:
            try:
                os.unlink(self._s3_temp_path)
            except OSError:
                pass
            self._s3_temp_path = None
        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except Exception:
                pass

    def _results_consumer_loop(self):
        """Consume results from inference process and update display state."""
        try:
            while True:
                try:
                    result = self._results_queue.get(timeout=0.5)
                    if result is None:
                        break
                    if len(result) == 4:
                        detections, description, summary, res_epoch = result
                    else:
                        detections, description, summary = result
                        res_epoch = None
                    if res_epoch is not None and res_epoch != self._stream_epoch:
                        continue
                    with self._display_lock:
                        self._display_detections = list(detections)
                        self._display_description = description
                        self._display_summary = summary or self._display_summary
                        self._results_received_count += 1
                except queue.Empty:
                    pass  # Timeout, keep looping
                except Exception as e:
                    log.warning("Results consumer: failed to process result: %s", e)
        except Exception as e:
            log.exception("Results consumer: %s", e)
        finally:
            log.info("Results consumer: exiting")

    def format_detection_description(
        self, detections_class_count: dict[str, int]
    ) -> str:
        """Build a short description from counts (alphabetical order; no runtime classes in main process)."""
        return format_detection_description(
            detections_class_count,
            sorted(detections_class_count.keys()),
        )

    def append_description(self, description):
        """No-op in main process; inference process maintains its own buffer."""
        pass

    def generate_image_description(self, frame):
        """Legacy: not used when inference is in separate process."""
        return ""

    def generate_summary(self, descriptions):
        """Summarize PPE compliance over a list of detection descriptions."""
        return generate_summary(descriptions)

    def get_latest_detection(self):
        """Return the most recent detection counts."""
        return self.latest_detection

    def get_latest_summary(self):
        """Return the most recent summary."""
        with self._display_lock:
            return self._display_summary or self.latest_summary

    def get_latest_tracked_persons(self):
        """Return the most recent tracked persons with their PPE status."""
        return self.latest_tracked_persons

    def _frame_reader_loop(self, session: int):
        """Continuously read frames to consume stream; keeps only latest."""
        try:
            self._frame_reader_loop_impl(session)
        except Exception as e:
            log.exception("Frame reader: thread crashed: %s", e)

    def _is_file_source(self):
        s = (self.video_source or "").strip()
        return s.startswith("s3://")

    def _frame_reader_loop_impl(self, session: int):
        fail_count = 0
        read_count = 0
        frame_interval = 0.0  # 0 = no throttle (live stream)
        if self._is_file_source() and self.cap is not None:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 0:
                frame_interval = 1.0 / fps
                log.info(
                    "File source: throttling to %.2f fps (%.3fs per frame)",
                    fps,
                    frame_interval,
                )
        while self._frame_reader_generation == session:
            cap = self.cap
            if cap is None or not cap.isOpened():
                if not hasattr(self, "_last_reader_cap_warn") or (
                    time.time() - self._last_reader_cap_warn >= 3.0
                ):
                    log.warning(
                        "Frame reader: waiting for VideoCapture (session=%s)",
                        session,
                    )
                    self._last_reader_cap_warn = time.time()
                time.sleep(0.05)
                continue
            t0 = time.perf_counter() if frame_interval > 0 else 0
            success, frame = cap.read()
            if success and frame is not None:
                if fail_count > 0:
                    log.debug(
                        "Frame reader: recovered after %d consecutive failures",
                        fail_count,
                    )
                fail_count = 0
                read_count += 1
                if read_count % 10 == 0:
                    log.debug(
                        "Frame reader: received %d frames from stream",
                        read_count,
                    )
                with self._latest_frame_lock:
                    self._latest_frame = frame.copy()
                    self._latest_frame_id = read_count
                    self._latest_frame_epoch = self._stream_epoch
                if frame_interval > 0:
                    elapsed = time.perf_counter() - t0
                    sleep_time = frame_interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
            elif self._frame_reader_generation != session:
                break
            else:
                # cap.read() returned False - end of file or stream issue
                if self._is_file_source():
                    # For MP4 files: loop from start
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    fail_count = 0
                    log.debug("Frame reader: MP4 ended, looping from start")
                else:
                    fail_count += 1
                    if fail_count == 1:
                        log.warning(
                            "Frame reader: first consecutive failure (cap.read returned False)"
                        )
                    elif fail_count in (5, 10, 15, 20, 25):
                        log.debug("Frame reader: %d consecutive failures", fail_count)
                    if fail_count >= 30:
                        self._reconnect_needed = True
                        log.warning(
                            "Frame reader: 30 consecutive failures, setting _reconnect_needed"
                        )
                time.sleep(0.01)

    def _start_frame_reader(self):
        """Start background thread that consumes stream as fast as possible."""
        self._frame_reader_generation += 1
        session = self._frame_reader_generation
        self._reader_thread = threading.Thread(
            target=self._frame_reader_loop,
            args=(session,),
            daemon=True,
        )
        self._reader_thread.start()

    def _stop_frame_reader(self):
        """Stop frame reader thread before reconnect/release."""
        self._frame_reader_generation += 1
        prev_thread_still_alive = False
        if self._reader_thread is not None:
            t = self._reader_thread
            self._reader_thread = None
            t.join(timeout=15.0)
            prev_thread_still_alive = t.is_alive()
            if prev_thread_still_alive:
                log.error(
                    "Frame reader: previous thread still alive after 15s join "
                    "(may be stuck in cap.read); new session already active"
                )
        with self._latest_frame_lock:
            self._latest_frame = None
            self._latest_frame_id = None
            self._latest_frame_epoch = -1

    def _reconnect_stream(self):
        """Reconnect to live stream with retry and backoff."""
        self._stop_frame_reader()
        max_retries = 5
        base_delay = 2
        for attempt in range(max_retries):
            try:
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                    time.sleep(0.5)
                self.cap = cv2.VideoCapture(self.video_source)
                if self.cap.isOpened():
                    log.info(f"Reconnected to stream (attempt {attempt + 1})")
                    self._start_frame_reader()
                    return True
            except Exception as e:
                log.warning(f"Reconnect attempt {attempt + 1} failed: {e}")
            delay = base_delay * (2**attempt)
            time.sleep(delay)
        log.error("Failed to reconnect to stream after all retries")
        return False

    def get_frame_for_display(self, resize_to=None):
        """
        Get the latest frame and detections for display. Never blocks on inference.

        When shared memory queue is empty: put frame in buffer for inference.
        When shared memory queue is not empty: do not overwrite; just return frame for display.

        Returns (frame, detections, frame_id, frame_epoch). Frame may be None if unavailable.
        ``frame_epoch`` is the buffer epoch for this frame (for MJPEG dedup vs ``_stream_epoch``).
        """
        frame = None
        frame_id = None
        if self._reconnect_needed:
            now = time.time()
            if now - self._last_reconnect_warn_time >= 5.0:
                log.warning(
                    "get_frame_for_display: reconnect in progress (no frame yet)"
                )
                self._last_reconnect_warn_time = now
            return None, [], None, -1

        with self._latest_frame_lock:
            if self._latest_frame is not None:
                frame = self._latest_frame.copy()
                frame_id = self._latest_frame_id
                frame_epoch = self._latest_frame_epoch
            else:
                frame_epoch = -1

        if frame is None:
            return None, [], None, -1

        # Shared memory producer: only put when inference has consumed previous frame.
        # frame_ready_event: set=inference has frame, clear=slot empty. We put when clear.
        # (start_streaming clears it on restart so we always put first frame after switch.)
        frame_ready = self._frame_ready_event.is_set()
        if not frame_ready and frame_epoch == self._stream_epoch:
            h, w = frame.shape[:2]
            size = h * w * 3

            if not self._shm_initialized:
                # First frame: create shared buffer and send config to inference process
                try:
                    self._shm = SharedMemory(create=True, size=size)
                    self._shm_h, self._shm_w = h, w
                    self._config_queue.put(
                        {
                            "kind": CONFIG_MSG_INIT_SHM,
                            "shm_name": self._shm.name,
                            "h": h,
                            "w": w,
                            "config_id": self._active_config_id,
                            "epoch": self._stream_epoch,
                        }
                    )
                    self._shm_initialized = True
                    log.info("Shared memory buffer created %dx%d for inference", h, w)
                except Exception as e:
                    log.exception("Failed to create shared memory: %s", e)
                    return frame, list(self._display_detections), frame_id, frame_epoch

            # Only write if dimensions match (resolution unchanged)
            if h == self._shm_h and w == self._shm_w:
                buf = np.ndarray((h, w, 3), dtype=np.uint8, buffer=self._shm.buf)
                buf[:] = frame
                self._frame_ready_event.set()
            else:
                log.warning(
                    "get_frame_for_display: frame %dx%d != shm %dx%d — not writing SHM",
                    h,
                    w,
                    self._shm_h,
                    self._shm_w,
                )

        with self._display_lock:
            detections = list(self._display_detections)

        if resize_to:
            h, w = frame.shape[:2]
            tw, th = resize_to
            scale_x = tw / w
            scale_y = th / h
            frame = cv2.resize(frame, resize_to)
            scaled = []
            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                scaled.append(
                    {
                        **d,
                        "bbox": (
                            round(x1 * scale_x),
                            round(y1 * scale_y),
                            round(x2 * scale_x),
                            round(y2 * scale_y),
                        ),
                    }
                )
            detections = scaled

        return frame, detections, frame_id, frame_epoch

    def capture_and_update(self, resize_to=None, caller=None):
        """Legacy: not used when inference is in separate process."""
        return None, []

    def generate_frames(self):
        """Legacy: not used when inference is in separate process."""
        pass
