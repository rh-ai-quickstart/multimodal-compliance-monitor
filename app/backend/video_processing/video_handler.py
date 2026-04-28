import atexit
import queue
import threading
import time
from collections import defaultdict

import cv2

from database import init_database, get_detection_classes_pipeline_maps
from logger import get_logger
from video_processing.consumer import FrameConsumer
from video_processing.inference import InferencePool
from video_processing.tracking import TrackerProcess

log = get_logger(__name__)


class VideoHandler:
    """Core video analysis pipeline for detection, summaries, and chat context.

    Inference runs in a pool of worker threads (each with its own Runtime) so
    multiple frames can be processed concurrently.  The heavy-lifting ops
    (gRPC I/O, cv2, numpy) release the GIL, giving true parallelism.
    """

    def __init__(self, video_source=None):
        """Initialize the demo. video_source can be None; call start_streaming() when user selects a source."""
        self.video_source = video_source
        self.cap = None
        self._streaming_started = False
        self.latest_detection = defaultdict(int)
        self.latest_summary = ""
        self.latest_description = ""
        self._display_lock = threading.Lock()

        self._frame_queue: queue.Queue = queue.Queue(maxsize=150)
        self._inference_out_queue: queue.Queue = queue.Queue(maxsize=350)
        self._stop_event = threading.Event()
        self._consumer: FrameConsumer | None = None
        self._inference_pool: InferencePool | None = None
        self._tracker: TrackerProcess | None = None
        self._active_config_id: int | None = None
        self._class_names_in_order: list[str] = []
        self._description_buffer: list[str] = []

        self.init_setup()

    def init_setup(self):
        init_database()
        log.info("PostgreSQL database initialized")

        self._consumer = FrameConsumer(None, self._frame_queue, self._stop_event)
        self._consumer.start()
        log.info("FrameConsumer thread started (idle)")

        self._inference_pool = InferencePool(
            self._frame_queue, self._inference_out_queue, self._stop_event
        )
        self._inference_pool.start()
        log.info("InferencePool started (idle)")

        self._tracker = TrackerProcess()
        self._tracker.start()
        log.info("TrackerProcess started (idle)")

        self._queue_monitor = threading.Thread(
            target=self._log_queue_sizes,
            name="queue-monitor",
            daemon=True,
        )
        self._queue_monitor.start()

        atexit.register(self._shutdown)

    def _log_queue_sizes(self) -> None:
        while not self._stop_event.wait(timeout=2.0):
            log.info(
                "queue sizes: frame_queue=%d/%d inference_out_queue=%d/%d",
                self._frame_queue.qsize(),
                self._frame_queue.maxsize,
                self._inference_out_queue.qsize(),
                self._inference_out_queue.maxsize,
            )

    def _shutdown(self):
        log.info("Shutting down VideoHandler")
        self._stop_event.set()
        if self._tracker is not None:
            self._tracker.stop()
        if self._inference_pool is not None:
            self._inference_pool.stop()
        if self._consumer is not None:
            self._consumer.stop()

    def start_streaming(self, video_source: str, config_id: int):
        self._description_buffer.clear()
        self._stop_event.clear()
        self.video_source = video_source
        self._active_config_id = config_id
        classes, include_in_counts, trackable, name_to_id = (
            get_detection_classes_pipeline_maps(config_id)
        )
        self._class_names_in_order = [classes[i] for i in sorted(classes)]
        self._inference_pool.configure(config_id)
        self._tracker.configure(
            trackable_by_class_id=trackable,
            detection_class_name_to_id=name_to_id,
        )
        self._consumer.set_source(video_source)
        self._streaming_started = True
        log.info(f"Streaming started for source={video_source} config_id={config_id}")

    def stop_streaming(self):
        if self._consumer is not None:
            self._consumer.make_idle()

        for q in (self._frame_queue, self._inference_out_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        if self._tracker is not None:
            self._tracker.reset()

        self._description_buffer.clear()
        self._active_config_id = None
        self._streaming_started = False
        log.info("Streaming stopped")

    def stop_streaming_if_active_config(self, config_id: int):
        if self._active_config_id != config_id:
            return
        log.info(f"Stopping stream: active config {config_id} was deleted")
        self.stop_streaming()

    def switch_video_source(self, new_video_source: str, new_config_id: int):
        self.stop_streaming()

        self.start_streaming(new_video_source, new_config_id)

    def get_frame_and_detection_from_inference(self):
        """Block until an inference result is ready and return it."""
        while not self._stop_event.is_set():
            try:
                return self._inference_out_queue.get(timeout=0.001)
            except queue.Empty:
                continue
        return None

    def _format_detection_description(
        self,
        detections_class_count: dict[str, int],
        class_names_in_order: list[str],
    ) -> str:
        """Build a short, human-readable description from counts; order follows config/model indices."""
        description = "Detected: "
        for item in class_names_in_order:
            if detections_class_count.get(item, 0) > 0:
                description += f"{item}: {detections_class_count[item]}, "
        return description.rstrip(", ")

    def _generate_summary(self, descriptions: list) -> str:
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

    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels for each detection onto *frame* (mutated in-place).

        Colors: cyan = tracked target, green = compliant PPE, red = non-compliant, yellow = other.
        Detections below VIDEO_FEED_DRAW_MIN_CONF are skipped.
        """
        VIDEO_FEED_DRAW_MIN_CONF = 0.5

        h_frame, w_frame = frame.shape[:2]
        line_type = cv2.LINE_AA
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1 = max(0, min(x1, w_frame - 1))
            y1 = max(0, min(y1, h_frame - 1))
            x2 = max(0, min(x2, w_frame - 1))
            y2 = max(0, min(y2, h_frame - 1))
            if x1 >= x2 or y1 >= y2:
                continue
            conf = detection["confidence"]
            currentClass = detection["class_name"]
            if conf <= VIDEO_FEED_DRAW_MIN_CONF:
                continue
            if detection.get("track_id") is not None:
                color = (0, 255, 255)  # Cyan for tracked targets
            elif currentClass in ["NO-Hardhat", "NO-Safety Vest", "NO-Mask"]:
                color = (0, 0, 255)  # Red for non-compliance
            elif currentClass in ["Hardhat", "Safety Vest", "Mask"]:
                color = (0, 255, 0)  # Green for compliance
            else:
                color = (255, 255, 0)  # Yellow for other objects
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2,
                lineType=line_type,
            )
            label = f"{currentClass} {conf:.2f}"
            if detection.get("track_id") is not None:
                label = f"{currentClass} #{detection['track_id']} {conf:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            label_y1 = max(0, y1 - text_size[1] - 10)
            label_y2 = y1
            label_x2 = min(w_frame, x1 + text_size[0])
            if label_x2 > x1 and label_y2 > label_y1:
                cv2.rectangle(
                    frame,
                    (x1, label_y1),
                    (label_x2, label_y2),
                    color,
                    -1,
                    lineType=line_type,
                )
            text_y = max(label_y1 + text_size[1] - 2, y1 - 5)
            cv2.putText(
                frame,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 0),
                2,
                lineType=line_type,
            )
        return frame

    def encode_mjpeg_chunk(self, frame, quality=95):
        """Encode *frame* as JPEG and wrap it in a multipart MJPEG chunk.

        Returns the chunk bytes ready to yield, or *None* if encoding fails.
        """
        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ret:
            log.warning(
                f"Video feed: cv2.imencode failed shape={getattr(frame, 'shape', None)}"
            )
            return None
        frame_bytes = buffer.tobytes()
        header = (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n"
        )
        return header + frame_bytes + b"\r\n"

    def frame_generator(self):
        time.sleep(1)
        frame_interval = self._consumer.frame_interval
        last_yield_time = time.perf_counter()
        try:
            while True:
                result = self.get_frame_and_detection_from_inference()
                if result is None:
                    continue

                self.latest_description = self._format_detection_description(
                    result.counts, self._class_names_in_order
                )
                self._description_buffer.append(self.latest_description)
                if len(self._description_buffer) > 50:
                    self._description_buffer.pop(0)
                self.latest_summary = self._generate_summary(self._description_buffer)

                frame = self.draw_detections(result.frame, result.detections)
                chunk = self.encode_mjpeg_chunk(frame)
                if chunk is None:
                    continue

                self._tracker.submit(result.detections)

                elapsed = time.perf_counter() - last_yield_time
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                # log.info(
                #     f"Time elapsed: {elapsed * 1000:.2f} ms, Sleeping for {sleep_time * 1000:.2f} ms"
                # )

                last_yield_time = time.perf_counter()

                try:
                    yield_start = time.perf_counter()
                    yield chunk
                    yield_end = time.perf_counter()
                    log.debug(f"Yield took {(yield_end - yield_start) * 1000:.2f} ms")
                except (BrokenPipeError, ConnectionResetError, OSError) as e:
                    log.warning(f"Video feed: client disconnected during yield: {e}")
                    break

        except Exception as e:
            log.exception(f"Video feed: exception in stream loop: {e}")

    def get_latested_description(self):
        """Return the most recent description."""
        with self._display_lock:
            return self.latest_description

    def get_latest_summary(self):
        """Return the most recent summary."""
        with self._display_lock:
            return self.latest_summary
