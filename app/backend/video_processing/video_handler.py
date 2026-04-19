import cv2
import threading
import atexit

# from typing import Self
# import json
# import numpy as np
# import os
# import tempfile
# import time
# from datetime import datetime
# from multiprocessing import Process, Queue, Event
# from multiprocessing.shared_memory import SharedMemory
from collections import defaultdict
import queue

# from response import process_detections
# from runtime import Runtime
from database import init_database
from logger import get_logger
from video_processing.cunsumer import FrameConsumer

log = get_logger(__name__)

# config_queue message kinds (multiprocessing — must be picklable dicts)
CONFIG_MSG_INIT_SHM = "INIT_SHM"
CONFIG_MSG_RELOAD_CONFIG = "RELOAD_CONFIG"


class VideoHandler:
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
        self.latest_detection = defaultdict(int)
        self.latest_summary = ""
        self.latest_description = ""
        self._display_lock = threading.Lock()

        self._frame_queue: queue.Queue = queue.Queue(maxsize=30)
        self._stop_event = threading.Event()
        self._consumer: FrameConsumer | None = None
        self._active_config_id: int | None = None

        self.init_setup()

    def init_setup(self):
        init_database()
        log.info("PostgreSQL database initialized")

        self._consumer = FrameConsumer(None, self._frame_queue, self._stop_event)
        self._consumer.start()
        log.info("FrameConsumer thread started (idle)")

        # TODO: Start Deepsort process
        # TODO: Start inference / results consumer thread

        atexit.register(self._shutdown)

    def _shutdown(self):
        log.info("Shutting down MultiModalAIDemo2")
        self._stop_event.set()
        if self._consumer is not None:
            self._consumer.stop()

    def start_streaming(self, video_source: str, config_id: int):
        self._stop_event.clear()
        self.video_source = video_source
        self._active_config_id = config_id
        self._consumer.set_source(video_source)
        self._streaming_started = True
        log.info(f"Streaming started for source={video_source} config_id={config_id}")

    def stop_streaming(self):
        if self._consumer is not None:
            self._consumer.make_idle()

        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

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

    def get_next_frame(self):
        """Takes frame from the frame buffer which is fed by the consumer thread."""
        try:
            return self._frame_queue.get(timeout=1.0)
        except queue.Empty:
            return None, -1

    def put_frame_for_inference(self, frame):
        "put the frame inside the buffer for bouning box detection the inference thread takes from this buffer"

    def put_detections_for_tracks(self, detections):
        "given the detection put the detections in the buffer for process deepsort to return tracks"

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
        try:
            while True:
                frame, frame_id = self.get_next_frame()
                if frame is None:
                    continue

                # detections = self.put_frame_for_inference(frame)

                # tracks = self.put_detections_for_tracks(detections)

                # annotated_frame = self.draw_detections(frame, tracks, detections)

                # # Update detections and safty treand (which is summary)
                # with self._display_lock:
                #     self.latest_description = TODO
                #     self.latest_summary = TODO

                chunk = self.encode_mjpeg_chunk(frame)
                if chunk is None:
                    continue

                try:
                    yield chunk
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
