from __future__ import annotations

import logging
import os
import queue as queue_mod
import threading
import time

import tempfile

import cv2

from minio_client import download_file

log = logging.getLogger(__name__)


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


_MAX_CONSECUTIVE_FAILURES = 30
_RECONNECT_BACKOFF_S = 2.0


def _classify_source(video_source: str) -> str:
    s = (video_source or "").strip().lower()
    if s.startswith("rtsp://"):
        return "rtsp"
    if s.startswith("s3://"):
        return "s3"
    return "file"


class FrameConsumer:
    """Daemon thread that reads video frames and puts them into a queue.

    Supports RTSP streams, S3-hosted videos, and local file paths.
    Can be started in idle mode (``video_source=None``); call
    ``set_source()`` later to begin reading frames.

    The read loop varies by source type:
      - RTSP: no throttle, reconnect on sustained failures
      - file / S3: throttle to native FPS, loop on EOF
    """

    def __init__(
        self,
        video_source: str | None,
        frame_queue: queue_mod.Queue,
        stop_event: threading.Event,
    ) -> None:
        self._video_source = video_source
        self._queue = frame_queue
        self._stop = stop_event
        self._source_type = _classify_source(video_source) if video_source else "file"

        self._cap: cv2.VideoCapture | None = None
        self._resolved_path: str | None = video_source
        self._temp_path: str | None = None
        self._frame_id = 0
        self.frame_interval: float = 0.0

        self._source_ready = threading.Event()
        if video_source is not None:
            self._source_ready.set()

        self._thread = threading.Thread(
            target=self._run,
            name="frame-consumer",
            daemon=True,
        )

    def start(self) -> None:
        if self._video_source is not None:
            self._open_capture()
        self._thread.start()

    def set_source(self, video_source: str) -> None:
        """Switch to a new video source (or start from idle)."""
        self._release_capture()
        self._cleanup_temp()
        self._frame_id = 0

        self._video_source = video_source
        self._source_type = _classify_source(video_source)
        self._open_capture()
        self._source_ready.set()

    def make_idle(self) -> None:
        """Return to idle mode without killing the thread."""
        self._source_ready.clear()
        self._video_source = None

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._source_ready.set()
        self._thread.join(timeout=timeout)
        self._release_capture()
        self._cleanup_temp()

    def _open_capture(self) -> None:
        """Resolve the source path and open ``cv2.VideoCapture``."""
        if self._source_type == "s3":
            self._resolved_path, self._temp_path = _resolve_video_source_to_path(
                self._video_source
            )
        else:
            self._resolved_path = self._video_source

        self._cap = cv2.VideoCapture(self._resolved_path)
        if not self._cap.isOpened():
            log.error(f"FrameConsumer: failed to open source {self._video_source}")
            return

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0:
            self.frame_interval = 1.0 / fps

    def _release_capture(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _cleanup_temp(self) -> None:
        if self._temp_path is not None:
            try:
                os.unlink(self._temp_path)
            except OSError:
                pass
            self._temp_path = None

    def _put_frame(self, frame) -> None:
        self._frame_id += 1
        while not self._stop.is_set():
            try:
                self._queue.put((frame, self._frame_id), timeout=0.1)
                return
            except queue_mod.Full:
                continue

    def _run(self) -> None:
        log.info("FrameConsumer thread alive (waiting for source)")
        try:
            while not self._stop.is_set():
                if not self._source_ready.wait(timeout=0.5):
                    continue
                if self._stop.is_set():
                    break

                log.info(
                    f"FrameConsumer reading (source_type={self._source_type}, source={self._video_source})"
                )
                if self._source_type == "rtsp":
                    self._read_loop_rtsp()
                else:
                    self._read_loop_file()

                self._release_capture()
                self._cleanup_temp()
        except Exception:
            log.exception("FrameConsumer thread crashed")
        finally:
            self._release_capture()
            log.info("FrameConsumer thread exited")

    # ---- RTSP read loop ----

    def _read_loop_rtsp(self) -> None:
        fail_count = 0

        while not self._stop.is_set() and self._source_ready.is_set():
            cap = self._cap
            if cap is None or not cap.isOpened():
                log.warning("FrameConsumer: capture not open, attempting reconnect")
                self._reconnect()
                continue

            success, frame = cap.read()

            if success and frame is not None:
                if fail_count > 0:
                    log.debug(
                        f"FrameConsumer: recovered after {fail_count} consecutive failures"
                    )
                fail_count = 0
                self._put_frame(frame)
            else:
                fail_count += 1
                if fail_count == 1:
                    log.warning("FrameConsumer: first read failure (RTSP)")
                elif fail_count % 10 == 0:
                    log.debug(f"FrameConsumer: {fail_count} consecutive failures")
                if fail_count >= _MAX_CONSECUTIVE_FAILURES:
                    log.warning(f"FrameConsumer: {fail_count} failures, reconnecting")
                    self._reconnect()
                    fail_count = 0
                time.sleep(0.01)

    def _reconnect(self) -> None:
        self._release_capture()
        time.sleep(_RECONNECT_BACKOFF_S)
        self._cap = cv2.VideoCapture(self._resolved_path)
        if self._cap.isOpened():
            log.info(f"FrameConsumer: reconnected to {self._video_source}")
        else:
            log.warning(f"FrameConsumer: reconnect failed for {self._video_source}")

    # ---- File / S3 read loop ----

    def _read_loop_file(self) -> None:
        frame_interval = 0.0
        cap = self._cap
        if cap is not None:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 0:
                frame_interval = 1.0 / fps
                log.info(
                    f"File source: throttling to {fps:.2f} fps ({frame_interval:.3f}s per frame)"
                )

        while not self._stop.is_set() and self._source_ready.is_set():
            cap = self._cap
            if cap is None or not cap.isOpened():
                log.warning("FrameConsumer: capture not open (file), stopping")
                break

            t0 = time.perf_counter() if frame_interval > 0 else 0

            success, frame = cap.read()

            if success and frame is not None:
                self._put_frame(frame)
                if frame_interval > 0:
                    elapsed = time.perf_counter() - t0
                    sleep_time = frame_interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                log.debug("FrameConsumer: file ended, looping from start")
