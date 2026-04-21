from __future__ import annotations

import dataclasses
import logging
import os
import queue as queue_mod
import threading
import time
from collections import defaultdict

import numpy as np

from database import (
    get_config_by_id,
    get_all_configs,
    get_detection_classes_pipeline_maps,
)
from response import process_detections
from runtime import Runtime

log = logging.getLogger(__name__)

_MSG_CONFIGURE = "_CONFIGURE"

_DEFAULT_WORKERS = 2
_DEFAULT_MIN_BATCH = 3
_DEFAULT_MAX_BATCH = 3


@dataclasses.dataclass
class InferenceResult:
    frame: np.ndarray
    frame_id: int
    detections: list[dict]
    counts: defaultdict
    tracker_input_dets: list


@dataclasses.dataclass(frozen=True)
class _PipelineConfig:
    """Immutable snapshot of everything a worker needs to build its own Runtime."""

    classes: dict[int, str]
    model_url: str
    model_name: str
    include_in_counts: dict[int, bool]
    trackable: dict[int, bool]
    version: int


class InferencePool:
    """Pool of daemon threads that run OVMS inference on frames from an input queue.

    Each worker owns its own ``Runtime`` instance so preprocessing buffers and
    gRPC stubs are never shared.  The pool sits idle until ``configure`` is
    called with a valid config_id.
    """

    def __init__(
        self,
        in_queue: queue_mod.Queue,
        out_queue: queue_mod.Queue,
        stop_event: threading.Event,
        num_workers: int | None = None,
    ) -> None:
        self._in_queue = in_queue
        self._out_queue = out_queue
        self._stop = stop_event

        if num_workers is None:
            num_workers = int(os.environ.get("INFERENCE_WORKERS", _DEFAULT_WORKERS))
        self._num_workers = max(1, num_workers)

        self._min_batch = max(
            1, int(os.environ.get("INFERENCE_MIN_BATCH", _DEFAULT_MIN_BATCH))
        )
        self._max_batch = max(
            self._min_batch,
            int(os.environ.get("INFERENCE_MAX_BATCH", _DEFAULT_MAX_BATCH)),
        )

        self._config: _PipelineConfig | None = None
        self._config_lock = threading.Lock()

        self._batch_lock = threading.Lock()

        self._reorder_lock = threading.Lock()
        self._pending: dict[int, InferenceResult] = {}
        self._next_frame_id: int = 1

        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        for i in range(self._num_workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"inference-worker-{i}",
                daemon=True,
            )
            self._threads.append(t)
            t.start()
        log.info(
            "InferencePool started with %d workers (batch min=%d max=%d)",
            self._num_workers,
            self._min_batch,
            self._max_batch,
        )

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        per_thread = timeout / max(len(self._threads), 1)
        for t in self._threads:
            t.join(timeout=per_thread)

    def configure(self, config_id: int) -> None:
        """Send a configure message through the queue so a worker rebuilds the shared pipeline config."""
        self._in_queue.put({"kind": _MSG_CONFIGURE, "config_id": config_id})

    def _build_pipeline_config(self, config_id: int) -> None:
        """Query the DB and store a new ``_PipelineConfig``.  Called under ``_config_lock``."""
        config = None
        if config_id is not None:
            try:
                config = get_config_by_id(int(config_id))
                log.info("InferencePool: using config id=%s", config_id)
            except (ValueError, TypeError):
                pass
        if not config:
            configs = get_all_configs()
            if configs:
                config = get_config_by_id(configs[0]["id"])
                log.info(
                    "InferencePool: falling back to first config id=%s", config["id"]
                )
        if not config:
            log.error("InferencePool: no config available, staying idle")
            self._config = None
            return

        (
            classes,
            include_in_counts,
            trackable,
            _name_to_id,
        ) = get_detection_classes_pipeline_maps(config["id"])

        if not classes:
            log.error("InferencePool: no detection classes for config %s", config["id"])
            self._config = None
            return

        model_url = (config.get("model_url") or "").strip()
        if not model_url:
            log.error("InferencePool: config %s has no model_url", config["id"])
            self._config = None
            return

        model_name = (config.get("model_name") or "").strip() or "ppe"

        prev_version = self._config.version if self._config else 0
        self._config = _PipelineConfig(
            classes=classes,
            model_url=model_url,
            model_name=model_name,
            include_in_counts=include_in_counts,
            trackable=trackable,
            version=prev_version + 1,
        )
        log.info(
            f"InferencePool: pipeline config v{self._config.version} "
            f"config_id={config['id']} service_url={model_url} model_name={model_name}"
        )

    def _handle_configure(self, config_id: int, name: str) -> None:
        with self._config_lock:
            self._build_pipeline_config(config_id)
        with self._reorder_lock:
            self._pending.clear()
            self._next_frame_id = 1

    def _push_result(self, result: InferenceResult) -> None:
        with self._reorder_lock:
            self._pending[result.frame_id] = result
            self._flush_pending()

    def _flush_pending(self) -> None:
        """Emit consecutive results from the reorder buffer to ``_out_queue``."""
        while self._next_frame_id in self._pending:
            result = self._pending.pop(self._next_frame_id)
            try:
                self._out_queue.put_nowait(result)
            except queue_mod.Full:
                try:
                    self._out_queue.get_nowait()
                except queue_mod.Empty:
                    pass
                try:
                    self._out_queue.put_nowait(result)
                except queue_mod.Full:
                    pass
            self._next_frame_id += 1

    def _worker_loop(self) -> None:
        name = threading.current_thread().name
        log.info("%s started (idle, waiting for configure)", name)

        local_runtime: Runtime | None = None
        local_config_version = 0
        min_batch = self._min_batch
        max_batch = self._max_batch

        try:
            while not self._stop.is_set():
                batch: list[tuple[np.ndarray, int]] = []

                with self._batch_lock:
                    # -- accumulate phase: collect at least min_batch frames --
                    while len(batch) < min_batch and not self._stop.is_set():
                        try:
                            item = self._in_queue.get(timeout=0.05)
                        except queue_mod.Empty:
                            continue

                        if (
                            isinstance(item, dict)
                            and item.get("kind") == _MSG_CONFIGURE
                        ):
                            self._handle_configure(item["config_id"], name)
                            local_runtime = None
                            local_config_version = 0
                            batch.clear()
                            continue

                        if self._config is None:
                            continue

                        batch.append(item)

                    # -- drain phase: greedily grab more up to max_batch --
                    if not self._stop.is_set() and batch:
                        while len(batch) < max_batch:
                            try:
                                item = self._in_queue.get_nowait()
                            except queue_mod.Empty:
                                break

                            if (
                                isinstance(item, dict)
                                and item.get("kind") == _MSG_CONFIGURE
                            ):
                                self._in_queue.put(item)
                                break

                            batch.append(item)

                if self._stop.is_set() or not batch:
                    continue

                # -- ensure runtime is up-to-date --
                cfg = self._config
                if cfg is None:
                    continue

                if local_config_version != cfg.version:
                    try:
                        local_runtime = Runtime(
                            classes=cfg.classes,
                            service_url=cfg.model_url,
                            model_name=cfg.model_name,
                        )
                        local_config_version = cfg.version
                        log.info("%s: Runtime created (config v%d)", name, cfg.version)
                    except Exception:
                        log.exception("%s: failed to create Runtime", name)
                        local_runtime = None
                        continue

                if local_runtime is None:
                    continue

                # -- batched inference --
                t0 = time.perf_counter()
                frames = [f for f, _ in batch]
                frame_ids = [fid for _, fid in batch]

                try:
                    all_detections = local_runtime.run_batch(frames)
                except Exception:
                    log.exception(
                        "%s: inference failed (batch=%d frames=[%d..%d]), dropping batch",
                        name,
                        len(batch),
                        frame_ids[0],
                        frame_ids[-1],
                    )
                    continue

                for i, dets_raw in enumerate(all_detections):
                    detections, counts, tracker_input_dets = process_detections(
                        dets_raw,
                        cfg.include_in_counts,
                        cfg.trackable,
                    )
                    self._push_result(
                        InferenceResult(
                            frame=frames[i],
                            frame_id=frame_ids[i],
                            detections=detections,
                            counts=counts,
                            tracker_input_dets=tracker_input_dets,
                        )
                    )

                elapsed = time.perf_counter() - t0
                fps = len(batch) / elapsed if elapsed > 0 else float("inf")
                log.info(
                    "%s batch=%d frames=[%d..%d] elapsed=%.4fs fps=%.1f",
                    name,
                    len(batch),
                    frame_ids[0],
                    frame_ids[-1],
                    elapsed,
                    fps,
                )

        except Exception:
            log.exception("%s crashed", name)
        finally:
            log.info("%s exited", name)
