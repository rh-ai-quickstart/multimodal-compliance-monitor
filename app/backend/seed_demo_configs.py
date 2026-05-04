"""Idempotent demo seed: four app_config rows — bird, ppe (MP4), traffic, ppe (RTSP live).

Copies sample MP4s from the data bucket into CONFIG_BUCKET at uploads/<basename>,
using the same filenames as in the data bucket (unique per demo). The RTSP row has no
MinIO copy or thumbnail; it appears in the Source RTSP dropdown.
"""

from __future__ import annotations

import os
import time

from logger import get_logger
from database import (
    insert_config,
    replace_detection_classes,
)
from minio_client import (
    copy_object,
    get_config_bucket,
    object_exists,
    get_minio_client,
)
from minio.error import S3Error
from thumbnail_utils import generate_thumbnail_for_video_source

log = get_logger(__name__)

# Ultralytics YOLOv8 COCO pretrained class order (nc=80).
_COCO80_LINES = """
person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
couch
potted plant
bed
dining table
toilet
tv
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
book
clock
vase
scissors
teddy bear
hair drier
toothbrush
"""
COCO80: tuple[str, ...] = tuple(
    line.strip() for line in _COCO80_LINES.strip().splitlines() if line.strip()
)


def _traffic_class_entries() -> list[tuple[int, str, bool, bool]]:
    track_idx = {2, 3, 5, 7}  # car, motorcycle, bus, truck
    return [(i, COCO80[i], i in track_idx, True) for i in range(len(COCO80))]


def _bird_class_entries() -> list[tuple[int, str, bool, bool]]:
    return [
        (0, "Bluejays", True, True),
        (1, "Cardinals", True, True),
    ]


def _ppe_class_entries() -> list[tuple[int, str, bool, bool]]:
    return [
        (0, "Hardhat", False, True),
        (1, "Mask", False, True),
        (2, "NO-Hardhat", False, True),
        (3, "NO-Mask", False, True),
        (4, "NO-Safety Vest", False, True),
        (5, "Person", True, True),
        (6, "Safety Cone", False, False),
        (7, "Safety Vest", False, True),
        (8, "machinery", False, False),
        (9, "vehicle", False, False),
    ]


def _default_model_url() -> str:
    explicit = (os.getenv("DEFAULT_OVMS_MODEL_URL") or "").strip()
    if explicit:
        return explicit
    openshift = os.getenv("OPENSHIFT", "false").lower() == "true"
    if openshift:
        return "http://ppe-predictor:9000"
    return "ovms:8081"


def _default_rtsp_live_url() -> str:
    """In-cluster MediaMTX path used by video-stream; OpenShift vs compose hostnames differ."""
    openshift = os.getenv("OPENSHIFT", "false").lower() == "true"
    if openshift:
        return "rtsp://ppe-compliance-monitor-ppe-compliance-monitor-video-stream:8554/live"
    return "rtsp://video-stream:8554/live"


def _ensure_object_with_retry(
    dest_bucket: str,
    dest_key: str,
    src_bucket: str,
    src_key: str,
    max_retries: int = 12,
    delay_s: float = 2.0,
) -> None:
    if object_exists(dest_bucket, dest_key):
        log.debug(f"Seed object already present: {dest_bucket}/{dest_key}")
        return
    for attempt in range(max_retries):
        try:
            if not object_exists(src_bucket, src_key):
                log.warning(
                    f"Seed source not ready s3://{src_bucket}/{src_key} "
                    f"(attempt {attempt + 1}/{max_retries}); retrying..."
                )
                time.sleep(delay_s)
                continue
            copy_object(dest_bucket, dest_key, src_bucket, src_key)
            return
        except S3Error as e:
            log.warning(
                f"Seed copy failed (attempt {attempt + 1}/{max_retries}): {e}; retrying..."
            )
            time.sleep(delay_s)
    raise RuntimeError(
        f"Could not copy s3://{src_bucket}/{src_key} to "
        f"s3://{dest_bucket}/{dest_key} after {max_retries} attempts"
    )


def _ping_minio(max_attempts: int = 15, delay_s: float = 2.0) -> None:
    for attempt in range(max_attempts):
        try:
            client = get_minio_client()
            client.list_buckets()
            return
        except Exception as e:
            if attempt == max_attempts - 1:
                raise RuntimeError(
                    f"MinIO not reachable after {max_attempts} attempts: {e}"
                ) from e
            log.warning(
                f"MinIO not ready (attempt {attempt + 1}/{max_attempts}): {e}; retrying..."
            )
            time.sleep(delay_s)


def insert_demo_configs() -> None:
    """Insert demo app_config rows; caller should invoke only when DB has no configs yet."""
    data_bucket = os.getenv("MINIO_VIDEO_BUCKET", "data").strip() or "data"
    cfg_bucket = get_config_bucket()
    model_url = _default_model_url()
    # Each tuple: (served OVMS/Triton model id, video filename in data bucket, class entries).

    demos: list[tuple[str, str, list[tuple[int, str, bool, bool]]]] = [
        ("bird", "bluejayclear.mp4", _bird_class_entries()),
        ("ppe", "combined-video-no-gap-rooftop.mp4", _ppe_class_entries()),
        ("yolov8n", "cars.mp4", _traffic_class_entries()),
    ]

    log.info(
        "Seeding demo app configs (model_url=%s, traffic demo model=yolov8n)",
        model_url,
    )
    _ping_minio()

    for served_model_id, video_filename, entries in demos:
        dest_key = f"uploads/{video_filename}"
        _ensure_object_with_retry(cfg_bucket, dest_key, data_bucket, video_filename)
        video_uri = f"s3://{cfg_bucket}/{dest_key}"
        cfg_id = insert_config(model_url, video_uri, served_model_id)
        replace_detection_classes(cfg_id, entries)
        generate_thumbnail_for_video_source(video_uri)
        log.info(
            "Seeded app_config id=%s model_name=%s video=%s",
            cfg_id,
            served_model_id,
            video_uri,
        )

    rtsp_url = _default_rtsp_live_url()
    ppe_live_id = insert_config(model_url, rtsp_url, "ppe")
    replace_detection_classes(ppe_live_id, _ppe_class_entries())
    log.info(
        f"Seeded app_config id={ppe_live_id} model_name=ppe (RTSP live) video={rtsp_url}"
    )
