"""S3 video thumbnail generation for config video sources."""

import os
import tempfile

import cv2

from s3_client import download_file, get_config_bucket, object_exists, upload_bytes
from logger import get_logger

log = get_logger(__name__)

_THUMBNAIL_TIMESTAMP_S = 17.0


def is_s3_video_path(path) -> bool:
    """True if path is a universal S3 object URI (s3://bucket/key)."""
    if not path or not isinstance(path, str):
        return False
    return path.strip().startswith("s3://")


def parse_s3_video_path(video_path: str):
    """Parse S3 URI into (bucket, object_key). Returns None if not s3:// path."""
    if not video_path or not isinstance(video_path, str):
        return None
    p = video_path.strip()
    if p.startswith("s3://"):
        parts = p[5:].split("/", 1)
        if len(parts) == 2:
            return (parts[0], parts[1])
    return None


def generate_thumbnail_for_video_source(video_path: str):
    """Generate a JPEG thumbnail from S3 video, upload to S3. Returns S3 key or None.

    Thumbnails are always stored in the config bucket (``CONFIG_BUCKET`` / ``get_config_bucket()``)
    under ``thumbnails/``, matching the ``/api/thumbnails/...`` route and the Config dialog **Upload**
    flow. The video may live in any bucket; we download from its URI then write the JPEG to
    the config bucket so the UI can load ``/api/thumbnails/<stem>.jpg``.
    """
    if not is_s3_video_path(video_path):
        return None
    parsed = parse_s3_video_path(video_path)
    if not parsed:
        return None
    bucket, key = parsed
    stem = os.path.splitext(os.path.basename(key))[0]
    if not stem:
        return None
    thumb_bucket = get_config_bucket()
    thumb_key = f"thumbnails/{stem}.jpg"
    try:
        if object_exists(thumb_bucket, thumb_key):
            log.debug("Thumbnail already exists: %s/%s", thumb_bucket, thumb_key)
            return thumb_key
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            download_file(bucket, key, tmp_path)
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                log.warning("Could not open video for thumbnail: %s", video_path)
                return None
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_pos = int(_THUMBNAIL_TIMESTAMP_S * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                ret_jpg, buf = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                if ret_jpg:
                    upload_bytes(
                        thumb_bucket,
                        thumb_key,
                        buf.tobytes(),
                        content_type="image/jpeg",
                    )
                    log.info(
                        "Generated thumbnail: %s/%s (video was %s)",
                        thumb_bucket,
                        thumb_key,
                        video_path,
                    )
                    return thumb_key
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception as e:
        log.warning("Thumbnail generation failed for %s: %s", video_path, e)
    return None
