"""S3 helpers for aws-compatible-storage, using boto3 and AWS_* settings."""

import io
import os
import time

import boto3
from boto3.exceptions import RetriesExceededError
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from logger import get_logger

log = get_logger(__name__)

# Config bucket for user uploads and thumbnails (enables horizontal scaling)
CONFIG_BUCKET = os.getenv("CONFIG_BUCKET", "config")


def get_config_bucket():
    """Return the bucket name for config uploads and thumbnails."""
    return CONFIG_BUCKET


def get_s3_client():
    """Create a path-style S3 client; the endpoint URL determines HTTP or HTTPS."""
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("AWS_ENDPOINT_URL")
        or "http://aws-compatible-storage:7480",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID") or "s4admin",
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY") or "s4secret",
        region_name=os.getenv("AWS_DEFAULT_REGION") or "us-east-1",
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def download_file(
    bucket: str,
    object_name: str,
    local_path: str,
    max_retries: int = 5,
    retry_delay: int = 3,
) -> str:
    """
    Download a file from S3 to a local path.

    Args:
        bucket: S3 bucket name
        object_name: Object key/path in the bucket
        local_path: Local filesystem path to save the file
        max_retries: Maximum number of retry attempts
        retry_delay: Seconds to wait between retries

    Returns:
        The local path where the file was saved

    Raises:
        ClientError, BotoCoreError, RetriesExceededError: If all retries fail
    """
    client = get_s3_client()

    # Ensure the directory exists
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

    for attempt in range(max_retries):
        try:
            log.info(
                f"Downloading {bucket}/{object_name} to {local_path} (attempt {attempt + 1}/{max_retries})"
            )
            client.download_file(bucket, object_name, local_path)
            log.info(f"Successfully downloaded {bucket}/{object_name}")
            return local_path
        except (ClientError, BotoCoreError, RetriesExceededError) as e:
            if attempt < max_retries - 1:
                log.warning(
                    f"Download failed: {e}. Retrying in {retry_delay} seconds..."
                )
                time.sleep(retry_delay)
            else:
                log.error(f"Download failed after {max_retries} attempts: {e}")
                raise


def upload_file(
    bucket: str,
    object_name: str,
    file_path: str,
    content_type: str | None = None,
) -> None:
    """
    Upload a file from the local filesystem to S3.

    Args:
        bucket: S3 bucket name
        object_name: Object key/path in the bucket
        file_path: Local filesystem path to the file
        content_type: Optional MIME type (e.g. 'video/mp4', 'image/jpeg')
    """
    client = get_s3_client()
    extra_args = {"ContentType": content_type} if content_type else {}
    client.upload_file(file_path, bucket, object_name, ExtraArgs=extra_args)
    log.info(f"Uploaded {file_path} to {bucket}/{object_name}")


def copy_object(
    dest_bucket: str,
    dest_key: str,
    src_bucket: str,
    src_key: str,
) -> None:
    """Server-side copy within or across buckets on the same S3 endpoint."""
    client = get_s3_client()
    client.copy({"Bucket": src_bucket, "Key": src_key}, dest_bucket, dest_key)
    log.info(f"Copied s3://{src_bucket}/{src_key} to s3://{dest_bucket}/{dest_key}")


def upload_bytes(
    bucket: str,
    object_name: str,
    data: bytes,
    content_type: str | None = None,
) -> None:
    """
    Upload bytes to S3.

    Args:
        bucket: S3 bucket name
        object_name: Object key/path in the bucket
        data: Raw bytes to upload
        content_type: Optional MIME type
    """
    client = get_s3_client()
    extra_args = {"ContentType": content_type} if content_type else {}
    client.upload_fileobj(
        io.BytesIO(data),
        bucket,
        object_name,
        ExtraArgs=extra_args,
    )
    log.info(f"Uploaded {len(data)} bytes to {bucket}/{object_name}")


def get_object_stream(bucket: str, object_name: str):
    """
    Get an object from S3 as a stream.

    Returns:
        StreamingBody with read() and iter_chunks(); the caller must close() it.
    """
    client = get_s3_client()
    return client.get_object(Bucket=bucket, Key=object_name)["Body"]


def object_exists(bucket: str, object_name: str) -> bool:
    """Check for an S3 object, propagating authentication and service errors."""
    try:
        client = get_s3_client()
        client.head_object(Bucket=bucket, Key=object_name)
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in {"404", "NoSuchKey", "NotFound"} or status == 404:
            return False
        raise
