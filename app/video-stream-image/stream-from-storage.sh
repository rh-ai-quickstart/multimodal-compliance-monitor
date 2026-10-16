#!/bin/sh
# Downloads an MP4 from aws-compatible-storage and streams it to MediaMTX.
set -eu

AWS_ENDPOINT_URL="${AWS_ENDPOINT_URL:-http://aws-compatible-storage:7480}"
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-s4admin}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-s4secret}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export AWS_PAGER=""
VIDEO_BUCKET="${VIDEO_BUCKET:-data}"
VIDEO_KEY="${VIDEO_KEY:-combined-video-no-gap-rooftop.mp4}"
MEDIAMTX_HOST="${MEDIAMTX_HOST:-video-stream}"
MEDIAMTX_PORT="${MEDIAMTX_PORT:-8554}"
MEDIAMTX_PATH="${MEDIAMTX_PATH:-live}"
VIDEO_PATH="/tmp/video.mp4"

echo "=== Video Stream Publisher ==="
echo "Downloading s3://${VIDEO_BUCKET}/${VIDEO_KEY}..."
attempt=1
until aws --endpoint-url "$AWS_ENDPOINT_URL" s3 cp \
	"s3://${VIDEO_BUCKET}/${VIDEO_KEY}" "$VIDEO_PATH" --only-show-errors; do
	if [ "$attempt" -ge 60 ]; then
		echo "Could not download s3://${VIDEO_BUCKET}/${VIDEO_KEY} after 60 attempts" >&2
		exit 1
	fi
	echo "Download failed (attempt ${attempt}/60), retrying in 5 seconds..."
	sleep 5
	attempt=$((attempt + 1))
done
if [ ! -f "$VIDEO_PATH" ] || [ ! -s "$VIDEO_PATH" ]; then
	echo "$VIDEO_PATH missing or empty after download" >&2
	exit 1
fi
echo "Video downloaded"

echo "Waiting for MediaMTX to be ready..."
attempt=1
until nc -z -w 2 "$MEDIAMTX_HOST" "$MEDIAMTX_PORT"; do
	if [ "$attempt" -ge 60 ]; then
		echo "MediaMTX ${MEDIAMTX_HOST}:${MEDIAMTX_PORT} not reachable after 60 attempts" >&2
		exit 1
	fi
	echo "MediaMTX not ready (attempt ${attempt}/60), retrying in 2 seconds..."
	sleep 2
	attempt=$((attempt + 1))
done
echo "MediaMTX ready"

echo "Starting FFmpeg stream (loop)..."
exec ffmpeg -re -stream_loop -1 -i "$VIDEO_PATH" -c copy -f rtsp \
	"rtsp://${MEDIAMTX_HOST}:${MEDIAMTX_PORT}/${MEDIAMTX_PATH}"
