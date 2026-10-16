#!/bin/sh
set -e

echo "=== PPE Compliance Monitor Data Uploader ==="

AWS_ENDPOINT_URL="${AWS_ENDPOINT_URL:-http://aws-compatible-storage:7480}"
export AWS_ENDPOINT_URL
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-s4admin}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-s4secret}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export AWS_PAGER=""

aws_s3() {
	aws --endpoint-url "$AWS_ENDPOINT_URL" s3 "$@"
}

aws_s3api() {
	aws --endpoint-url "$AWS_ENDPOINT_URL" s3api "$@"
}

echo "Waiting for S3-compatible storage to be ready..."
until aws_s3 ls >/dev/null 2>&1; do
	echo "S3-compatible storage not ready, retrying in 2 seconds..."
	sleep 2
done
echo "S3-compatible storage connection established"

echo "Creating buckets..."
for bucket in models data config; do
	aws_s3 mb "s3://${bucket}" >/dev/null 2>&1 || true
done
echo "Buckets ready"

RUNTIME_TYPE="${RUNTIME_TYPE:-}"
echo "Runtime type: ${RUNTIME_TYPE}"

# Regenerate OVMS config.json from env vars when any OVMS_CONFIG_* is set.
# Writes to /tmp since the baked-in file under /upload is read-only under
# OpenShift's random-UID SCC.
OVMS_CONFIG_FILE="/upload/models/ovms/config.json"

regen_ovms_config() {
	MOUNT_BASE="${OVMS_CLUSTER_MOUNT_BASE:-/mnt/models}"
	NIREQ="${OVMS_CONFIG_NIREQ:-2}"
	PLUGIN_CFG="${OVMS_CONFIG_PLUGIN_CONFIG}"
	if [ -z "$PLUGIN_CFG" ]; then
		PLUGIN_CFG='{"PERFORMANCE_HINT": "THROUGHPUT"}'
	fi
	SHAPE="${OVMS_CONFIG_SHAPE:-}"
	OUT="/tmp/ovms-config.json"

	first=true
	printf '{\n  "model_config_list": [\n' >"$OUT"
	for d in /upload/models/ovms/*/; do
		[ -d "$d" ] || continue
		name=$(basename "$d")
		case "$name" in *-onnx) continue ;; esac
		[ -f "${d}1/${name}.xml" ] || continue

		if [ "$first" = true ]; then first=false; else printf ',\n' >>"$OUT"; fi
		printf '    {\n      "config": {\n' >>"$OUT"
		printf '        "name": "%s",\n' "$name" >>"$OUT"
		printf '        "base_path": "%s/%s",\n' "$MOUNT_BASE" "$name" >>"$OUT"
		printf '        "nireq": %s,\n' "$NIREQ" >>"$OUT"
		printf '        "plugin_config": %s' "$PLUGIN_CFG" >>"$OUT"
		if [ -n "$SHAPE" ]; then
			printf ',\n        "shape": %s' "$SHAPE" >>"$OUT"
		fi
		printf '\n      }\n    }' >>"$OUT"
	done
	printf '\n  ]\n}\n' >>"$OUT"
	OVMS_CONFIG_FILE="$OUT"
	echo "Regenerated config.json (nireq=$NIREQ)"
}

object_exists() {
	aws_s3api head-object --bucket "$1" --key "$2" >/dev/null 2>&1
}

upload_tree() {
	local_dir="$1"
	bucket="$2"
	prefix="$3"
	aws_s3 cp "$local_dir" "s3://${bucket}/${prefix}/" --recursive
}

# OVMS assets must reach object storage whenever the data image includes them. This is not
# gated on RUNTIME_TYPE because an OVMS InferenceService may use the KServe
# storage prefix even when the init Job runs with runtimeType=kserve.
if [ -d /upload/models/ovms ]; then
	echo "Checking / uploading OpenVINO model trees (ovms/<model>/1/)..."
	for d in /upload/models/ovms/*/; do
		[ -d "$d" ] || continue
		base=$(basename "$d")
		case "$base" in *-onnx) continue ;; esac
		[ -f "${d}1/${base}.xml" ] || continue
		if ! object_exists models "ovms/${base}/1/${base}.xml"; then
			echo "Uploading OpenVINO model: ovms/${base}/"
			upload_tree "$d" models "ovms/${base}"
		else
			echo "OpenVINO ovms/${base} already present, skipping"
		fi
	done
	if [ -n "${OVMS_CONFIG_NIREQ:-}" ] ||
		[ -n "${OVMS_CONFIG_PLUGIN_CONFIG:-}" ] ||
		[ -n "${OVMS_CONFIG_SHAPE:-}" ]; then
		regen_ovms_config
	fi
	if [ -f "$OVMS_CONFIG_FILE" ]; then
		echo "Uploading OpenVINO config.json (multi-model OVMS)..."
		aws_s3 cp "$OVMS_CONFIG_FILE" s3://models/ovms/config.json
	fi
fi

if [ "$RUNTIME_TYPE" = "kserve" ]; then
	echo "Checking / uploading Triton ONNX model trees (triton/<model>/1/model.onnx)..."
	for d in /upload/models/triton/*/; do
		[ -d "$d" ] || continue
		stem=$(basename "$d")
		onnx_path="${d}1/model.onnx"
		[ -f "$onnx_path" ] || continue
		if ! object_exists models "triton/${stem}/1/model.onnx"; then
			echo "Uploading Triton ONNX model: triton/${stem}/"
			upload_tree "$d" models "triton/${stem}"
		else
			echo "Triton ONNX ${stem} already present, skipping"
		fi
	done
	if [ -f /upload/triton-config/config.pbtxt ] &&
		[ -f /upload/models/triton/ppe/1/model.onnx ]; then
		echo "Uploading Triton config for triton/ppe/..."
		aws_s3 cp /upload/triton-config/config.pbtxt s3://models/triton/ppe/config.pbtxt
	fi
elif [ "$RUNTIME_TYPE" = "openvino" ]; then
	echo "Skipping Triton ONNX uploads (runtime is OpenVINO)."
else
	echo "ERROR: Unknown RUNTIME_TYPE '${RUNTIME_TYPE}'. Expected 'openvino' or 'kserve'."
	exit 1
fi

echo "Uploading raw .pt files (for reference / other runtimes)..."
for f in /upload/models-pt/*.pt; do
	[ -f "$f" ] || continue
	bn=$(basename "$f")
	if ! object_exists models "$bn"; then
		echo "Uploading ${bn}"
		aws_s3 cp "$f" "s3://models/${bn}"
	else
		echo "${bn} already in bucket, skipping"
	fi
done

echo "Checking sample videos in data bucket..."
for vid in cars.mp4 combined-video-no-gap-rooftop.mp4 bluejayclear.mp4; do
	if ! object_exists data "$vid"; then
		echo "Uploading video (${vid})..."
		aws_s3 cp "/upload/data/${vid}" s3://data/
		echo "Uploaded ${vid}"
	else
		echo "${vid} already in bucket, skipping"
	fi
done

echo "=== Data upload complete ==="
echo ""
echo "Files in object storage:"
echo "--- models bucket ---"
aws_s3 ls s3://models/
echo "--- data bucket ---"
aws_s3 ls s3://data/
