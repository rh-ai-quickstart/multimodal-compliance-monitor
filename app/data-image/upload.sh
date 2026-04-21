#!/bin/sh
set -e

echo "=== PPE Compliance Monitor Data Uploader ==="

export MC_CONFIG_DIR=/tmp/.mc

echo "Waiting for MinIO to be ready..."
until mc alias set myminio "${MINIO_ENDPOINT}" "${MINIO_ACCESS_KEY}" "${MINIO_SECRET_KEY}" 2>/dev/null; do
	echo "MinIO not ready, retrying in 2 seconds..."
	sleep 2
done
echo "MinIO connection established"

echo "Creating buckets..."
mc mb --ignore-existing myminio/models
mc mb --ignore-existing myminio/data
mc mb --ignore-existing myminio/config
echo "Buckets ready"

RUNTIME_TYPE="${RUNTIME_TYPE}"
echo "Runtime type: ${RUNTIME_TYPE}"

# Regenerate OVMS config.json from env vars when any OVMS_CONFIG_* is set.
# Writes to /tmp (always writable) since the baked-in file under /upload is
# owned by root and read-only under OpenShift's random-UID SCC.
OVMS_CONFIG_FILE="/upload/models/ovms/config.json"

regen_ovms_config() {
	MOUNT_BASE="${OVMS_CLUSTER_MOUNT_BASE:-/mnt/models}"
	NIREQ="${OVMS_CONFIG_NIREQ:-2}"
	PLUGIN_CFG="${OVMS_CONFIG_PLUGIN_CONFIG}"
	if [ -z "$PLUGIN_CFG" ]; then
		PLUGIN_CFG='{"PERFORMANCE_HINT": "THROUGHPUT"}'
	fi
	SHAPE="${OVMS_CONFIG_SHAPE}"
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

# Regenerate OVMS config.json from env vars when any OVMS_CONFIG_* is set.
# Writes to /tmp (always writable) since the baked-in file under /upload is
# owned by root and read-only under OpenShift's random-UID SCC.
OVMS_CONFIG_FILE="/upload/models/ovms/config.json"

regen_ovms_config() {
	MOUNT_BASE="${OVMS_CLUSTER_MOUNT_BASE:-/mnt/models}"
	NIREQ="${OVMS_CONFIG_NIREQ:-2}"
	PLUGIN_CFG="${OVMS_CONFIG_PLUGIN_CONFIG}"
	if [ -z "$PLUGIN_CFG" ]; then
		PLUGIN_CFG='{"PERFORMANCE_HINT": "THROUGHPUT"}'
	fi
	SHAPE="${OVMS_CONFIG_SHAPE}"
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

# OVMS assets must reach MinIO whenever the data image includes them. The init Job
# uses modelServing.runtimeType (often kserve) while an OVMS InferenceService still
# expects models/ovms/config.json under the KServe storage prefix—so this is not
# gated on RUNTIME_TYPE.
if [ -d /upload/models/ovms ]; then
	echo "Checking / uploading OpenVINO model trees (ovms/<model>/1/)..."
	for d in /upload/models/ovms/*/; do
		[ -d "$d" ] || continue
		base=$(basename "$d")
		case "$base" in *-onnx) continue ;; esac
		if [ ! -f "${d}1/${base}.xml" ]; then
			continue
		fi
		if ! mc stat "myminio/models/ovms/${base}/1/${base}.xml" >/dev/null 2>&1; then
			echo "Uploading OpenVINO model: ovms/${base}/"
			mc cp --recursive "$d" "myminio/models/ovms/${base}/"
		else
			echo "OpenVINO ovms/${base} already present, skipping"
		fi
	done
	if [ -n "${OVMS_CONFIG_NIREQ:-}" ] || [ -n "${OVMS_CONFIG_PLUGIN_CONFIG:-}" ] || [ -n "${OVMS_CONFIG_SHAPE:-}" ]; then
		regen_ovms_config
	fi
	if [ -f "$OVMS_CONFIG_FILE" ]; then
		echo "Uploading OpenVINO config.json (multi-model OVMS)..."
		mc cp "$OVMS_CONFIG_FILE" myminio/models/ovms/config.json
	fi
fi

if [ "$RUNTIME_TYPE" = "kserve" ]; then
	echo "Checking / uploading Triton ONNX model trees (triton/<model>/1/model.onnx)..."
	for d in /upload/models/triton/*/; do
		[ -d "$d" ] || continue
		stem=$(basename "$d")
		onnx_path="${d}1/model.onnx"
		if [ ! -f "$onnx_path" ]; then
			continue
		fi
		if ! mc stat "myminio/models/triton/${stem}/1/model.onnx" >/dev/null 2>&1; then
			echo "Uploading Triton ONNX model: triton/${stem}/"
			mc cp --recursive "$d" "myminio/models/triton/${stem}/"
		else
			echo "Triton ONNX ${stem} already present, skipping"
		fi
	done
	# Optional Triton config (GPU / TensorRT); repo template targets ppe I/O shape only—do not copy to other stems.
	if [ -f /upload/triton-config/config.pbtxt ] && [ -f /upload/models/triton/ppe/1/model.onnx ]; then
		echo "Uploading Triton config for triton/ppe/..."
		mc cp /upload/triton-config/config.pbtxt myminio/models/triton/ppe/config.pbtxt
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
	if ! mc stat "myminio/models/${bn}" >/dev/null 2>&1; then
		echo "Uploading ${bn}"
		mc cp "$f" "myminio/models/${bn}"
	else
		echo "${bn} already in bucket, skipping"
	fi
done

echo "Checking sample videos in data bucket..."
for vid in cars.mp4 combined-video-no-gap-rooftop.mp4 bluejayclear.mp4; do
	if ! mc stat "myminio/data/${vid}" >/dev/null 2>&1; then
		echo "Uploading video (${vid})..."
		mc cp "/upload/data/${vid}" myminio/data/
		echo "Uploaded ${vid}"
	else
		echo "${vid} already in bucket, skipping"
	fi
done

echo "=== Data upload complete ==="

echo ""
echo "Files in MinIO:"
echo "--- models bucket ---"
mc ls myminio/models/
echo "--- data bucket ---"
mc ls myminio/data/
