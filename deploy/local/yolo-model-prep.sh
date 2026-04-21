#!/usr/bin/env bash
# Export each /source/*.pt to OpenVINO under /models/ovms/<name>/1/ and write /models/ovms/config.json
set -euo pipefail

pip install --no-cache-dir 'git+https://github.com/openai/CLIP.git'

shopt -s nullglob
pts=(/source/*.pt)
if [[ ${#pts[@]} -eq 0 ]]; then
	echo "ERROR: No .pt files in /source (mount app/models)." >&2
	exit 1
fi

for pt in "${pts[@]}"; do
	stem=$(basename "$pt" .pt)
	if [[ $stem == "custome_ppe" ]]; then
		echo "skip (excluded): $stem"
		continue
	fi
	target_dir="/models/ovms/${stem}/1"
	target_xml="${target_dir}/${stem}.xml"
	if [[ -f $target_xml ]]; then
		echo "skip (exists): $stem"
		continue
	fi
	echo "exporting: $stem"
	mkdir -p "$target_dir"
	cp "$pt" "/tmp/${stem}.pt"
	yolo export "model=/tmp/${stem}.pt" format=openvino task=detect dynamic=True
	cp "/tmp/${stem}_openvino_model/"*.xml "$target_xml"
	cp "/tmp/${stem}_openvino_model/"*.bin "${target_dir}/${stem}.bin"
done

# Same OVMS per-model tuning as data-image export (nireq, plugin_config, optional batch/device via env).
python3 -c "
import json
import sys

sys.path.insert(0, '/export_models')
from export_models import write_ovms_config_json

write_ovms_config_json('/models', '/models/ovms')
with open('/models/ovms/config.json', encoding='utf-8') as f:
    cfg = json.load(f)
names = [e['config']['name'] for e in cfg.get('model_config_list', [])]
if not names:
    raise SystemExit('No models found under /models/ovms — export failed?')
print(f'Wrote /models/ovms/config.json with {len(names)} model(s): {names}')
"

echo "yolo-model-prep: complete"
