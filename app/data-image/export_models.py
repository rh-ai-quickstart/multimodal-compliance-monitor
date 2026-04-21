#!/usr/bin/env python3
"""Export every *.pt under WEIGHTS_DIR to OpenVino IR and ONNX (Ultralytics).

Run after: pip install git+https://github.com/openai/CLIP.git
(in the exporter image Dockerfile).
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys

WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/weights")
OUT_OPENVINO = os.environ.get("OUT_OPENVINO", "/exported-openvino")
OUT_ONNX = os.environ.get("OUT_ONNX", "/exported-onnx")
EXCLUDE_STEMS = frozenset(
    s.strip()
    for s in os.environ.get("EXPORT_EXCLUDE_STEMS", "custome_ppe").split(",")
    if s.strip()
)

# OVMS on OpenShift mounts the models bucket at /mnt/models; must match create_runtime.py.
OVMS_MOUNT_BASE = os.environ.get("OVMS_CLUSTER_MOUNT_BASE", "/mnt/models")


def _ovms_config_nireq() -> int:
    raw = os.environ.get("OVMS_CONFIG_NIREQ", "2").strip()
    try:
        n = int(raw)
    except ValueError:
        print(f"warning: invalid OVMS_CONFIG_NIREQ={raw!r}, using 2", file=sys.stderr)
        return 2
    return max(1, n)


def _ovms_config_plugin_config() -> dict:
    raw = os.environ.get("OVMS_CONFIG_PLUGIN_CONFIG")
    if raw is not None and raw.strip():
        try:
            val = json.loads(raw)
        except json.JSONDecodeError as e:
            print(
                f"ERROR: OVMS_CONFIG_PLUGIN_CONFIG must be JSON: {e}", file=sys.stderr
            )
            sys.exit(1)
        if not isinstance(val, dict):
            print(
                "ERROR: OVMS_CONFIG_PLUGIN_CONFIG must be a JSON object",
                file=sys.stderr,
            )
            sys.exit(1)
        return val
    return {"PERFORMANCE_HINT": "THROUGHPUT"}


def _ovms_per_model_extras() -> dict:
    """Fields OVMS reads per model in config.json (not on CLI with --config_path)."""
    extras: dict = {
        "nireq": _ovms_config_nireq(),
        "plugin_config": _ovms_config_plugin_config(),
    }
    device = os.environ.get("OVMS_CONFIG_TARGET_DEVICE", "").strip()
    if device:
        extras["target_device"] = device
    batch = os.environ.get("OVMS_CONFIG_BATCH_SIZE", "").strip()
    if batch:
        extras["batch_size"] = batch
    shape = os.environ.get("OVMS_CONFIG_SHAPE", "").strip()
    if shape:
        extras["shape"] = json.loads(shape) if shape.startswith("{") else shape
    return extras


def write_ovms_config_json(root: str, mount_base: str = OVMS_MOUNT_BASE) -> None:
    """Emit OVMS multi-model config.json under ovms/ (ISVC storage path prefix = ovms)."""
    entries: list[dict] = []
    ovms_dir = os.path.join(root, "ovms")
    if not os.path.isdir(ovms_dir):
        print("warning: skipping config.json (no ovms/ export dir)", file=sys.stderr)
        return
    for name in sorted(os.listdir(ovms_dir)):
        path = os.path.join(ovms_dir, name)
        if not os.path.isdir(path) or name.startswith("."):
            continue
        if name.endswith("-onnx"):
            continue
        xml = os.path.join(path, "1", f"{name}.xml")
        if os.path.isfile(xml):
            model_cfg: dict = {
                "name": name,
                "base_path": f"{mount_base.rstrip('/')}/{name}",
            }
            model_cfg.update(_ovms_per_model_extras())
            entries.append({"config": model_cfg})
    if not entries:
        print("warning: skipping config.json (no OpenVINO model dirs)", file=sys.stderr)
        return
    cfg = {"model_config_list": entries}
    out = os.path.join(ovms_dir, "config.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"wrote {out} with {len(entries)} openvino model(s)")


def main() -> None:
    os.makedirs(OUT_OPENVINO, exist_ok=True)
    os.makedirs(OUT_ONNX, exist_ok=True)

    pts = sorted(glob.glob(os.path.join(WEIGHTS_DIR, "*.pt")))
    if not pts:
        print(f"No .pt files in {WEIGHTS_DIR}", file=sys.stderr)
        sys.exit(1)

    for pt_path in pts:
        stem = os.path.splitext(os.path.basename(pt_path))[0]
        if stem in EXCLUDE_STEMS:
            print(f"skip (excluded): {stem}")
            continue

        d_ov = os.path.join(OUT_OPENVINO, "ovms", stem, "1")
        xml_path = os.path.join(d_ov, f"{stem}.xml")
        if os.path.isfile(xml_path):
            print(f"skip (exists): {stem}")
            continue

        tmp_pt = f"/tmp/{stem}.pt"
        shutil.copy(pt_path, tmp_pt)
        subprocess.run(
            [
                "yolo",
                "export",
                f"model={tmp_pt}",
                "format=openvino",
                "task=detect",
                "dynamic=True",
            ],
            check=True,
        )
        os.makedirs(d_ov, exist_ok=True)
        ov_dir = f"/tmp/{stem}_openvino_model"
        for pat, dest_name in (("*.xml", f"{stem}.xml"), ("*.bin", f"{stem}.bin")):
            found = glob.glob(os.path.join(ov_dir, pat))
            if not found:
                print(f"ERROR: no {pat} in {ov_dir}", file=sys.stderr)
                sys.exit(1)
            shutil.copy(found[0], os.path.join(d_ov, dest_name))

        subprocess.run(
            [
                "yolo",
                "export",
                f"model={tmp_pt}",
                "format=onnx",
                "task=detect",
                "dynamic=True",
            ],
            check=True,
        )
        # Triton layout: triton/<stem>/1/model.onnx (IR lives under ovms/<stem>/1/).
        d_onx = os.path.join(OUT_ONNX, "triton", stem, "1")
        os.makedirs(d_onx, exist_ok=True)
        onnx_src = f"/tmp/{stem}.onnx"
        shutil.copy(onnx_src, os.path.join(d_onx, "model.onnx"))
        print(f"exported: {stem}")

    write_ovms_config_json(OUT_OPENVINO)
    print("export_models: done")


if __name__ == "__main__":
    main()
