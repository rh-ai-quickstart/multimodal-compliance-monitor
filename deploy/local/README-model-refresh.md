# YOLO `.pt` weights and the OVMS model repo (local Podman)

This describes how **`deploy/local/podman-compose.yaml`** consumes Ultralytics **`.pt`** checkpoints under **`app/models/`**.

OpenVINO Model Server (OVMS) does **not** load `.pt` directly. The **`yolo-model-prep`** service exports each `*.pt` to OpenVINO IR (`.xml` / `.bin`) on the Podman volume **`model_repo`**, mounted at **`/models`** inside OVMS. The **model id** OVMS uses is the **file stem**: for `app/models/bird.pt` the stem is **`bird`**, and IR lands under **`/models/bird/1/bird.xml`** (and companion `.bin`).

Examples below use **`bird.pt`** / **`bird`** or **`yolov8n.pt`** / **`yolov8n`**; substitute your own stem (e.g. **`ppe`** for **`ppe.pt`**).

## Automated import with `deploy/local/import_model_weight.py` (recommended)

The script **`deploy/local/import_model_weight.py`** automates the same outcome as the manual steps later in this doc: it ensures weights live under **`app/models/`**, refreshes the OpenVINO export on the **`model_repo`** volume when needed, and reloads OVMS.

### How to run

Run from the **repository root** (so paths like **`app/models/`** resolve correctly):

```bash
cd /path/to/ppe-compliance-monitor
python deploy/local/import_model_weight.py bird.pt
```

Other examples:

```bash
# YOLOv8n COCO checkpoint (stem: yolov8n)
python deploy/local/import_model_weight.py yolov8n.pt

# Already in app/models
python deploy/local/import_model_weight.py app/models/bird.pt

# Any path; file is copied to app/models/<stem>.pt if it is not already there
python deploy/local/import_model_weight.py /tmp/my-run/weights/best.pt

# Optional: non-default compose file
python deploy/local/import_model_weight.py bird.pt --compose-file /path/to/podman-compose.yaml
```

Help:

```bash
python deploy/local/import_model_weight.py --help
```

### What it does

1. **Resolves** the **`.pt`** file you named. The **model stem** is the basename without **`.pt`** (e.g. **`bird.pt`** → stem **`bird`**). Weights are normalized to **`app/models/<stem>.pt`** by copying when the source is elsewhere.
2. **Rejects** **`custome_ppe`** (that stem is excluded by **`deploy/local/yolo-model-prep.sh`**, same as a manual prep run).
3. **Finds** the Podman volume that backs OVMS **`model_repo`** (prefers inspecting a running **ovms** container’s **`/models`** mount; otherwise a single `*_model_repo` volume).
4. **Decides new vs replacement** by checking for **`/models/<stem>/1/<stem>.xml`** on that volume:
   - **Not present** → first-time import for that stem; no delete.
   - **Present** → replacement; removes **`/models/<stem>`** so prep will re-export (mirrors the manual **`rm -rf /models/<stem>`** flow).
5. **Runs export** via Compose: **`podman-compose -f … run --rm yolo-model-prep`** (or **`podman compose`**). That **service** runs **`deploy/local/yolo-model-prep.sh`** inside the image (see **`podman-compose.yaml`**); the Python importer does **not** shell out to the **`.sh`** file directly.
6. **Restarts** **ovms** if it is already running, or **`up -d ovms`** if not, so **`config.json`** and models are picked up.

### Requirements and limits

- **Podman** and **`podman-compose`** (or **`podman compose`**) must work on your machine.
- **Compose driver:** the script prefers **`podman-compose`** when it is on **`PATH`**, otherwise **`podman compose`**. Set **`PODMAN_COMPOSE=compose`** or **`PODMAN_COMPOSE=podman-compose`** to force one.
- The **`model_repo`** volume must already exist (e.g. you have run **`podman-compose -f deploy/local/podman-compose.yaml up -d ovms`** at least once). If there is no volume yet, run that once before relying on the importer.
- If several `*_model_repo` volumes exist and **ovms** is not running, the script may exit with an error; start **ovms** once or remove stray volumes so the correct mount can be detected.

After a successful run, configure the app with **Model name** = that stem (e.g. **`bird`** or **`yolov8n`**) and **Model URL** **`ovms:8081`** as in [App configuration](#app-configuration).

The sections below describe the **manual** first-time and replacement flows if you prefer not to use the script.

---

## Adding a new weight file (first time for that stem)

Use this when you introduce a **`app/models/<stem>.pt`** that has **never** been exported to **`model_repo`** (no existing **`/models/<stem>`** tree, or this is a fresh volume).

1. **Place** the checkpoint next to other weights:

   ```text
   app/models/bird.pt
   ```

2. **Start** export + OVMS (prep runs before OVMS comes up):

   ```bash
   cd /path/to/ppe-compliance-monitor
   podman-compose -f deploy/local/podman-compose.yaml up -d ovms
   ```

   Or bring up the full app stack:

   ```bash
   make local-build-up
   ```

3. **Confirm** prep exported the new model (you should see **`exporting: bird`**, not only skips):

   ```bash
   podman logs "$(podman ps -aq -f name=yolo-model-prep | head -1)" 2>&1 | grep -E 'exporting:|skip '
   ```

4. **Confirm** OVMS is ready:

   ```bash
   curl -sf http://localhost:8080/v2/health/ready && echo OK
   ```

5. In the app, use a config whose **Model name** matches the stem (for this example: **`bird`**). See [App configuration](#app-configuration).

---

## Replacing an existing `.pt` (same stem, new weights)

Prep **skips** export when **`/models/<stem>/1/<stem>.xml`** already exists. Copying a new `bird.pt` over the old file **does not** refresh OVMS; the stale IR keeps serving.

1. **Overwrite** the checkpoint on the host:

   ```text
   app/models/bird.pt
   ```

2. **Remove** the cached IR for that stem on the **`model_repo`** volume.

   Resolve the volume name from the **running OVMS** container (adjust the container name if yours differs; do not rely on `grep … | head -1` across unrelated volumes):

   ```bash
   cd /path/to/ppe-compliance-monitor
   VOL=$(podman inspect local_ovms_1 --format '{{range .Mounts}}{{if eq .Destination "/models"}}{{.Name}}{{end}}{{end}}')
   podman run --rm -v "$VOL:/models:Z" docker.io/library/alpine:3.19 sh -c 'rm -rf /models/bird && ls /models'
   ```

   After **`rm -rf /models/bird`**, the listing should no longer contain a **`bird`** directory (you may still see **`config.json`**, **`ppe`**, other stems, etc.).

3. **Rerun** export + OVMS:

   ```bash
   podman-compose -f deploy/local/podman-compose.yaml up -d ovms
   ```

   or:

   ```bash
   make local-build-up
   ```

4. **Confirm** logs show a real export again, e.g. **`exporting: bird`**, not **`skip (exists): bird`**.

5. **Confirm** OVMS health as in the first-time flow.

---

## App configuration

For the active config (UI or API), **Model name** must match the **stem** of the `.pt` OVMS is serving.

Example for **`bird.pt`**:

| Setting        | Example value |
|----------------|---------------|
| Model URL      | `ovms:8081`   |
| Model name     | `bird`        |
| Classes        | Your class ids and labels in training order (e.g. Bluejays, Cardinals). |

Local compose typically sets **`MODEL_INPUT_NAME=x`**, which matches this OVMS export path.

### JSON for `POST /api/config` (bird demo)

Create a config with the same fields the Config UI sends. **`classes`** keys are model class indices (strings). Each value is an object with:

| Field | Required | Description |
|--------|----------|-------------|
| **`name`** | yes | Label for that model class index. |
| **`trackable`** | yes | Whether that class is used for object tracking (e.g. Person and DeepSORT). |
| **`include_in_counts`** | no (default **`true`**) | When **`false`**, detections for that class are omitted from bounding boxes, per-class aggregate counts, and tracker input. Omit this field when every class should appear (typical bird demo). |

Example body (OVMS on compose service **`ovms`**, sample video in MinIO **`data`** bucket after seeds / upload job):

```json
{
  "model_url": "ovms:8081",
  "model_name": "bird",
  "video_source": "s3://data/bluejayclear.mp4",
  "classes": {
    "0": { "name": "Bluejays", "trackable": true },
    "1": { "name": "Cardinals", "trackable": true }
  }
}
```

Example **`curl`** (replace host/port if needed):

```bash
curl -sS -X POST "http://localhost:8888/api/config" \
  -H "Content-Type: application/json" \
  -d '{
  "model_url": "ovms:8081",
  "model_name": "bird",
  "video_source": "s3://data/bluejayclear.mp4",
  "classes": {
    "0": {"name": "Bluejays", "trackable": true},
    "1": {"name": "Cardinals", "trackable": true}
  }
}'
```

Updates use the same JSON shape on **`PUT /api/config/<id>`**. Adjust **`video_source`** for your environment (other `s3://` objects, file paths, or RTSP URLs as supported by the backend).

### YOLOv8n (`yolov8n.pt`) and `cars.mp4` (COCO)

- **Weights:** keep **`app/models/yolov8n.pt`** and import with **`python deploy/local/import_model_weight.py yolov8n.pt`** (or rely on a full stack build that runs **`yolo-model-prep`** over all **`app/models/*.pt`**). In the Config dialog, **Model name** must be **`yolov8n`** (the file stem).
- **Video:** **`app/data/cars.mp4`** is included in the data-loader image and uploaded to MinIO by **`app/data-image/upload.sh`** as **`data/cars.mp4`**. Use **`video_source`**: **`s3://data/cars.mp4`** for playback from MinIO.
- **Classes:** YOLOv8n uses the **80-class COCO** output order. Every index **`"0"`**–**`"79"`** should appear in **`classes`** so labels match the model head. The example below is tuned for a **vehicle-focused** demo on **`cars.mp4`**: only **`car`** (index **`2`**) is **`trackable`** and **`include_in_counts`: `true`**; all other classes remain mapped for correct post-processing but are excluded from counts and overlay-related aggregation. Change **`trackable`** / **`include_in_counts`** (e.g. set **`person`** to trackable) if you want people-tracking or more classes visible.

Use this shape for **`POST /api/config`** (merge the **`classes`** object below into the body). Top-level fields:

```json
{
  "model_url": "ovms:8081",
  "model_name": "yolov8n",
  "video_source": "s3://data/cars.mp4"
}
```

**`classes`** (80 COCO indices; copy as the fourth key alongside the fields above):

```json
{
  "0": {"name": "person", "trackable": false, "include_in_counts": false},
  "1": {"name": "bicycle", "trackable": false, "include_in_counts": false},
  "2": {"name": "car", "trackable": true, "include_in_counts": true},
  "3": {"name": "motorcycle", "trackable": false, "include_in_counts": false},
  "4": {"name": "airplane", "trackable": false, "include_in_counts": false},
  "5": {"name": "bus", "trackable": false, "include_in_counts": false},
  "6": {"name": "train", "trackable": false, "include_in_counts": false},
  "7": {"name": "truck", "trackable": false, "include_in_counts": false},
  "8": {"name": "boat", "trackable": false, "include_in_counts": false},
  "9": {"name": "traffic light", "trackable": false, "include_in_counts": false},
  "10": {"name": "fire hydrant", "trackable": false, "include_in_counts": false},
  "11": {"name": "stop sign", "trackable": false, "include_in_counts": false},
  "12": {"name": "parking meter", "trackable": false, "include_in_counts": false},
  "13": {"name": "bench", "trackable": false, "include_in_counts": false},
  "14": {"name": "bird", "trackable": false, "include_in_counts": false},
  "15": {"name": "cat", "trackable": false, "include_in_counts": false},
  "16": {"name": "dog", "trackable": false, "include_in_counts": false},
  "17": {"name": "horse", "trackable": false, "include_in_counts": false},
  "18": {"name": "sheep", "trackable": false, "include_in_counts": false},
  "19": {"name": "cow", "trackable": false, "include_in_counts": false},
  "20": {"name": "elephant", "trackable": false, "include_in_counts": false},
  "21": {"name": "bear", "trackable": false, "include_in_counts": false},
  "22": {"name": "zebra", "trackable": false, "include_in_counts": false},
  "23": {"name": "giraffe", "trackable": false, "include_in_counts": false},
  "24": {"name": "backpack", "trackable": false, "include_in_counts": false},
  "25": {"name": "umbrella", "trackable": false, "include_in_counts": false},
  "26": {"name": "handbag", "trackable": false, "include_in_counts": false},
  "27": {"name": "tie", "trackable": false, "include_in_counts": false},
  "28": {"name": "suitcase", "trackable": false, "include_in_counts": false},
  "29": {"name": "frisbee", "trackable": false, "include_in_counts": false},
  "30": {"name": "skis", "trackable": false, "include_in_counts": false},
  "31": {"name": "snowboard", "trackable": false, "include_in_counts": false},
  "32": {"name": "sports ball", "trackable": false, "include_in_counts": false},
  "33": {"name": "kite", "trackable": false, "include_in_counts": false},
  "34": {"name": "baseball bat", "trackable": false, "include_in_counts": false},
  "35": {"name": "baseball glove", "trackable": false, "include_in_counts": false},
  "36": {"name": "skateboard", "trackable": false, "include_in_counts": false},
  "37": {"name": "surfboard", "trackable": false, "include_in_counts": false},
  "38": {"name": "tennis racket", "trackable": false, "include_in_counts": false},
  "39": {"name": "bottle", "trackable": false, "include_in_counts": false},
  "40": {"name": "wine glass", "trackable": false, "include_in_counts": false},
  "41": {"name": "cup", "trackable": false, "include_in_counts": false},
  "42": {"name": "fork", "trackable": false, "include_in_counts": false},
  "43": {"name": "knife", "trackable": false, "include_in_counts": false},
  "44": {"name": "spoon", "trackable": false, "include_in_counts": false},
  "45": {"name": "bowl", "trackable": false, "include_in_counts": false},
  "46": {"name": "banana", "trackable": false, "include_in_counts": false},
  "47": {"name": "apple", "trackable": false, "include_in_counts": false},
  "48": {"name": "sandwich", "trackable": false, "include_in_counts": false},
  "49": {"name": "orange", "trackable": false, "include_in_counts": false},
  "50": {"name": "broccoli", "trackable": false, "include_in_counts": false},
  "51": {"name": "carrot", "trackable": false, "include_in_counts": false},
  "52": {"name": "hot dog", "trackable": false, "include_in_counts": false},
  "53": {"name": "pizza", "trackable": false, "include_in_counts": false},
  "54": {"name": "donut", "trackable": false, "include_in_counts": false},
  "55": {"name": "cake", "trackable": false, "include_in_counts": false},
  "56": {"name": "chair", "trackable": false, "include_in_counts": false},
  "57": {"name": "couch", "trackable": false, "include_in_counts": false},
  "58": {"name": "potted plant", "trackable": false, "include_in_counts": false},
  "59": {"name": "bed", "trackable": false, "include_in_counts": false},
  "60": {"name": "dining table", "trackable": false, "include_in_counts": false},
  "61": {"name": "toilet", "trackable": false, "include_in_counts": false},
  "62": {"name": "tv", "trackable": false, "include_in_counts": false},
  "63": {"name": "laptop", "trackable": false, "include_in_counts": false},
  "64": {"name": "mouse", "trackable": false, "include_in_counts": false},
  "65": {"name": "remote", "trackable": false, "include_in_counts": false},
  "66": {"name": "keyboard", "trackable": false, "include_in_counts": false},
  "67": {"name": "cell phone", "trackable": false, "include_in_counts": false},
  "68": {"name": "microwave", "trackable": false, "include_in_counts": false},
  "69": {"name": "oven", "trackable": false, "include_in_counts": false},
  "70": {"name": "toaster", "trackable": false, "include_in_counts": false},
  "71": {"name": "sink", "trackable": false, "include_in_counts": false},
  "72": {"name": "refrigerator", "trackable": false, "include_in_counts": false},
  "73": {"name": "book", "trackable": false, "include_in_counts": false},
  "74": {"name": "clock", "trackable": false, "include_in_counts": false},
  "75": {"name": "vase", "trackable": false, "include_in_counts": false},
  "76": {"name": "scissors", "trackable": false, "include_in_counts": false},
  "77": {"name": "teddy bear", "trackable": false, "include_in_counts": false},
  "78": {"name": "hair drier", "trackable": false, "include_in_counts": false},
  "79": {"name": "toothbrush", "trackable": false, "include_in_counts": false}
}
```

RTSP / MediaMTX defaults in compose may still target the rooftop PPE sample clip; **`cars.mp4`** is intended for **`s3://data/cars.mp4`** (or your chosen **video_source**) in the config.

### PPE-style `classes` with `include_in_counts`

Multi-class PPE configs keep every model index in **`classes`** for correct name mapping, but set **`include_in_counts`: `false`** on indices that should not drive overlays or summaries (e.g. safety cones, machinery, vehicle). Indices without **`include_in_counts`** default to **`true`**.

Only **`classes`** is shown below; use the same top-level fields (**`model_url`**, **`model_name`**, **`video_source`**, etc.) as in the bird example.

```json
{
  "classes": {
    "0": { "name": "Hardhat", "trackable": false },
    "1": { "name": "Mask", "trackable": false },
    "2": { "name": "NO-Hardhat", "trackable": false },
    "3": { "name": "NO-Mask", "trackable": false },
    "4": { "name": "NO-Safety Vest", "trackable": false },
    "5": { "name": "Person", "trackable": true },
    "6": { "name": "Safety Cone", "trackable": false, "include_in_counts": false },
    "7": { "name": "Safety Vest", "trackable": false },
    "8": { "name": "machinery", "trackable": false, "include_in_counts": false },
    "9": { "name": "vehicle", "trackable": false, "include_in_counts": false }
  }
}
```

---

## Notes

- **`make local-build-up`** runs **`podman-compose down`**, which **does not** delete **`model_repo`**. Exported IR persists until you remove **`/<stem>`** on that volume or recreate the volume.
- If multiple Podman volumes look like model storage, always resolve **`VOL`** from **`podman inspect`** on the OVMS container’s **`/models`** mount, not from an arbitrary **`grep`**.
- Multiple weights: each **`app/models/foo.pt`** gets stem **`foo`** and directory **`/models/foo`**; replacing **one** stem only requires deleting **`/models/foo`** before re-up.
