# Training a Custom YOLO Model

This folder contains everything you need to train a YOLO model for object detection using your own images. The training container image bundles **`yolo_training.ipynb`** under **`training/`** in the notebook root; your **`upload/`** tree (images and labels) is something you add locally or by unpacking **`upload.tar.gz`** in Jupyter (see below).

## Overview

The Multimodal monitor can use a custom-trained YOLO model for object detection. This training workflow helps you:

1. Prepare images and labels in YOLO format
2. Build a dataset and train a model
3. Use the trained model with the multimodal monitor app

## Folder Structure

```
training/
├── yolo_training.ipynb        # Run this notebook (also seeded into the OpenShift training image)
├── upload/                    # Your dataset staging (not in the image; create locally or unpack upload.tar.gz)
│   ├── train_images/          # Training images
│   ├── train_labels/          # YOLO-format .txt labels
│   ├── val_images/            # Validation images
│   └── val_labels/            # Validation labels
└── README.md
```

## End-to-end workflow

1. **Gather images** for training and validation (different photos for each split).
2. **Label objects** with bounding boxes (recommended: [Label Studio](#labeling-with-label-studio-bounding-boxes); export **YOLO** format).
3. **Prepare your data**: Package files into `upload.tar.gz` with structure `upload/train_images/`, `upload/train_labels/`, `upload/val_images/`, `upload/val_labels/`. Upload the tarball via Jupyter and the notebook will auto-extract it (Section 1.5).
    Each image and its `.txt` label must share the **same base name**. Each `.txt` file contains YOLO annotations (normalized coordinates for object bounding boxes).
4. Run **`yolo_training.ipynb`** from **section 1** through **section 6** (what each section does is summarized in [Notebook sections (reference)](#notebook-sections-reference)).

## Labeling with Label Studio (bounding boxes)

These steps describe a typical workflow; menu names vary slightly across [Label Studio](https://labelstud.io/) versions.

1. Start Label Studio (**self-hosted** or **cloud**) and **create a project**.
2. **Choose a labeling template** suited to **object detection / bounding boxes** (rectangle labels on images).
3. **Define labels** — one name per object class (e.g. **Badge**, **Helmet**). Use names you will reuse in the notebook when prompted or in the **`CLASSES`** environment variable.
4. **Import** your images into the project.
5. For each task, draw **rectangles** over objects and assign the correct class; submit tasks until the set is annotated.
6. **Export annotations** using a **YOLO** export option (often packaged as a zip). Label Studio emits images and **`labels/`** with `.txt` files in YOLO normalized format.
7. **Split train vs validation** manually: copy a portion of pairs into **`upload/val_images/`** + **`upload/val_labels/`** and the rest into **`upload/train_*`**. Avoid using the same image in both splits.
8. **Align names** — the notebook matches images to labels by file base name (`photo.jpg` ↔ `photo.txt`). Rename if the export used different prefixes.

Alternatives — also export **YOLO** layout and arrange under **`upload/`**: [Makesense.ai](https://www.makesense.ai), [LabelImg](https://github.com/HumanSignal/labelImg).

## Dataset Requirements

### Label Format (YOLO)

Each `.txt` file has one line per object:

```
class_id  x_center  y_center  width  height
```

All coordinates are normalized (0–1). Images with no objects use an **empty** `.txt` file.

### Train vs. Validation

- **Train:** Images used to update model weights. Include both examples that contain targets and negatives (no objects), if applicable.
- **Validation:** Held-out images for evaluation. Must be **different** from train images. Include positives and negatives so metrics (mAP, precision, recall) are meaningful.

## Notebook sections (reference)

Matches **`yolo_training.ipynb`**.

The notebook uses **`WORKSPACE_ROOT = Path.cwd()`**. That directory must contain **`yolo_training.ipynb`** and (after you prepare data) a sibling **`upload/`** tree:

- **`upload/train_images/`**, **`upload/train_labels/`** (required for the copy step)
- **`upload/val_images/`**, **`upload/val_labels/`** (optional)

**On OpenShift**, open the notebook from **`training/yolo_training.ipynb`** so the kernel’s cwd is the **`training`** folder (e.g. **`~/notebooks/training`** or **`…/<notebookRootDir>/training`**). **Locally from Git**, start Jupyter under **`training/`** so cwd matches the repo layout.

**Environment variables** (optional; skip prompts in section 1):

- **`CLASSES`** — comma-separated names (default `Badge`)
- **`OUTPUT_ROOT`** — YOLO dataset root (default `./yolo_dataset`, resolved absolute)

**High-level flow** (same wording as the notebook intro):

1. **Step 1 (intro)** — **Section 1 (Configuration)**: install Ultralytics, collect config, create **`upload/…`** folders.
2. **Step 2 (intro)** — **Section 1.5 (Auto-Extract)**: if you uploaded **`upload.tar.gz`**, the notebook automatically extracts it.
3. **Step 3 (intro)** — Run **sections 2** through **5** without skipping: copy → label matching → YAML → train.

| Notebook section | What it does |
|---------------------|----------------|
| **1. Configuration & Create Upload Folders** | `%pip install ultralytics`; **`CLASSES`** / **`OUTPUT_ROOT`**; mkdir **`upload/…`**; prints paths. |
| **2. Auto-Extract Upload Archive** | Searches for **`upload.tar.gz`** in training folder or parent, automatically extracts it. Optional step (skip if uploading files manually). |
| **3. Create YOLO Structure & Copy from Upload Folders** | Builds **`OUTPUT_ROOT/images/*`** and **`labels/*`** from **`upload/`**. |
| **4. Label Matching** | Ensures each image has a matching `.txt` (empty file = negative example). |
| **5. Generate Dataset YAML** | Writes **`data.yaml`** under **`OUTPUT_ROOT`**. |
| **6. Train YOLO Model** | **`YOLO('yolov8n.pt').train(...)`** → **`runs/detect/badge-demo/weights/`**. |

### What the notebook code is doing (brief details)

1. **Section 1: Configuration and paths**
   - Installs `ultralytics` in the notebook kernel.
   - Reads `CLASSES` and `OUTPUT_ROOT` from environment variables if set; otherwise prompts with defaults.
   - Converts class names into an indexed mapping (`0 -> Badge`, etc.) used in YOLO `data.yaml`.
   - Resolves `WORKSPACE_ROOT = Path.cwd()` and defines `UPLOAD_ROOT = WORKSPACE_ROOT / "upload"`.
   - Ensures upload staging folders exist (`train_images`, `train_labels`, `val_images`, `val_labels`) and prints resolved paths.

2. **Section 2: Auto-extract upload archive (optional)**
   - Searches for `upload.tar.gz` in the training folder and one level up (parent directory).
   - If found, automatically extracts the tarball to `WORKSPACE_ROOT`, creating/updating the `upload/` directory structure.
   - Shows extraction summary with file counts for each subdirectory (train_images, train_labels, val_images, val_labels).
   - If not found, displays clear error message with search locations and next steps.
   - Optional: skip if you're uploading files directly to the `upload/` folders via Jupyter's file browser.

3. **Section 3: Build YOLO dataset tree and copy inputs**
   - Creates YOLO directory structure under `OUTPUT_ROOT`:
     `images/train`, `images/val`, `labels/train`, `labels/val`.
   - Scans your `upload/` folders and copies image files and label files into YOLO layout.
   - Uses file extension filtering and keeps original filenames so image-label base names can be matched later.

4. **Section 4: Label matching / negative examples**
   - Compares images and labels by base filename in each split.
   - For any image without a matching label file, creates an **empty** `.txt` label.
   - This preserves negative samples (images with no objects) and prevents training errors from missing labels.

5. **Section 5: Generate `data.yaml`**
   - Writes dataset metadata file at `OUTPUT_ROOT/data.yaml`.
   - Sets train/val image directories and the number of classes.
   - Emits class index-to-name mapping from `CLASSES`, which YOLO uses during training and metrics reporting.

6. **Section 6: Train YOLO**
   - Loads pretrained weights with `YOLO("yolov8n.pt")` (transfer learning starting point).
   - Calls `model.train(...)` using the generated `data.yaml` and notebook defaults (for example `epochs=100`, `imgsz=640`, run name `badge-demo`).
   - Writes outputs under `runs/detect/badge-demo/`, including checkpoints (`best.pt`, `last.pt`) and training artifacts/plots.

---

## Running the notebook: two approaches

Use **either** OpenShift Jupyter **or** local JupyterLab.

---

### 1. On OpenShift (Jupyter training pod)

#### Deploy (from repo root)

Deploy with your usual OpenShift flow (your environment may use **`.env`** for OpenAI); for example:

```bash
make deploy-openvino
```

Rebuild and push the training image when **`training/jupyter-training/Dockerfile`** or the bundled notebook changes:

```bash
make build-jupyter-training && make push-jupyter-training
```

#### Open JupyterLab and log in

With **`route.enabled`** (default), use **Networking → Routes** or **`oc get route -n <namespace>`**. TLS uses edge termination unless you override chart values.

Log in with the **`token`** stored in the auth Secret. If the chart creates it (you did not set **`jupyter-training.auth.existingSecretName`**), the Secret **`metadata.name`** is **`<parent Helm release>-jupyter-training-auth`**, from ```deploy/helm/jupyter-training/templates/secret.yaml``` (the **`jupyter-training.fullname`** helper plus **`auth`**). For **`helm install ppe-compliance-monitor …`**, that Secret is **`ppe-compliance-monitor-jupyter-training-auth`**. Use **secret name first**, then **`-n <your-namespace>`**, so the command is never parsed as **`oc get secret -n -jupyter-training-auth`** (missing namespace).

Retrieve the token (replace **`<your-namespace>`**):

```bash
oc get secret ppe-compliance-monitor-jupyter-training-auth \
  -n <your-namespace> \
  -o jsonpath='{.data.token}' | base64 -d
echo
```

If your **parent release name** is not **`ppe-compliance-monitor`**, substitute **`<your-parent-release>-jupyter-training-auth`** for the Secret name. Example pattern:

```bash
oc get secret <your-parent-release>-jupyter-training-auth \
  -n <your-namespace> \
  -o jsonpath='{.data.token}' | base64 -d
echo
```

If you use **`jupyter-training.auth.existingSecretName`**, substitute that Secret name instead.

> **Release note (security):** The default Jupyter auth token is **`changeme`** in both `deploy/helm/jupyter-training/values.yaml` and parent values (`deploy/helm/ppe-compliance-monitor/values.yaml`) when **`existingSecretName`** is empty. This is demo-only; production deployments must set a real secret via **`jupyter-training.auth.existingSecretName`** (or override **`jupyter-training.auth.token`** with a strong unique value).

#### Prepare `upload.tar.gz` on your laptop

The training image does **not** ship **`upload/`**. Build an archive whose entries start with **`upload/`**:

```bash
cd training
tar czf upload.tar.gz upload/
tar tzf upload.tar.gz | head   # expect upload/train_images/..., upload/train_labels/..., etc.
```

#### Upload the tarball in JupyterLab

Use **Upload** or drag-and-drop. You can place **`upload.tar.gz`** in either location:

- **Beside `training/`** (under the notebook root — the folder that contains **`training/`**), or
- **Inside `training/`** next to **`yolo_training.ipynb`** (works well if you already navigated into **`training`** in the file browser).

#### Auto-extract in the notebook

After uploading `upload.tar.gz`, simply run **Section 2** in the notebook. The cell will:
- Automatically search for `upload.tar.gz` in both locations (training folder and parent)
- Extract it to the correct location
- Show you what was extracted with file counts
- Provide clear error messages if the file isn’t found

#### Persistence (PVC)

Paths under the default **`notebookRootDir`** (often **`/tmp/jupyter-home/notebooks`**) may **not** survive pod restarts. The chart mounts the workspace PVC at **`/home/jovyan/work`** and exposes it as **`pvc-workspace`** in the tree. Copy **`runs/`**, **`best.pt`**, or **`OUTPUT_ROOT`** outputs you care about into **`/home/jovyan/work/...`** if you need them across restarts.

#### Run the notebook

Open **`training/yolo_training.ipynb`**, then run all cells **top to bottom** starting at **section 1**:
- **Section 1**: Configuration
- **Section 2**: Auto-extract `upload.tar.gz` (if you uploaded the tarball)
- **Sections 3-6**: Dataset creation and training

After **`upload/`** is populated (via section 1.5 or manual upload), continue through **section 5**.

---

### 2. Locally on your laptop

1. **Python environment** — Use a virtualenv or conda env with Python 3.11+ (or match your team standard).

2. **Install JupyterLab and Ultralytics:**
   ```bash
   pip install jupyterlab ultralytics
   ```

3. **Start Jupyter from the training directory** (so the notebook’s working directory matches `upload/`):
   ```bash
   cd training
   jupyter lab
   ```

4. **Open `yolo_training.ipynb`** and run the cells **top to bottom**.  
   The first code cell runs `%pip install ultralytics`.

5. **Data** — Complete [Label Studio](#labeling-with-label-studio-bounding-boxes) (or another tool), then arrange your images and YOLO labels under **`upload/`** subdirectories:
   - Create `upload.tar.gz` and upload it via Jupyter, then run **Section 2** to auto-extract
   
   Example label files live under **`upload/train_labels/`** and **`upload/val_labels/`** in the repo; image binaries are omitted from git. Run **sections 2–5** after your data is ready (summarized in [Notebook sections (reference)](#notebook-sections-reference)).

## Using Your Own Data

1. Replace or add images in `upload/train_images/` and `upload/val_images/`.
2. Create matching `.txt` label files in `upload/train_labels/` and `upload/val_labels/`.
3. Run the notebook. **Section 1** will prompt for class names if they differ from the default.

## Training Output

When training completes:

| Path | Description |
|------|-------------|
| `runs/detect/badge-demo/weights/best.pt` | Best model by validation metrics. Use for inference. |
| `runs/detect/badge-demo/weights/last.pt` | Final epoch checkpoint. |
| `runs/detect/badge-demo/results.png` | Loss and metrics plots. |

## Environment Variables (Optional)

Set these to skip prompts when running the notebook:

- `CLASSES` – Comma-separated class names (default: `Badge`)
- `OUTPUT_ROOT` – Output directory for YOLO dataset (default: `./yolo_dataset`)
