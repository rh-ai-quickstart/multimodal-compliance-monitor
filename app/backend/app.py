from flask import Flask, Response, request, jsonify, Blueprint
from flask_cors import CORS
import cv2
import json
import os
import time
from datetime import datetime

from tracing import init_tracing
from minio_client import (
    get_config_bucket,
    upload_bytes,
    get_object_stream,
    object_exists,
)
from multimodel import MultiModalAIDemo
from chat import LLMChat
from database import (
    count_app_configs,
    get_all_configs,
    get_classes_for_config,
    get_config_by_id,
    insert_config,
    delete_config,
    replace_detection_classes,
)
from seed_demo_configs import insert_demo_configs
from thumbnail_utils import generate_thumbnail_for_video_source, is_s3_video_path
from logger import get_logger

log = get_logger(__name__)

init_tracing()

# Minimum detection confidence to draw a box on the MJPEG video feed.
VIDEO_FEED_DRAW_MIN_CONF = 0.5

app = Flask(__name__)
api = Blueprint("api", __name__, url_prefix="/api")

cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000")
if cors_origins.strip() == "*":
    cors_allowed_origins = "*"
else:
    cors_allowed_origins = [
        origin.strip() for origin in cors_origins.split(",") if origin.strip()
    ]

CORS(app, resources={r"/*": {"origins": cors_allowed_origins}})


# Video source is selected dynamically by the user (MP4 or RTSP from config).
demo = MultiModalAIDemo()
demo.setup_components()
if count_app_configs() == 0:
    insert_demo_configs()
log.info("MultiModalAIDemo initialized (video source selected from UI)")

llm_chat = LLMChat()

latest_description = "Initializing..."
latest_summary = "Processing video..."


def generate_response_frames(client_remote=None, feed_config_param=None):
    """Generate MJPEG video stream for the /api/video_feed endpoint.

    This generator runs in a loop, fetching the latest frame and detections from
    the demo (which are produced by a separate inference process). It draws
    bounding boxes and labels on each frame, encodes as JPEG, and yields
    multipart MJPEG chunks for the browser to display.

    Flow:
        - get_frame_for_display() returns the latest frame + detections (never blocks on inference)
        - Draws boxes (cyan=person, green=compliant PPE, red=non-compliant)
        - Encodes frame as JPEG and yields multipart chunk (--frame + Content-Type + Content-Length + bytes)

    The display path is decoupled from inference: frames come from a reader thread,
    detections from an inference process. If no frame is available, the loop
    continues without yielding. Duplicate frames (same buffer epoch + frame_id) are skipped
    so reader read_count resets on source switch do not stall the stream; periodic resend
    keeps MJPEG alive if the reader temporarily stalls on the same id.
    """
    global latest_description, latest_summary
    none_count = 0
    last_none_log = 0.0
    frame_count = 0
    last_sent_key = (
        None  # (frame_epoch, frame_id) — epoch distinguishes source switches
    )
    last_jpeg_wall_s = 0.0
    try:
        while True:
            frame, detections, frame_id, frame_epoch = demo.get_frame_for_display(
                resize_to=(1920, 1080)
            )
            if frame is None:
                none_count += 1
                now = time.time()
                if now - last_none_log >= 5.0:
                    log.warning(
                        "Video feed: no frame to display (%d times in ~5s, %d JPEGs sent)",
                        none_count,
                        frame_count,
                    )
                    last_none_log = now
                    none_count = 0
                time.sleep(0.02)
                continue

            dup_key = (
                (frame_epoch, frame_id)
                if frame_id is not None and frame_epoch >= 0
                else None
            )
            now_wall = time.time()
            mjpeg_keepalive = (
                dup_key is not None
                and dup_key == last_sent_key
                and last_jpeg_wall_s > 0
                and (now_wall - last_jpeg_wall_s) >= 0.35
            )
            if dup_key is not None and dup_key == last_sent_key and not mjpeg_keepalive:
                time.sleep(0.001)  # Avoid tight loop when waiting for new frame
                continue

            try:
                annotated_frame = frame.copy()
            except Exception as e:
                log.exception("Video feed: frame.copy() failed: %s", e)
                continue
            h_frame, w_frame = annotated_frame.shape[:2]
            # Draw bounding boxes and labels for each detection (Person, PPE items)
            line_type = cv2.LINE_AA
            for detection in detections:
                x1, y1, x2, y2 = detection["bbox"]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                x1 = max(0, min(x1, w_frame - 1))
                y1 = max(0, min(y1, h_frame - 1))
                x2 = max(0, min(x2, w_frame - 1))
                y2 = max(0, min(y2, h_frame - 1))
                if x1 >= x2 or y1 >= y2:
                    continue
                conf = detection["confidence"]
                currentClass = detection["class_name"]
                if conf > VIDEO_FEED_DRAW_MIN_CONF:
                    if detection.get("track_id") is not None:
                        color = (0, 255, 255)  # Cyan for tracked targets
                    elif currentClass in ["NO-Hardhat", "NO-Safety Vest", "NO-Mask"]:
                        color = (0, 0, 255)  # Red for non-compliance
                    elif currentClass in ["Hardhat", "Safety Vest", "Mask"]:
                        color = (0, 255, 0)  # Green for compliance
                    else:
                        color = (255, 255, 0)  # Yellow for other objects
                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2,
                        lineType=line_type,
                    )
                    label = f"{currentClass} {conf:.2f}"
                    if detection.get("track_id") is not None:
                        label = f"{currentClass} #{detection['track_id']} {conf:.2f}"
                    text_size = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
                    )[0]
                    label_y1 = max(0, y1 - text_size[1] - 10)
                    label_y2 = y1
                    label_x2 = min(w_frame, x1 + text_size[0])
                    if label_x2 > x1 and label_y2 > label_y1:
                        cv2.rectangle(
                            annotated_frame,
                            (x1, label_y1),
                            (label_x2, label_y2),
                            color,
                            -1,
                            lineType=line_type,
                        )
                    text_y = max(label_y1 + text_size[1] - 2, y1 - 5)
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 0),
                        2,
                        lineType=line_type,
                    )

            # Read from shared state (updated by inference thread)
            with demo._display_lock:
                if demo._display_description:
                    latest_description = demo._display_description
                latest_summary = (
                    demo._display_summary or demo.latest_summary or latest_summary
                )

            ret, buffer = cv2.imencode(
                ".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            if not ret:
                log.warning(
                    "Video feed: cv2.imencode failed shape=%s",
                    getattr(annotated_frame, "shape", None),
                )
                continue
            frame_bytes = buffer.tobytes()
            frame_count += 1
            last_sent_key = dup_key
            last_jpeg_wall_s = time.time()
            try:
                # Content-Length helps Chrome parse each part correctly (avoids distortion from boundary misparsing)
                header = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n"
                )
                yield header + frame_bytes + b"\r\n"
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                log.warning(
                    "Video feed: client disconnected during yield: %s (frames_sent=%s)",
                    e,
                    frame_count,
                )
                break
    except Exception as e:
        log.exception("Video feed: exception in stream loop: %s", e)


@api.route("/video_feed")
def video_feed():
    """Video streaming route."""
    feed_cfg = request.args.get("config")
    response = Response(
        generate_response_frames(
            request.remote_addr,
            feed_config_param=feed_cfg,
        ),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
    # Disable proxy buffering to reduce periodic pauses (OpenShift/HAProxy)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["X-Accel-Buffering"] = "no"
    return response


@api.route("/")
def api_root():
    """Simple health response for the API root."""
    return jsonify({"status": "ok"})


@api.route("/latest_info")
def latest_info():
    """Return the latest description and summary. Reads from shared state (no inference trigger)."""
    global latest_description, latest_summary
    with demo._display_lock:
        if demo._display_description:
            latest_description = demo._display_description
        latest_summary = demo._display_summary or demo.latest_summary or latest_summary
        results_received = demo._results_received_count > 0
        _cfg = demo._active_config_id
    process_ready = (
        demo._inference_ready_event is not None and demo._inference_ready_event.is_set()
    )
    # Clear the "Loading model" overlay once inference has bound the pipeline OR produced
    # a result. Relying only on the first result can leave the UI blank if OVMS is slow or
    # the first frame is still in flight after INIT/RELOAD.
    ui_ready = results_received or process_ready
    _video = (
        (demo.video_source or "")[:200] if getattr(demo, "video_source", None) else ""
    )
    return jsonify(
        {
            "description": latest_description,
            "summary": latest_summary,
            "inference_ready": ui_ready,
            "inference_process_ready": process_ready,
            "active_config_id": _cfg,
            "video_source": _video,
        }
    )


@api.route("/chat", methods=["POST"])
def chat():
    """Answer a question based on latest description and summary.

    Supports streaming via Server-Sent Events when ``stream=true`` is sent in
    the JSON body.  An optional ``session_id`` field enables per-session
    conversation memory (defaults to ``"default"``).

    When ``app_config_id`` is provided, all SQL queries are scoped to that
    config's detection data (enforced at the tool level).
    """
    global latest_description, latest_summary
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Field 'question' is required."}), 400

    session_id = (data.get("session_id") or "default").strip()
    user_description = (data.get("description") or "").strip()

    if user_description:
        desc = user_description
    else:
        with demo._display_lock:
            desc = demo._display_description or latest_description
    context = desc.replace("Detected: ", "", 1)

    app_config_id = data.get("app_config_id")
    classes_info = None
    if app_config_id is not None:
        try:
            app_config_id = int(app_config_id)
        except (ValueError, TypeError):
            return jsonify({"error": "app_config_id must be an integer"}), 400
        raw = get_classes_for_config(app_config_id)
        classes_info = [
            {"name": v["name"], "trackable": v["trackable"]} for v in raw.values()
        ]
        log.debug(f"classes_info: {classes_info}")

    try:
        answer = llm_chat.chat(
            question=question,
            context=context,
            session_id=session_id,
            app_config_id=app_config_id,
            classes_info=classes_info,
        )
    except Exception as e:
        log.exception("chat: LLM error: %s", e)
        return jsonify({"error": f"LLM error: {e}"}), 500

    return jsonify({"answer": answer})


@api.route("/chat/reset", methods=["POST"])
def chat_reset():
    """Clear the LLM conversation memory for a given session."""
    data = request.get_json(silent=True) or {}
    session_id = (data.get("session_id") or "").strip()
    if not session_id:
        return jsonify({"error": "Field 'session_id' is required."}), 400
    llm_chat.clear_history(session_id)
    log.info("chat_reset: cleared session %r", session_id)
    return jsonify({"message": "Session cleared"})


def _parse_classes(value: str | dict) -> tuple[dict, list[tuple[int, str, bool, bool]]]:
    """
    Parse classes from new JSON format. Returns (mapping, entries).
    Format: {"0":{"name":"Person","trackable":true,"include_in_counts":true}, ...}
    include_in_counts defaults to true when omitted.
    mapping: {"0":"Person","1":"Hardhat"} for app_config.classes
    entries: [(model_class_index, name, trackable, include_in_counts), ...]
    """
    if value is None:
        raise ValueError("Classes cannot be empty")
    obj = value if isinstance(value, dict) else json.loads(str(value).strip())
    if not isinstance(obj, dict):
        raise ValueError(
            'Classes must be an object like {"0":{"name":"Person","trackable":true}}'
        )
    if not obj:
        raise ValueError("Classes cannot be empty")
    mapping = {}
    entries = []
    for idx_str, v in obj.items():
        if not isinstance(v, dict):
            raise ValueError(
                f'Class "{idx_str}" must be an object with "name" and "trackable"'
            )
        name = v.get("name")
        if not name or not str(name).strip():
            raise ValueError(f'Class "{idx_str}" must have a non-empty "name"')
        name = str(name).strip()
        trackable = bool(v.get("trackable", False))
        include_in_counts = bool(v.get("include_in_counts", True))
        try:
            model_class_index = int(idx_str)
        except ValueError:
            raise ValueError(f'Class key "{idx_str}" must be an integer (model index)')
        mapping[idx_str] = name
        entries.append((model_class_index, name, trackable, include_in_counts))
    return mapping, entries


@api.route("/config", methods=["GET"])
def config_list():
    """List all app configs."""
    try:
        configs = get_all_configs()
        # Ensure classes is JSON-serializable (PostgreSQL JSONB may return dict)
        for c in configs:
            if isinstance(c.get("created_at"), datetime):
                c["created_at"] = c["created_at"].isoformat()
        return jsonify(configs)
    except Exception as e:
        log.exception("config_list: %s", e)
        return jsonify({"error": str(e)}), 500


@api.route("/config", methods=["POST"])
def config_create():
    """Create a new app config."""
    data = request.get_json(silent=True) or {}
    model_url = (data.get("model_url") or "").strip()
    model_name = (data.get("model_name") or "").strip()
    video_source = (data.get("video_source") or "").strip()
    classes_raw = data.get("classes")
    if classes_raw is None:
        return jsonify({"error": "Field 'classes' is required"}), 400
    try:
        classes, entries = _parse_classes(classes_raw)
    except (ValueError, json.JSONDecodeError) as e:
        return jsonify({"error": str(e)}), 400
    if not model_url:
        return jsonify({"error": "Field 'model_url' is required"}), 400
    if not model_name:
        return jsonify({"error": "Field 'model_name' is required"}), 400
    if not video_source:
        return jsonify({"error": "Field 'video_source' is required"}), 400
    try:
        config_id = insert_config(model_url, video_source, model_name)
        replace_detection_classes(config_id, entries)
        if is_s3_video_path(video_source):
            generate_thumbnail_for_video_source(video_source)
        return jsonify({"id": config_id, "message": "Config created"}), 201
    except Exception as e:
        log.exception("config_create: %s", e)
        return jsonify({"error": str(e)}), 500


@api.route("/config/<int:config_id>", methods=["DELETE"])
def config_delete(config_id):
    """Delete an app config and all dependent rows (classes, tracks, observations)."""
    try:
        cfg = get_config_by_id(config_id)
        if not cfg:
            return jsonify({"error": "Config not found"}), 404
        demo.stop_streaming_if_active_config(config_id)
        deleted = delete_config(config_id)
        if not deleted:
            return jsonify({"error": "Config not found"}), 404
        return jsonify({"message": "Config deleted"})
    except Exception as e:
        log.exception(f"config_delete: {e}")
        return jsonify({"error": str(e)}), 500


@api.route("/active_config", methods=["POST"])
def active_config_set():
    """Set the active video source from a config. Switches to that config's video (MP4 path or RTSP URL)."""
    data = request.get_json(silent=True) or {}
    config_id = data.get("config_id")
    if config_id is None:
        return jsonify({"error": "Field 'config_id' is required"}), 400
    try:
        config_id = int(config_id)
    except (ValueError, TypeError):
        return jsonify({"error": "config_id must be an integer"}), 400
    config = get_config_by_id(config_id)
    if not config:
        return jsonify({"error": "Config not found"}), 404
    video_source = (config.get("video_source") or "").strip()
    if not video_source:
        return jsonify({"error": "Config has no video source"}), 400
    try:
        demo.start_streaming(video_source, config_id=config_id)
        return jsonify(
            {
                "message": "Active config set",
                "video_source": video_source,
                "active_config_id": demo._active_config_id,
            }
        )
    except Exception as e:
        log.exception("active_config_set: %s", e)
        return jsonify({"error": str(e)}), 500


# Config storage: MinIO only (enables horizontal scaling)
log.info("Config storage: MinIO bucket=%s", get_config_bucket())


@api.route("/thumbnails/<path:filename>")
def serve_thumbnail(filename):
    """Serve a thumbnail image by filename (e.g. video.jpg) from MinIO."""
    if ".." in filename or "/" in filename:
        return jsonify({"error": "Invalid filename"}), 400
    if not filename.lower().endswith(".jpg"):
        return jsonify({"error": "Only .jpg thumbnails are served"}), 400
    thumb_key = f"thumbnails/{filename}"
    if object_exists(get_config_bucket(), thumb_key):
        try:
            resp = get_object_stream(get_config_bucket(), thumb_key)
            try:
                data = resp.read()
                return Response(data, mimetype="image/jpeg")
            finally:
                resp.close()
                resp.release_conn()
        except Exception as e:
            log.exception("serve_thumbnail: %s", e)
            return jsonify({"error": "Failed to load thumbnail"}), 500
    return jsonify({"error": "Thumbnail not found"}), 404


@api.route("/config/upload", methods=["POST"])
def config_upload():
    """Upload a video file to MinIO. Returns S3 URI for video_source (e.g. s3://config/uploads/filename.mp4)."""
    if "file" not in request.files:
        return jsonify({"error": "No file in request"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No filename"}), 400
    if not f.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        return jsonify(
            {"error": "Only video files (mp4, avi, mov, mkv) are allowed"}
        ), 400
    safe_name = os.path.basename(f.filename)
    try:
        bucket = get_config_bucket()
        object_key = f"uploads/{safe_name}"
        data = f.read()
        upload_bytes(bucket, object_key, data, content_type="video/mp4")
        path = f"s3://{bucket}/{object_key}"
        return jsonify({"path": path, "filename": safe_name})
    except Exception as e:
        log.exception("config_upload: %s", e)
        return jsonify({"error": str(e)}), 500


app.register_blueprint(api)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8888"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
