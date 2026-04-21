from flask import Flask, Response, request, jsonify, Blueprint
from flask_cors import CORS
import json
import os

# import cv2
# import time
from datetime import datetime

from tracing import init_tracing
from minio_client import (
    get_config_bucket,
    upload_bytes,
    get_object_stream,
    object_exists,
)

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
from video_processing.video_handler import VideoHandler

log = get_logger(__name__)

init_tracing()

# Minimum detection confidence to draw a box on the MJPEG video feed.

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

video_handler = VideoHandler()
if count_app_configs() == 0:
    insert_demo_configs()
log.info("VideoHandler initialized (video source selected from UI)")

llm_chat = LLMChat()

latest_description = "Initializing..."
latest_summary = "Processing video..."


@api.route("/video_feed")
def video_feed():
    """Video streaming route."""
    # feed_cfg = request.args.get("config")
    response = Response(
        video_handler.frame_generator(),
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
    """Return the latest description and summary."""
    global latest_description, latest_summary
    desc = video_handler.get_latested_description() or latest_description
    summary = video_handler.get_latest_summary() or latest_summary
    _cfg = video_handler._active_config_id
    _video = (
        (video_handler.video_source or "")[:200] if video_handler.video_source else ""
    )

    return jsonify(
        {
            "description": desc,
            "summary": summary,
            "inference_ready": True,
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
        desc = video_handler.get_latested_description() or latest_description
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
        video_handler.stop_streaming_if_active_config(config_id)
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
        video_handler.switch_video_source(video_source, config_id)
        return jsonify(
            {
                "message": "Active config set",
                "video_source": video_source,
                "active_config_id": video_handler._active_config_id,
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
