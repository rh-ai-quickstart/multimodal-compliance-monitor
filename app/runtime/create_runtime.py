import os
import json
import time
import requests
from kubernetes import client, config
from kubernetes.client.rest import ApiException

OVMS_IMAGE = (
    "registry.redhat.io/rhoai/odh-openvino-model-server-rhel9"
    "@sha256:7daebf4a4205b9b81293932a1dfec705f59a83de8097e468e875088996c1f224"
)


def load_config():
    """Load deployment configuration from environment variables."""
    namespace = os.getenv("NAMESPACE", "ppe-compliance-monitor")

    runtime_args_raw = os.getenv("RUNTIME_ARGS", "[]")
    runtime_command_raw = os.getenv("RUNTIME_COMMAND", "[]")
    runtime_env_raw = os.getenv("RUNTIME_ENV", "{}")
    try:
        runtime_args = json.loads(runtime_args_raw)
    except (json.JSONDecodeError, TypeError):
        runtime_args = []
    try:
        runtime_command = json.loads(runtime_command_raw)
    except (json.JSONDecodeError, TypeError):
        runtime_command = []
    try:
        runtime_env = json.loads(runtime_env_raw)
    except (json.JSONDecodeError, TypeError):
        runtime_env = {}

    return {
        "namespace": namespace,
        "runtime_type": os.getenv("RUNTIME_TYPE", "openvino").lower(),
        "deploy_enabled": os.getenv("DEPLOY_MODEL", "true").lower() == "true",
        "minio_access_key": os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        "minio_secret_key": os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        "serving_runtime": os.getenv("SERVING_RUNTIME", f"{namespace}-deploy"),
        "create_serving_runtime": os.getenv("CREATE_SERVING_RUNTIME", "true").lower()
        == "true",
        "runtime_image": os.getenv("SERVING_RUNTIME_IMAGE", OVMS_IMAGE),
        "rest_port": int(os.getenv("REST_PORT", "8888")),
        "grpc_port": int(os.getenv("GRPC_PORT", "9000")),
        "runtime_args": runtime_args,
        "runtime_command": runtime_command,
        "runtime_env": runtime_env,
        "runtime_template_name": os.getenv("RUNTIME_TEMPLATE_NAME", ""),
        "runtime_template_display_name": os.getenv("RUNTIME_TEMPLATE_DISPLAY_NAME", ""),
        "model_format": os.getenv("MODEL_FORMAT", "pytorch"),
        "model_format_version": os.getenv("MODEL_FORMAT_VERSION", "2"),
        "deploy_from_registry": os.getenv("DEPLOY_FROM_REGISTRY", "false").lower()
        == "true",
        "model_registry_url": os.getenv("MODEL_REGISTRY_URL", ""),
        "model_name": os.getenv("MODEL_NAME", "ppe"),
        "model_version": os.getenv("MODEL_VERSION", "1"),
        "s3_bucket": os.getenv("S3_BUCKET", "models"),
        "s3_model_path": os.getenv("S3_MODEL_PATH", "ppe/"),
        "minio_endpoint": os.getenv(
            "MINIO_ENDPOINT", f"http://minio.{namespace}.svc.cluster.local:9000"
        ),
        "model_version_to_deploy": os.getenv("MODEL_VERSION_TO_DEPLOY", ""),
        "replicas_min": int(os.getenv("REPLICAS_MIN", "1")),
        "replicas_max": int(os.getenv("REPLICAS_MAX", "1")),
        "resources": {
            "requests": {
                "cpu": os.getenv("RESOURCE_REQ_CPU", "2"),
                "memory": os.getenv("RESOURCE_REQ_MEMORY", "4Gi"),
            },
            "limits": {
                "cpu": os.getenv("RESOURCE_LIM_CPU", "2"),
                "memory": os.getenv("RESOURCE_LIM_MEMORY", "4Gi"),
            },
        },
        "gpu_enabled": os.getenv("GPU_ENABLED", "false").lower() == "true",
        "gpu_count": os.getenv("GPU_COUNT", "1"),
        "gpu_tolerations": json.loads(os.getenv("GPU_TOLERATIONS", "[]")),
    }


def load_model_info_from_s3(cfg):
    """Build model deployment info from S3 bucket environment variables."""
    return {
        "model_name": cfg["model_name"],
        "model_version": cfg["model_version"],
        "bucket": cfg["s3_bucket"],
        "model_path": cfg["s3_model_path"],
        "minio_endpoint": cfg["minio_endpoint"],
    }


def load_model_info_from_registry(cfg):
    """Fetch model deployment info from Model Registry."""
    api_base = f"{cfg['model_registry_url']}/api/model_registry/v1alpha3"
    model_name = cfg["model_name"]
    version_override = cfg["model_version_to_deploy"]

    print("Deploying from Model Registry:")
    print(f"  Registry URL: {cfg['model_registry_url']}")
    print(f"  Model Name: {model_name}")
    print(f"  Version: {version_override or 'latest'}")

    print(f"Searching for registered model: {model_name}")
    response = requests.get(f"{api_base}/registered_model", params={"name": model_name})

    if response.status_code == 404:
        raise RuntimeError(f"Model '{model_name}' not found in registry")
    response.raise_for_status()

    registered_model = response.json()
    registered_model_id = registered_model.get("id")
    if not registered_model_id:
        raise RuntimeError(
            f"Model found but 'id' field is missing. Response: {registered_model}"
        )
    print(f"Found registered model: {registered_model_id}")

    versions_response = requests.get(
        f"{api_base}/registered_models/{registered_model_id}/versions"
    )
    if not versions_response.ok:
        print(f"Failed to get versions: {versions_response.status_code}")
        versions_response.raise_for_status()

    versions = versions_response.json().get("items", [])
    print(f"Found {len(versions)} versions")

    if not versions:
        raise RuntimeError(f"No versions found for model '{model_name}'")

    target_version = _find_model_version(versions, version_override)
    model_version = target_version["name"]
    version_id = target_version["id"]
    print(f"Using version: {model_version} (id: {version_id})")

    storage_uri, minio_endpoint = _extract_storage_info(
        api_base, target_version, version_id
    )

    if not storage_uri:
        raise RuntimeError("Could not find storage URI in model registry")

    bucket, model_path = _parse_s3_uri(storage_uri)

    if not minio_endpoint:
        minio_endpoint = cfg["minio_endpoint"]

    print("Resolved deployment info from registry:")
    print(f"  Bucket: {bucket}, Path: {model_path}")
    print(f"  MinIO Endpoint: {minio_endpoint}")

    return {
        "model_name": model_name,
        "model_version": model_version,
        "bucket": bucket,
        "model_path": model_path,
        "minio_endpoint": minio_endpoint,
    }


def _find_model_version(versions, version_override):
    """Find specific version or return latest."""
    if version_override:
        for v in versions:
            if v["name"] == version_override:
                return v
        raise RuntimeError(f"Version '{version_override}' not found")
    return sorted(
        versions, key=lambda x: x.get("createTimeSinceEpoch", "0"), reverse=True
    )[0]


def _extract_storage_info(api_base, target_version, version_id):
    """Extract storage URI and MinIO endpoint from version metadata."""
    custom_props = target_version.get("customProperties", {})
    storage_uri = custom_props.get("storage_uri", {}).get("string_value", "")
    minio_endpoint = custom_props.get("minio_endpoint", {}).get("string_value", "")

    if not storage_uri:
        artifacts_response = requests.get(
            f"{api_base}/model_versions/{version_id}/artifacts"
        )
        if artifacts_response.ok:
            artifacts = artifacts_response.json().get("items", [])
            print(f"Found {len(artifacts)} artifacts")
            if artifacts:
                storage_uri = artifacts[0].get("uri", "")

    return storage_uri, minio_endpoint


def _parse_s3_uri(storage_uri):
    """Parse S3 URI into bucket and path."""
    print(f"Storage URI: {storage_uri}")
    if not storage_uri.startswith("s3://"):
        raise RuntimeError(f"Unsupported storage URI format: {storage_uri}")
    uri_parts = storage_uri[5:].split("/", 1)
    bucket = uri_parts[0]
    model_path = uri_parts[1] if len(uri_parts) > 1 else ""
    return bucket, model_path


def init_kubernetes_client():
    """Initialize Kubernetes API clients."""
    try:
        config.load_incluster_config()
        print("Loaded in-cluster Kubernetes config")
    except config.ConfigException:
        config.load_kube_config()
        print("Loaded kubeconfig from local")
    return client.CoreV1Api(), client.CustomObjectsApi()


def create_or_update_resource(create_fn, update_fn, resource_name):
    """Create a K8s resource, or update if it already exists."""
    try:
        create_fn()
        print(f"{resource_name} created")
    except ApiException as e:
        if e.status == 409:
            update_fn()
            print(f"{resource_name} updated")
        else:
            raise


def create_storage_secret(core_v1, cfg, model_info):
    """Create the storage-config secret for S3/MinIO access.

    Matches the Helm template structure: the data connection is keyed
    by namespace so KServe can look it up via storage.key.
    """
    print("\nCreating storage-config secret...")

    storage_config_json = json.dumps(
        {
            "type": "s3",
            "access_key_id": cfg["minio_access_key"],
            "secret_access_key": cfg["minio_secret_key"],
            "endpoint_url": model_info["minio_endpoint"],
            "bucket": model_info["bucket"],
            "region": "",
        }
    )

    secret = client.V1Secret(
        api_version="v1",
        kind="Secret",
        metadata=client.V1ObjectMeta(
            name="storage-config",
            namespace=cfg["namespace"],
            labels={
                "opendatahub.io/dashboard": "true",
            },
        ),
        type="Opaque",
        string_data={
            cfg["namespace"]: storage_config_json,
        },
    )

    create_or_update_resource(
        lambda: core_v1.create_namespaced_secret(cfg["namespace"], secret),
        lambda: core_v1.replace_namespaced_secret(
            "storage-config", cfg["namespace"], secret
        ),
        "Storage config secret",
    )

    try:
        core_v1.read_namespaced_secret("storage-config", cfg["namespace"])
        print("  Verified: storage-config secret exists")
    except ApiException as e:
        print(f"  WARNING: storage-config secret not found after creation: {e.status}")
        raise RuntimeError("storage-config secret was not persisted")


def create_service_account(core_v1, cfg):
    """Create service account used by the InferenceService."""
    sa_name = f"{cfg['serving_runtime']}-sa"
    print(f"\nCreating service account: {sa_name}...")

    sa = client.V1ServiceAccount(
        api_version="v1",
        kind="ServiceAccount",
        metadata=client.V1ObjectMeta(
            name=sa_name,
            namespace=cfg["namespace"],
        ),
    )

    create_or_update_resource(
        lambda: core_v1.create_namespaced_service_account(cfg["namespace"], sa),
        lambda: core_v1.patch_namespaced_service_account(sa_name, cfg["namespace"], sa),
        f"Service account {sa_name}",
    )
    return sa_name


# def _parse_cpu_value(cpu_str):
#     """Parse Kubernetes CPU string to integer core count."""
#     if cpu_str.endswith("m"):
#         return max(1, int(cpu_str[:-1]) // 1000)
#     return int(cpu_str)


def _build_ovms_args(cfg):
    """Build OVMS container args optimized for low-concurrency REST inference."""
    # num_cpus = _parse_cpu_value(cfg["resources"]["limits"]["cpu"])
    plugin_config = json.dumps(
        {
            "PERFORMANCE_HINT": "THROUGHPUT",
        }
    )
    return [
        "--model_name={{.Name}}",
        f"--port={cfg['grpc_port']}",
        f"--rest_port={cfg['rest_port']}",
        "--model_path=/mnt/models",
        "--file_system_poll_wait_seconds=0",
        "--metrics_enable",
        "--nireq=2",
        "--rest_workers=2",
        "--rest_bind_address=0.0.0.0",
        "--cache_dir=/tmp/ovms_cache",
        f"--plugin_config={plugin_config}",
        # f"--rest_workers={max(num_cpus, 4)}",
    ]


def build_serving_runtime_spec(cfg):
    """Build the OVMS ServingRuntime specification.

    Produces the same resource structure as the Helm template in
    deploy/helm/ppe-compliance-monitor/templates/openshift-runetime.yaml.
    """
    return {
        "apiVersion": "serving.kserve.io/v1alpha1",
        "kind": "ServingRuntime",
        "metadata": {
            "name": cfg["serving_runtime"],
            "namespace": cfg["namespace"],
            "annotations": {
                "opendatahub.io/apiProtocol": "REST",
                "opendatahub.io/recommended-accelerators": '["nvidia.com/gpu"]',
                "opendatahub.io/serving-runtime-scope": "global",
                "opendatahub.io/template-display-name": "OpenVINO Model Server",
                "opendatahub.io/template-name": "kserve-ovms",
                "openshift.io/display-name": "OpenVINO Model Server",
            },
            "labels": {
                "opendatahub.io/dashboard": "true",
            },
        },
        "spec": {
            "annotations": {
                "opendatahub.io/kserve-runtime": "ovms",
                "prometheus.io/path": "/metrics",
                "prometheus.io/port": str(cfg["rest_port"]),
            },
            "containers": [
                {
                    "name": "kserve-container",
                    "image": cfg["runtime_image"],
                    "args": _build_ovms_args(cfg),
                    "ports": [
                        {
                            "containerPort": cfg["rest_port"],
                            "protocol": "TCP",
                        }
                    ],
                }
            ],
            "multiModel": False,
            "protocolVersions": ["v2", "grpc-v2"],
            "supportedModelFormats": [
                {"name": "openvino_ir", "version": "opset13", "autoSelect": True},
                {"name": "onnx", "version": "1"},
                {"name": "tensorflow", "version": "1", "autoSelect": True},
                {"name": "tensorflow", "version": "2", "autoSelect": True},
                {"name": "paddle", "version": "2", "autoSelect": True},
                {"name": "pytorch", "version": "2", "autoSelect": True},
            ],
        },
    }


def _build_triton_args(cfg):
    """Build Triton Inference Server container args for KServe single-model serving."""
    return [
        "tritonserver",
        "--model-store=/mnt/models",
        f"--http-port={cfg['rest_port']}",
        f"--grpc-port={cfg['grpc_port']}",
        "--allow-http=true",
        "--allow-grpc=true",
    ]


def build_kserve_serving_runtime_spec(cfg):
    """Build a KServe ServingRuntime using NVIDIA Triton Inference Server.

    GPU-accelerated ONNX serving with V2/OIP REST protocol.
    Configurable via values.yaml (image, args, command, env).
    """
    template_name = cfg.get("runtime_template_name") or "kserve-tritonserver"
    template_display = (
        cfg.get("runtime_template_display_name") or "NVIDIA Triton Inference Server"
    )

    container = {
        "name": "kserve-container",
        "image": cfg["runtime_image"],
        "args": cfg.get("runtime_args") or _build_triton_args(cfg),
        "ports": [
            {
                "containerPort": cfg["rest_port"],
                "protocol": "TCP",
            }
        ],
        "volumeMounts": [
            {"name": "shm", "mountPath": "/dev/shm"},
        ],
    }
    if cfg.get("runtime_command"):
        container["command"] = cfg["runtime_command"]

    env_list = [{"name": k, "value": v} for k, v in cfg.get("runtime_env", {}).items()]
    if env_list:
        container["env"] = env_list

    return {
        "apiVersion": "serving.kserve.io/v1alpha1",
        "kind": "ServingRuntime",
        "metadata": {
            "name": cfg["serving_runtime"],
            "namespace": cfg["namespace"],
            "annotations": {
                "opendatahub.io/apiProtocol": "REST",
                "opendatahub.io/recommended-accelerators": '["nvidia.com/gpu"]',
                "opendatahub.io/serving-runtime-scope": "global",
                "opendatahub.io/template-display-name": template_display,
                "opendatahub.io/template-name": template_name,
                "openshift.io/display-name": template_display,
            },
            "labels": {
                "opendatahub.io/dashboard": "true",
            },
        },
        "spec": {
            "annotations": {
                "prometheus.io/path": "/metrics",
                "prometheus.io/port": str(cfg["rest_port"]),
            },
            "containers": [container],
            "multiModel": False,
            "protocolVersions": ["v2", "grpc-v2"],
            "volumes": [
                {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "2Gi"}},
            ],
            "supportedModelFormats": [
                {"name": "onnx", "version": "1", "autoSelect": True},
                {"name": "tensorflow", "version": "1", "autoSelect": True},
                {"name": "tensorflow", "version": "2", "autoSelect": True},
                {"name": "pytorch", "version": "1", "autoSelect": True},
                {"name": "tensorrt", "version": "8", "autoSelect": True},
            ],
        },
    }


def create_serving_runtime(custom_api, cfg):
    """Create or update the ServingRuntime."""
    if not cfg["create_serving_runtime"]:
        print(f"\nUsing existing ServingRuntime: {cfg['serving_runtime']}")
        return

    print(f"\nCreating ServingRuntime: {cfg['serving_runtime']}...")
    print(f"  Using image: {cfg['runtime_image']}")

    if cfg["runtime_type"] == "kserve":
        spec = build_kserve_serving_runtime_spec(cfg)
    else:
        spec = build_serving_runtime_spec(cfg)

    create_or_update_resource(
        lambda: custom_api.create_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1alpha1",
            namespace=cfg["namespace"],
            plural="servingruntimes",
            body=spec,
        ),
        lambda: custom_api.patch_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1alpha1",
            namespace=cfg["namespace"],
            plural="servingruntimes",
            name=cfg["serving_runtime"],
            body=spec,
        ),
        f"ServingRuntime {cfg['serving_runtime']}",
    )


def build_inference_service_spec(cfg, model_info, sa_name):
    """Build the InferenceService specification.

    Uses storage.key (namespace) + storage.path pattern matching the Helm
    template, with RawDeployment mode and OpenDataHub annotations.
    When GPU is enabled, adds nvidia.com/gpu resources and node tolerations.
    """
    isvc_name = cfg["model_name"]

    resources = cfg["resources"]
    if cfg.get("gpu_enabled"):
        gpu_res = {"nvidia.com/gpu": cfg.get("gpu_count", "1")}
        resources = {
            "requests": {**resources["requests"], **gpu_res},
            "limits": {**resources["limits"], **gpu_res},
        }

    predictor = {
        "automountServiceAccountToken": False,
        "deploymentStrategy": {"type": "RollingUpdate"},
        "minReplicas": cfg["replicas_min"],
        "maxReplicas": cfg["replicas_max"],
        "model": {
            "modelFormat": {
                "name": cfg["model_format"],
                "version": cfg["model_format_version"],
            },
            "name": "",
            "resources": resources,
            "runtime": cfg["serving_runtime"],
            "storage": {
                "key": cfg["namespace"],
                "path": model_info["model_path"],
            },
        },
        "serviceAccountName": sa_name,
    }

    if cfg.get("gpu_tolerations"):
        predictor["tolerations"] = cfg["gpu_tolerations"]

    return {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": isvc_name,
            "namespace": cfg["namespace"],
            "annotations": {
                "security.opendatahub.io/enable-auth": "false",
                "openshift.io/display-name": cfg["serving_runtime"],
                "serving.kserve.io/deploymentMode": "RawDeployment",
                "opendatahub.io/model-type": "predictive",
                "model-version": model_info["model_version"],
            },
            "labels": {
                "opendatahub.io/dashboard": "true",
            },
        },
        "spec": {
            "predictor": predictor,
        },
    }


def create_inference_service(custom_api, cfg, model_info, sa_name):
    """Create or update the InferenceService."""
    print("\nDeploying InferenceService...")

    spec = build_inference_service_spec(cfg, model_info, sa_name)
    isvc_name = spec["metadata"]["name"]

    create_or_update_resource(
        lambda: custom_api.create_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=cfg["namespace"],
            plural="inferenceservices",
            body=spec,
        ),
        lambda: custom_api.patch_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=cfg["namespace"],
            plural="inferenceservices",
            name=isvc_name,
            body=spec,
        ),
        f"InferenceService {isvc_name}",
    )
    return isvc_name


def wait_for_inference_service(custom_api, cfg, isvc_name, timeout_seconds=300):
    """Wait for InferenceService to become ready."""
    print("\nWaiting for InferenceService to be ready...")

    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        try:
            isvc = custom_api.get_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=cfg["namespace"],
                plural="inferenceservices",
                name=isvc_name,
            )

            status = isvc.get("status", {})
            for condition in status.get("conditions", []):
                if condition["type"] == "Ready" and condition["status"] == "True":
                    url = status.get("url", "N/A")
                    print("\nInferenceService is ready!")
                    print(f"Inference endpoint: {url}")
                    return True

            elapsed = int(time.time() - start_time)
            print(f"Waiting for InferenceService... ({elapsed}s)", end="\r")

        except ApiException:
            pass

        time.sleep(10)

    print("\nTimeout waiting for InferenceService to be ready")
    print(f"Check status: kubectl get isvc {isvc_name} -n {cfg['namespace']}")
    return False


def deploy():
    """Deploy the PPE model as an InferenceService on OpenShift AI.

    Deployment sources (selected via environment variables):
      1. S3 bucket (default) -- set S3_BUCKET and S3_MODEL_PATH
      2. Model Registry      -- set DEPLOY_FROM_REGISTRY=true and MODEL_REGISTRY_URL
    """
    cfg = load_config()

    if not cfg["deploy_enabled"]:
        print("Model deployment not enabled. Skipping.")
        return

    if cfg["deploy_from_registry"] and cfg["model_registry_url"]:
        try:
            model_info = load_model_info_from_registry(cfg)
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch from Model Registry: {e}")
    else:
        model_info = load_model_info_from_s3(cfg)

    print("\nDeployment configuration:")
    print(f"  Runtime Type: {cfg['runtime_type']}")
    print(f"  Model: {model_info['model_name']}")
    print(f"  Version: {model_info['model_version']}")
    print(f"  Namespace: {cfg['namespace']}")
    print(f"  Bucket: {model_info['bucket']}")
    print(f"  Model Path: {model_info['model_path']}")
    print(f"  Serving Runtime: {cfg['serving_runtime']}")
    print(f"  Runtime Image: {cfg['runtime_image']}")
    print(f"  Model Format: {cfg['model_format']} v{cfg['model_format_version']}")

    core_v1, custom_api = init_kubernetes_client()

    create_storage_secret(core_v1, cfg, model_info)
    sa_name = create_service_account(core_v1, cfg)
    create_serving_runtime(custom_api, cfg)
    isvc_name = create_inference_service(custom_api, cfg, model_info, sa_name)
    wait_for_inference_service(custom_api, cfg, isvc_name)


if __name__ == "__main__":
    deploy()
