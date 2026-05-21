.DEFAULT_GOAL := help
.PHONY: help local-up local-build-up local-down build push build-all push-all build-data push-data build-eval push-eval build-runtime push-runtime helm-deps deploy deploy-cpu deploy-gpu deploy-openvino-labelstudio deploy-labelstudio undeploy dev-backend dev-frontend local-build build-push-data build-jupyter-training push-jupyter-training kill-ports check-openai-env eval eval-k8s init-eval-db
help:
	@echo "Available targets:"
	@echo "  local-up   - Start local stack with Podman Compose"
	@echo "  local-build-up - Build and start local stack (see podman-compose backend env)"
	@echo "  local-down - Stop local stack"
	@echo "  build      - Build backend and frontend images"
	@echo "  push       - Push backend and frontend images"
	@echo "  build-all  - Build all container images (backend, frontend, data, eval, runtime, jupyter-training)"
	@echo "  push-all   - Push all container images to IMAGE_REGISTRY"
	@echo "  build-data - Build data container image (video + models)"
	@echo "  build-eval - Build eval container image"
	@echo "  build-runtime - Build runtime deployer container image"
	@echo "  build-push-data - Build and push data container image (video + models)"
	@echo "  deploy     - Deploy to OpenShift with OpenVINO CPU runtime (default)"
	@echo "  deploy-cpu - Deploy with CPU runtime (OpenVINO Model Server)"
	@echo "  deploy-gpu - Deploy with GPU runtime (kserve/Triton)"
	@echo "  build-jupyter-training - Build Jupyter training image (notebook under ~/work/training)"
	@echo "  push-jupyter-training - Push Jupyter training image (after build-jupyter-training)"
	@echo "  deploy-openvino-labelstudio - Deploy OpenVINO runtime with Label Studio enabled"
	@echo "  deploy-labelstudio - Deploy and enable Label Studio"
	@echo "  helm-deps  - Update Helm chart dependencies"
	@echo "  undeploy   - Remove manifests from OpenShift"
	@echo "  dev-backend - Create venv, install deps, run backend"
	@echo "  dev-frontend - Install deps and run frontend"
	@echo "  eval         - Run evaluation against the running backend (local)"
	@echo "                 Use EVAL_FEATURE=alerts to eval alerts (default: chat)"
	@echo "                 Use EVAL_DATASET=bird to select a dataset (default: ppe)"
	@echo "  eval-k8s     - Run LLM chat evaluation against the deployed K8s backend (helm test)"
	@echo "  init-eval-db - Snapshot the running DB into app/evals/db_seed_data.sql"


# Load .env file if it exists
ifneq (,$(wildcard .env))
  include .env
  export OPENAI_API_TOKEN OPENAI_API_ENDPOINT OPENAI_MODEL OPENAI_TEMPERATURE
endif

COMPOSE_FILE ?= $(CURDIR)/deploy/local/podman-compose.yaml
NAMESPACE ?= ppe-compliance-monitor-demo
PLATFORM_RELEASE ?= linux/amd64
PLATFORM_LOCAL ?= $(shell uname -m | sed -e 's/x86_64/linux\/amd64/' -e 's/arm64/linux\/arm64/' -e 's/aarch64/linux\/arm64/')

IMAGE_NAME ?= ppe-compliance-monitor
IMAGE_TAG ?= latest
IMAGE_REGISTRY ?= quay.io/rh-ai-quickstart
IMAGE_REPOSITORY := $(if $(IMAGE_REGISTRY),$(IMAGE_REGISTRY)/,)$(IMAGE_NAME)
BACKEND_IMAGE := $(IMAGE_REPOSITORY)-backend:$(IMAGE_TAG)
FRONTEND_IMAGE := $(IMAGE_REPOSITORY)-frontend:$(IMAGE_TAG)
DATA_IMAGE := $(IMAGE_REPOSITORY)-data:$(IMAGE_TAG)
EVAL_IMAGE := $(IMAGE_REPOSITORY)-eval:$(IMAGE_TAG)
RUNTIME_IMAGE := $(IMAGE_REPOSITORY)-runtime:$(IMAGE_TAG)
JUPYTER_TRAINING_IMAGE := $(IMAGE_REPOSITORY)-jupyter-training:$(IMAGE_TAG)
LOCAL_BACKEND_IMAGE ?= ppe-compliance-monitor-backend:local
LOCAL_FRONTEND_IMAGE ?= ppe-compliance-monitor-frontend:local
LOCAL_DATA_IMAGE ?= ppe-compliance-monitor-data:local
PYTHON ?= python3
VENV_DIR ?= .venv
BACKEND_DIR ?= app/backend
FRONTEND_DIR ?= app/frontend
HELM_RELEASE ?= ppe-compliance-monitor
HELM_CHART ?= deploy/helm/ppe-compliance-monitor
# Model serving runtime: "openvino" (CPU/OVMS) or "kserve" (GPU/Triton)
RUNTIME_TYPE ?= openvino
LABEL_STUDIO_ENABLED ?=
EVAL_FEATURE ?= chat
EVAL_DATASET ?= ppe

helm-deps:
	@echo "Updating Helm chart dependencies..."
	cd $(HELM_CHART) && helm dependency update

check-openai-env:
	@token="$(OPENAI_API_TOKEN)"; \
	endpoint="$(OPENAI_API_ENDPOINT)"; \
	model="$(OPENAI_MODEL)"; \
	temp="$(OPENAI_TEMPERATURE)"; \
	if [ -z "$$token" ] || [ -z "$$endpoint" ] || [ -z "$$model" ]; then \
		echo "==> Missing required OpenAI environment variables."; \
		if [ -z "$$token" ]; then printf "  OPENAI_API_TOKEN: "; read token; fi; \
		if [ -z "$$endpoint" ]; then printf "  OPENAI_API_ENDPOINT: "; read endpoint; fi; \
		if [ -z "$$model" ]; then printf "  OPENAI_MODEL: "; read model; fi; \
		if [ -z "$$temp" ]; then temp="0.7"; fi; \
		printf 'OPENAI_API_TOKEN=%s\nOPENAI_API_ENDPOINT=%s\nOPENAI_MODEL=%s\nOPENAI_TEMPERATURE=%s\n' \
			"$$token" "$$endpoint" "$$model" "$$temp" > .env; \
		echo "==> Saved to .env"; \
	else \
		echo "==> OpenAI environment variables loaded from .env"; \
	fi

local-build-up: kill-ports local-down
	PODMAN_DEFAULT_PLATFORM=$(PLATFORM_LOCAL) podman-compose -f $(COMPOSE_FILE) up --build

local-build:
	@should_build=0; \
	if ! podman image exists $(LOCAL_BACKEND_IMAGE); then should_build=1; fi; \
	if ! podman image exists $(LOCAL_FRONTEND_IMAGE); then should_build=1; fi; \
	if [ $$should_build -eq 1 ]; then \
		echo "Building local image..."; \
		PODMAN_DEFAULT_PLATFORM=$(PLATFORM_LOCAL) podman-compose -f $(COMPOSE_FILE) build; \
	else \
		echo "Local image is up-to-date; skipping build."; \
	fi

local-down:
	podman-compose -f $(COMPOSE_FILE) down

build:
	podman build --platform $(PLATFORM_RELEASE) -t $(BACKEND_IMAGE) -f app/backend/Dockerfile app/backend
	podman build --platform $(PLATFORM_RELEASE) -t $(FRONTEND_IMAGE) -f app/frontend/Dockerfile app/frontend

push:
	@if podman image exists $(BACKEND_IMAGE) && podman image exists $(FRONTEND_IMAGE); then \
		podman push $(BACKEND_IMAGE); \
		podman push $(FRONTEND_IMAGE); \
	else \
		echo "Images not found. Run 'make build' first."; \
		exit 1; \
	fi

build-push-data:
	podman build --platform $(PLATFORM_RELEASE) -t $(DATA_IMAGE) -f app/data-image/Dockerfile app
	podman push $(DATA_IMAGE)

build-jupyter-training:
	podman build --platform $(PLATFORM_RELEASE) -t $(JUPYTER_TRAINING_IMAGE) -f training/jupyter-training/Dockerfile training

push-jupyter-training:
	@if podman image exists $(JUPYTER_TRAINING_IMAGE); then \
		podman push $(JUPYTER_TRAINING_IMAGE); \
	else \
		echo "Image $(JUPYTER_TRAINING_IMAGE) not found. Run 'make build-jupyter-training' first."; \
		exit 1; \
	fi

build-data:
	podman build --platform $(PLATFORM_RELEASE) -t $(DATA_IMAGE) -f app/data-image/Dockerfile app

push-data:
	@if podman image exists $(DATA_IMAGE); then \
		podman push $(DATA_IMAGE); \
	else \
		echo "Image $(DATA_IMAGE) not found. Run 'make build-data' first."; \
		exit 1; \
	fi

build-eval:
	podman build --platform $(PLATFORM_RELEASE) -t $(EVAL_IMAGE) -f app/evals/Containerfile app/evals

push-eval:
	@if podman image exists $(EVAL_IMAGE); then \
		podman push $(EVAL_IMAGE); \
	else \
		echo "Image $(EVAL_IMAGE) not found. Run 'make build-eval' first."; \
		exit 1; \
	fi

build-runtime:
	podman build --platform $(PLATFORM_RELEASE) -t $(RUNTIME_IMAGE) -f app/runtime/Containerfile app/runtime

push-runtime:
	@if podman image exists $(RUNTIME_IMAGE); then \
		podman push $(RUNTIME_IMAGE); \
	else \
		echo "Image $(RUNTIME_IMAGE) not found. Run 'make build-runtime' first."; \
		exit 1; \
	fi

build-all: build build-data build-eval build-runtime build-jupyter-training
	@echo "All images built successfully"

push-all:
	@echo "Pushing all images to $(IMAGE_REGISTRY)..."
	@if podman image exists $(BACKEND_IMAGE); then podman push $(BACKEND_IMAGE); else echo "Warning: $(BACKEND_IMAGE) not found"; fi
	@if podman image exists $(FRONTEND_IMAGE); then podman push $(FRONTEND_IMAGE); else echo "Warning: $(FRONTEND_IMAGE) not found"; fi
	@if podman image exists $(DATA_IMAGE); then podman push $(DATA_IMAGE); else echo "Warning: $(DATA_IMAGE) not found"; fi
	@if podman image exists $(EVAL_IMAGE); then podman push $(EVAL_IMAGE); else echo "Warning: $(EVAL_IMAGE) not found"; fi
	@if podman image exists $(RUNTIME_IMAGE); then podman push $(RUNTIME_IMAGE); else echo "Warning: $(RUNTIME_IMAGE) not found"; fi
	@if podman image exists $(JUPYTER_TRAINING_IMAGE); then podman push $(JUPYTER_TRAINING_IMAGE); else echo "Warning: $(JUPYTER_TRAINING_IMAGE) not found"; fi
	@echo "All available images pushed to $(IMAGE_REGISTRY)"

deploy: check-openai-env
	@. ./.env; \
	domain=$$(oc get ingresses.config/cluster -o jsonpath='{.spec.domain}' 2>/dev/null || true); \
	if [ -n "$(NAMESPACE)" ]; then oc new-project "$(NAMESPACE)" --display-name="$(NAMESPACE)" >/dev/null 2>&1 || oc project "$(NAMESPACE)"; fi; \
	if [ -n "$$domain" ]; then \
		host="$(HELM_RELEASE)-$(NAMESPACE).$$domain"; \
		ls_host="$(HELM_RELEASE)-ls-$(NAMESPACE).$$domain"; \
	else \
		host=""; \
		ls_host=""; \
	fi; \
	helm_args="--set global.imageRegistry=$(IMAGE_REGISTRY) \
		--set jupyter-training.imageRegistry=$(IMAGE_REGISTRY) \
		--set backend.image.tag=$(IMAGE_TAG) \
		--set frontend.image.tag=$(IMAGE_TAG) \
		--set data.image.tag=$(IMAGE_TAG) \
		--set eval.image.tag=$(IMAGE_TAG) \
		--set runtimeDeployer.image.tag=$(IMAGE_TAG) \
		--set jupyter-training.image.tag=$(IMAGE_TAG) \
		--set modelServing.runtimeType=$(RUNTIME_TYPE) \
		$(if $(strip $(LABEL_STUDIO_ENABLED)),--set labelStudio.enabled=$(LABEL_STUDIO_ENABLED),) \
		$${host:+--set openshift.sharedHost=$$host} \
		$${ls_host:+--set labelStudio.route.host=$$ls_host} \
		--set openai.apiToken=$$OPENAI_API_TOKEN \
		--set openai.apiEndpoint=$$OPENAI_API_ENDPOINT \
		--set openai.model=$$OPENAI_MODEL \
		--set openai.temperature=$$OPENAI_TEMPERATURE"; \
	helm upgrade --install $(HELM_RELEASE) $(HELM_CHART) \
		--namespace $(NAMESPACE) --create-namespace $$helm_args

deploy-gpu: ## Deploy with GPU runtime (kserve/Triton)
	$(MAKE) deploy RUNTIME_TYPE=kserve

deploy-cpu: ## Deploy with CPU runtime (OpenVINO Model Server)
	$(MAKE) deploy RUNTIME_TYPE=openvino

deploy-openvino-labelstudio: ## Deploy OpenVINO runtime with Label Studio enabled
	$(MAKE) deploy RUNTIME_TYPE=openvino LABEL_STUDIO_ENABLED=true

deploy-labelstudio: ## Deploy with Label Studio enabled
	$(MAKE) deploy LABEL_STUDIO_ENABLED=true

undeploy:
	@if [ -n "$(NAMESPACE)" ]; then oc project "$(NAMESPACE)"; fi
	@helm uninstall $(HELM_RELEASE) --namespace $(NAMESPACE) 2>/dev/null || echo "Release $(HELM_RELEASE) not found (already uninstalled or never deployed)."

eval: check-openai-env ## Run evaluation against the running backend (local).
	@mkdir -p $(CURDIR)/app/evals/preds/$(EVAL_FEATURE)
	EVAL_FEATURE=$(EVAL_FEATURE) EVAL_DATASET=$(EVAL_DATASET) podman-compose -f $(COMPOSE_FILE) --profile eval run --rm --no-deps --build \
	  -v $(CURDIR)/app/evals/preds:/evals/preds:z,U \
	  backend-eval

eval-k8s: ## Run LLM chat evaluation against the deployed K8s backend (helm test).
	helm test $(HELM_RELEASE) --namespace $(NAMESPACE) --logs

init-eval-db: ## Snapshot the running DB into app/evals/db_seed_data.sql for eval use.
	podman-compose -f $(COMPOSE_FILE) --profile eval run --rm --no-deps init-eval-db

kill-ports: ## Kill processes using required ports
	@echo "Killing processes on application ports..."
	@if [ "$$(uname)" = "Darwin" ]; then \
		lsof -ti :3000 | xargs kill -9 2>/dev/null || true; \
	else \
		fuser -k 3000/tcp 2>/dev/null || true; \
	fi
	@echo "   ✓ Frontend 3000 killed"
	@if [ "$$(uname)" = "Darwin" ]; then \
		lsof -ti :8888 | xargs kill -9 2>/dev/null || true; \
	else \
		fuser -k 8888/tcp 2>/dev/null || true; \
	fi
	@echo "   ✓ Backend 8888 killed"
	@if [ "$$(uname)" = "Darwin" ]; then \
		lsof -ti :8080 | xargs kill -9 2>/dev/null || true; \
	else \
		fuser -k 8080/tcp 2>/dev/null || true; \
	fi
	@echo "   ✓ OVMS REST 8080 killed"
	@if [ "$$(uname)" = "Darwin" ]; then \
		lsof -ti :8081 | xargs kill -9 2>/dev/null || true; \
	else \
		fuser -k 8081/tcp 2>/dev/null || true; \
	fi
	@echo "   ✓ OVMS gRPC 8081 killed"
	@if [ "$$(uname)" = "Darwin" ]; then \
		lsof -ti :9000 | xargs kill -9 2>/dev/null || true; \
	else \
		fuser -k 9000/tcp 2>/dev/null || true; \
	fi
	@echo "   ✓ MinIO 9000 killed"
	@if [ "$$(uname)" = "Darwin" ]; then \
		lsof -ti :9001 | xargs kill -9 2>/dev/null || true; \
	else \
		fuser -k 9001/tcp 2>/dev/null || true; \
	fi
	@echo "   ✓ MinIO Console 9001 killed"
	@if [ "$$(uname)" = "Darwin" ]; then \
		lsof -ti :5432 | xargs kill -9 2>/dev/null || true; \
	else \
		fuser -k 5432/tcp 2>/dev/null || true; \
	fi
	@echo "   ✓ PostgreSQL 5432 killed"
	@if [ "$$(uname)" = "Darwin" ]; then \
		lsof -ti :8000 | xargs kill -9 2>/dev/null || true; \
	else \
		fuser -k 8000/tcp 2>/dev/null || true; \
	fi
	@echo "   ✓ Postgres MCP 8000 killed"
	@if [ "$$(uname)" = "Darwin" ]; then \
		lsof -ti :8554 | xargs kill -9 2>/dev/null || true; \
	else \
		fuser -k 8554/tcp 2>/dev/null || true; \
	fi
	@echo "   ✓ MediaMTX 8554 killed"
	@if [ "$$(uname)" = "Darwin" ]; then \
		lsof -ti :8082 | xargs kill -9 2>/dev/null || true; \
	else \
		fuser -k 8082/tcp 2>/dev/null || true; \
	fi
	@echo "   ✓ Label Studio 8082 killed"
	@if [ "$$(uname)" = "Darwin" ]; then \
		lsof -ti :6006 | xargs kill -9 2>/dev/null || true; \
	else \
		fuser -k 6006/tcp 2>/dev/null || true; \
	fi
	@echo "   ✓ Phoenix 6006 killed"
	@echo "All ports cleared."
