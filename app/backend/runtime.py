import numpy as np
import cv2
import os
import time
from urllib.parse import urlparse
from ovmsclient import make_grpc_client

import tritonclient.grpc as triton_grpc

from logger import get_logger
from response import Detection, postprocess_image, _raw_prediction_tensor

log = get_logger(__name__)


class Runtime:
    def __init__(
        self,
        classes: dict[int, str] | None = None,
        service_url: str | None = None,
        model_name: str | None = None,
    ):
        self.service_url = service_url
        if not self.service_url or not str(self.service_url).strip():
            raise ValueError(
                "Runtime requires service_url (model_url from config). "
                "Add a config with an inferencing URL via the Config dialog."
            )
        _env = (os.getenv("MODEL_INPUT_NAME") or "").strip()
        self.input_name = _env or "x"
        self.model_name = (model_name or "").strip() or (
            os.getenv("MODEL_NAME") or ""
        ).strip()
        if not self.model_name:
            raise ValueError(
                "Runtime requires model_name (OVMS model id from app_config). "
                "Set it in the Configuration dialog."
            )
        _ver = (os.getenv("MODEL_VERSION") or "1").strip() or "1"
        try:
            self.model_version = int(_ver)
        except ValueError:
            self.model_version = 1
        self._model_version_str = str(self.model_version)

        _to = (os.getenv("OVMS_GRPC_TIMEOUT") or "").strip()
        try:
            self._ovms_predict_timeout = float(_to) if _to else 20.0
        except ValueError:
            self._ovms_predict_timeout = 20.0

        runtime_type = os.getenv("RUNTIME_TYPE", "openvino").lower()
        openshift_mode = os.getenv("OPENSHIFT", "false").lower() == "true"

        if runtime_type == "kserve":
            grpc_url = self.service_url.replace("https://", "").replace("http://", "")
            self._triton_client = triton_grpc.InferenceServerClient(
                url=grpc_url,
                channel_args=[
                    ("grpc.max_send_message_length", -1),
                    ("grpc.max_receive_message_length", -1),
                    ("grpc.optimization_target", "throughput"),
                ],
            )
            self._infer_output = triton_grpc.InferRequestedOutput("output0")
            self.inference_fun = self.kserve_inference_grpc
        elif openshift_mode:
            parsed = urlparse(self.service_url)
            if parsed.hostname:
                host = parsed.hostname
                port = parsed.port or 9000
                grpc_url = f"{host}:{port}"
            else:
                grpc_url = self.service_url
            self._grpc_client = make_grpc_client(grpc_url)
            self.inference_fun = self.remote_inference
        else:
            self._grpc_client = make_grpc_client(self.service_url)
            self.inference_fun = self.local_inference
        if not classes or len(classes) == 0:
            raise ValueError(
                "Runtime requires classes from detection_classes. "
                "Set ACTIVE_CONFIG_ID to a valid app_config id."
            )
        self.CLASSES = classes
        log.info(f"Runtime using {len(self.CLASSES)} classes from config")

        self._pad_shape: tuple[int, int] = (0, 0)
        self._padded: np.ndarray | None = None
        self._scale: float = 1.0
        self._batch_blob: np.ndarray | None = None

    def preprocess_image(self, image: np.ndarray):
        """
        Preprocess the image for the model.
        """
        height, width = image.shape[:2]

        if (height, width) != self._pad_shape:
            self._scale = max(height, width) / 640
            new_w = int(width / self._scale)
            new_h = int(height / self._scale)
            self._resized_shape = (new_w, new_h)
            self._padded = np.zeros((640, 640, 3), np.uint8)
            self._pad_shape = (height, width)

        resized = cv2.resize(image, self._resized_shape, interpolation=cv2.INTER_LINEAR)
        self._padded[: self._resized_shape[1], : self._resized_shape[0]] = resized

        blob = cv2.dnn.blobFromImage(self._padded, scalefactor=1 / 255, swapRB=True)
        return blob, self._scale

    def inference(self, image: np.ndarray) -> np.ndarray:
        """
        Inference the image for the model.
        """
        return self.inference_fun(image)

    def local_inference(self, image: np.ndarray) -> np.ndarray:
        """
        Local inference via persistent gRPC connection to OVMS.
        """
        inputs = {self.input_name: image}
        return self._grpc_client.predict(
            inputs,
            self.model_name,
            self.model_version,
            timeout=self._ovms_predict_timeout,
        )

    def remote_inference(self, image: np.ndarray) -> np.ndarray:
        """
        Remote inference via persistent gRPC connection to OVMS (binary protobuf).
        """
        inputs = {self.input_name: image}
        return self._grpc_client.predict(
            inputs,
            self.model_name,
            self.model_version,
            timeout=self._ovms_predict_timeout,
        )

    def kserve_inference_grpc(self, image: np.ndarray) -> np.ndarray:
        """
        Inference via KServe V2/Open Inference Protocol over gRPC using the
        Triton client.  Avoids HTTP serialization overhead entirely.
        """
        fp32_image = image if image.dtype == np.float32 else image.astype(np.float32)
        infer_input = triton_grpc.InferInput(
            self.input_name, list(fp32_image.shape), "FP32"
        )
        infer_input.set_data_from_numpy(fp32_image)
        result = self._triton_client.infer(
            model_name=self.model_name,
            model_version=self._model_version_str,
            inputs=[infer_input],
            outputs=[self._infer_output],
        )
        return result.as_numpy("output0")

    def run(self, image: np.ndarray) -> list[Detection]:
        """
        Run the inference for the image.
        """
        t0 = time.perf_counter()
        blob, scale = self.preprocess_image(image)
        t1 = time.perf_counter()
        outputs = self.inference(blob)
        t2 = time.perf_counter()
        detections = postprocess_image(outputs, scale, self.CLASSES)
        t3 = time.perf_counter()

        log.debug(
            f"Inference timing — preprocess: {(t1 - t0) * 1000:.1f}ms, "
            f"inference: {(t2 - t1) * 1000:.1f}ms, "
            f"postprocess: {(t3 - t2) * 1000:.1f}ms, "
            f"total: {(t3 - t0) * 1000:.1f}ms"
        )
        return detections

    def _preprocess_batch_into(
        self, images: list[np.ndarray], out: np.ndarray
    ) -> list[float]:
        """Preprocess images directly into a pre-allocated (N, 3, 640, 640) buffer.

        Fast path when all frames share the same resolution (typical for video):
        geometry is computed once, and the shared padded buffer is reused without
        re-zeroing unchanged regions.
        """
        n = len(images)
        scales: list[float] = [0.0] * n

        first_h, first_w = images[0].shape[:2]
        uniform = all(
            img.shape[0] == first_h and img.shape[1] == first_w for img in images
        )

        if uniform:
            if (first_h, first_w) != self._pad_shape:
                self._scale = max(first_h, first_w) / 640
                new_w = int(first_w / self._scale)
                new_h = int(first_h / self._scale)
                self._resized_shape = (new_w, new_h)
                self._padded = np.zeros((640, 640, 3), np.uint8)
                self._pad_shape = (first_h, first_w)

            rw, rh = self._resized_shape
            scale = self._scale
            padded = self._padded

            for i, img in enumerate(images):
                resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
                padded[:rh, :rw] = resized
                out[i, 0] = padded[:, :, 2]
                out[i, 1] = padded[:, :, 1]
                out[i, 2] = padded[:, :, 0]
                scales[i] = scale

            out *= 1.0 / 255.0
        else:
            for i, img in enumerate(images):
                blob, scale = self.preprocess_image(img)
                out[i] = blob[0]
                scales[i] = scale

        return scales

    def run_batch(self, images: list[np.ndarray]) -> list[list[Detection]]:
        """Run inference on a batch of images with a single gRPC call.

        Pre/post-processing are per-image loops; only the inference call is
        batched into one ``(N, 3, 640, 640)`` tensor.
        """
        if not images:
            return []
        if len(images) == 1:
            return [self.run(images[0])]

        t0 = time.perf_counter()
        n = len(images)

        if self._batch_blob is None or self._batch_blob.shape[0] < n:
            self._batch_blob = np.empty((n, 3, 640, 640), dtype=np.float32)
        batched_blob = self._batch_blob[:n]

        scales = self._preprocess_batch_into(images, batched_blob)
        t1 = time.perf_counter()

        raw_outputs = self.inference(batched_blob)
        t2 = time.perf_counter()

        raw_tensor = _raw_prediction_tensor(raw_outputs)

        results: list[list[Detection]] = []
        for i in range(n):
            per_image = raw_tensor[i : i + 1]
            dets = postprocess_image(per_image, scales[i], self.CLASSES)
            results.append(dets)
        t3 = time.perf_counter()

        log.debug(
            f"Batch inference ({n} images) — "
            f"preprocess: {(t1 - t0) * 1000:.1f}ms, "
            f"inference: {(t2 - t1) * 1000:.1f}ms, "
            f"postprocess: {(t3 - t2) * 1000:.1f}ms, "
            f"total: {(t3 - t0) * 1000:.1f}ms"
        )
        return results
