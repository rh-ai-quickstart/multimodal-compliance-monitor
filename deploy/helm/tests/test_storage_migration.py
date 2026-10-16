"""Helm regression checks; requires Helm, PyYAML, and chart dependencies.

From the repository root:
    python -m unittest discover -s deploy/helm/tests -v
"""

from pathlib import Path
import subprocess
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[3]
CHART = ROOT / "deploy/helm/ppe-compliance-monitor"


_TEST_CREDENTIALS = {
    "s3Credentials.create": "true",
    "s3Credentials.accessKeyId": "testkey",
    "s3Credentials.secretAccessKey": "testsecret",
    "s3Credentials.uiUsername": "admin",
    "s3Credentials.uiPassword": "changeme",
}


def render(**values):
    command = [
        "helm",
        "template",
        "migration",
        str(CHART),
        "--namespace",
        "storage-test",
    ]
    merged = {**_TEST_CREDENTIALS, **values}
    for key, value in merged.items():
        command.extend(["--set", f"{key}={value}"])
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


def resource(documents, kind, component):
    matches = [
        doc
        for doc in documents
        if doc["kind"] == kind
        and doc["metadata"].get("labels", {}).get("app.kubernetes.io/component")
        == component
    ]
    assert len(matches) == 1, (kind, component, len(matches))
    return matches[0]


class StorageNames(unittest.TestCase):
    def assert_storage_wiring(self, documents, expected_name, expected_secret):
        service = resource(documents, "Service", "storage")
        self.assertEqual(service["metadata"]["name"], expected_name)
        deployment = resource(documents, "Deployment", "storage")
        env_from = deployment["spec"]["template"]["spec"]["containers"][0]["envFrom"]
        self.assertIn({"secretRef": {"name": expected_secret}}, env_from)
        consumers = 0
        for doc in documents:
            if doc["kind"] not in ("Deployment", "Job"):
                continue
            spec = doc["spec"]["template"]["spec"]
            for container in spec.get("containers", []) + spec.get(
                "initContainers", []
            ):
                env = {entry["name"]: entry for entry in container.get("env", [])}
                if "AWS_ENDPOINT_URL" not in env:
                    continue
                consumers += 1
                self.assertEqual(
                    env["AWS_ENDPOINT_URL"]["value"], f"http://{expected_name}:7480"
                )
                for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
                    self.assertEqual(
                        env[key]["valueFrom"]["secretKeyRef"],
                        {"name": expected_secret, "key": key},
                    )
                if "AWS_COMPATIBLE_STORAGE_UI_URL" in env:
                    self.assertEqual(
                        env["AWS_COMPATIBLE_STORAGE_UI_URL"]["value"],
                        f"http://{expected_name}:5000/api",
                    )
        # Backend, runtime deployer, data uploader, bucket bootstrap, video init.
        self.assertEqual(consumers, 5)

    def test_default_storage_name(self):
        self.assert_storage_wiring(
            render(), "aws-compatible-storage", "aws-compatible-storage-credentials"
        )

    def test_application_fullname_does_not_change_storage_name(self):
        self.assert_storage_wiring(
            render(fullnameOverride="ppe-custom"),
            "aws-compatible-storage",
            "aws-compatible-storage-credentials",
        )

    def test_distinct_application_and_storage_names(self):
        documents = render(
            **{
                "fullnameOverride": "ppe-custom",
                "aws-compatible-storage.fullnameOverride": "s4",
                "aws-compatible-storage.s3.existingSecret": "s4-credentials",
            }
        )
        self.assert_storage_wiring(documents, "s4", "s4-credentials")

    def test_storage_without_fullname_override_uses_release_name(self):
        documents = render(
            **{
                "fullnameOverride": "ppe-custom",
                "aws-compatible-storage.fullnameOverride": "",
                "aws-compatible-storage.nameOverride": "objects",
            }
        )
        self.assert_storage_wiring(
            documents, "migration-objects", "aws-compatible-storage-credentials"
        )


class RuntimeJobNames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents = render()
        cls.job = resource(cls.documents, "Job", "runtime-deployer")

    def test_new_job_name_keeps_service_account_and_rbac_stable(self):
        account = resource(self.documents, "ServiceAccount", "runtime-deployer")[
            "metadata"
        ]["name"]
        self.assertEqual(account, "migration-ppe-compliance-monitor-runtime-deployer")
        self.assertRegex(self.job["metadata"]["name"], rf"^{account}-[0-9a-f]{{8}}$")
        self.assertEqual(
            self.job["spec"]["template"]["spec"]["serviceAccountName"], account
        )
        binding = resource(self.documents, "RoleBinding", "runtime-deployer")
        self.assertEqual(binding["subjects"][0]["name"], account)
        self.assertEqual(binding["roleRef"]["name"], account)
        self.assertNotIn("helm.sh/hook", self.job["metadata"].get("annotations", {}))

    def test_same_configuration_keeps_job_name(self):
        self.assertEqual(self.job, resource(render(), "Job", "runtime-deployer"))

    def test_configuration_changes_create_new_jobs(self):
        for key, value in {
            "runtimeDeployer.image.tag": "review-v2",
            "global.imageRegistry": "quay.io/review",
            "modelServing.runtimeType": "openvino",
            "modelServing.kserve.modelPath": "triton-v2",
            "modelServing.resources.requests.cpu": "3",
            "backend.modelVersion": "2",
            "storage.model.bucket": "new-models",
            "aws-compatible-storage.fullnameOverride": "s4",
            "aws-compatible-storage.s3.existingSecret": "new-credentials",
            "aws-compatible-storage.s3.region": "us-west-2",
            "s3Credentials.secretAccessKey": "rotated-demo-secret",
        }.items():
            with self.subTest(key=key):
                job = resource(render(**{key: value}), "Job", "runtime-deployer")
                self.assertNotEqual(
                    job["metadata"]["name"], self.job["metadata"]["name"]
                )

    def test_unrelated_frontend_change_keeps_job_name(self):
        job = resource(
            render(**{"frontend.image.tag": "new-ui"}), "Job", "runtime-deployer"
        )
        self.assertEqual(job["metadata"]["name"], self.job["metadata"]["name"])

    def test_long_names_preserve_hash_suffix(self):
        values = {"fullnameOverride": "p" * 46}
        first = resource(render(**values), "Job", "runtime-deployer")
        second = resource(
            render(**values, **{"backend.modelVersion": "2"}), "Job", "runtime-deployer"
        )
        for job in (first, second):
            self.assertLessEqual(len(job["metadata"]["name"]), 63)
            self.assertRegex(job["metadata"]["name"], r"-[0-9a-f]{8}$")
        self.assertNotEqual(first["metadata"]["name"], second["metadata"]["name"])


class SecretsIgnore(unittest.TestCase):
    def test_private_overlay_is_ignored(self):
        result = subprocess.run(
            [
                "git",
                "check-ignore",
                "deploy/helm/ppe-compliance-monitor/values-secrets.yaml",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
