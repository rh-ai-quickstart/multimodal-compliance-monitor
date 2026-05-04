"""Standalone evaluation script for the /api/chat endpoint.

Auto-discovers all ``*.json`` experiment files inside
``datasets/<EVAL_DATASET>/``, calls the running backend's chat REST API for
each entry, and uses DeepEval's GEval (Correctness) metric with VLLMJudge to
score the responses.

Before running, the live database is snapshotted to a volume-backed file.
Eval seed data is loaded, all experiments run, and the live data is restored
in a ``finally`` block (even on error).

Usage::

    EVAL_DATASET=ppe python run_eval.py
    EVAL_DATASET=bird python run_eval.py

Requires env vars: OPENAI_API_ENDPOINT, OPENAI_API_TOKEN, OPENAI_MODEL.
Optionally set BACKEND_URL (default http://localhost:8888).
"""

import asyncio
import json
import os
import sys
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, DisplayConfig
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics.g_eval import Rubric

from judge_model import VLLMJudge
from load_seed import save_snapshot, load_seed, restore_snapshot

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8888").rstrip("/")
CHAT_ENDPOINT = f"{BACKEND_URL}/api/chat"
THRESHOLD = 0.5

EVAL_DATASET = os.getenv("EVAL_DATASET", "ppe")
DATASETS_DIR = Path(__file__).parent / "datasets" / EVAL_DATASET
SEED_SQL_PATH = Path(__file__).parent / "db_seed_data.sql"
PREDS_DIR = Path(__file__).parent / "preds" / EVAL_DATASET

DATASET_APP_CONFIG_ID: dict[str, int] = {
    "bird": 1,
    "ppe": 2,
    "yolo": 3,
}
APP_CONFIG_ID = DATASET_APP_CONFIG_ID.get(EVAL_DATASET)


def discover_experiments() -> list[tuple[str, Path]]:
    """Return sorted ``(experiment_name, path)`` pairs for every JSON file in
    the dataset directory."""
    if not DATASETS_DIR.is_dir():
        available = ", ".join(
            p.name for p in (Path(__file__).parent / "datasets").iterdir() if p.is_dir()
        )
        print(f"ERROR: Dataset directory not found: {DATASETS_DIR}", file=sys.stderr)
        print(f"Available datasets: {available}", file=sys.stderr)
        sys.exit(1)

    experiments = sorted((p.stem, p) for p in DATASETS_DIR.glob("*.json"))
    if not experiments:
        print(
            f"ERROR: No JSON experiment files in {DATASETS_DIR}",
            file=sys.stderr,
        )
        sys.exit(1)
    return experiments


def load_dataset(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def call_chat(question: str, description: str, session_id: str) -> str:
    body: dict = {
        "question": question,
        "description": description,
        "session_id": session_id,
    }
    if APP_CONFIG_ID is not None:
        body["app_config_id"] = APP_CONFIG_ID
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        CHAT_ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["answer"]


async def _fetch_one(
    entry: dict, eval_run_id: str
) -> tuple[dict, str | None, str | None]:
    """Return (entry, actual_output, error_str)."""
    try:
        output = await asyncio.to_thread(
            call_chat,
            entry["question"],
            entry["description"],
            session_id=f"{eval_run_id}_{entry['id']}",
        )
        return entry, output, None
    except Exception as exc:
        return entry, None, str(exc)


async def _fetch_all(
    dataset: list[dict], eval_run_id: str
) -> list[tuple[dict, str | None, str | None]]:
    tasks = [_fetch_one(entry, eval_run_id) for entry in dataset]
    return await asyncio.gather(*tasks)


def run_experiment(
    experiment_name: str,
    dataset_path: Path,
    judge: VLLMJudge,
    eval_run_id: str,
) -> list[dict]:
    """Run a single experiment file and return its result dicts."""
    dataset = load_dataset(dataset_path)

    correctness = GEval(
        name="Correctness",
        # criteria="Determine whether the actual output correctly answers the input question with the same numerical facts as the expected output.",
        evaluation_steps=[
            "Check that the actual output directly answers the core question in the input.",
            "Verify all numerical values and yes/no conclusions match between the actual output and the expected output.",
            "Penalize contradicted or omitted key facts; extra detail or phrasing differences are acceptable.",
        ],
        rubric=[
            Rubric(
                score_range=(0, 2),
                expected_outcome="Numerical values not matching between the actual output and the expected output. Or yes/no conclusions not matching between the actual output and the expected output.",
            ),
            Rubric(
                score_range=(5, 7),
                expected_outcome="numberical values matching between the actual output and the expected output. there is some additional information.",
            ),
            Rubric(
                score_range=(8, 9),
                expected_outcome="Correct but missing minor details.",
            ),
            Rubric(score_range=(10, 10), expected_outcome="100% correct."),
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=judge,
        threshold=THRESHOLD,
    )

    print(f"Sending {len(dataset)} chat requests concurrently ...")
    fetch_results = asyncio.run(_fetch_all(dataset, eval_run_id))

    results: list[dict] = []
    test_cases: list[tuple[dict, LLMTestCase, str]] = []

    for entry, actual_output, error in fetch_results:
        if error:
            print(f"[{entry['id']}] ERROR: {error}")
            results.append(
                {
                    **entry,
                    "predicted": None,
                    "judge_score": 0.0,
                    "judge_reason": None,
                    "passed": False,
                    "error": error,
                }
            )
            continue
        tc = LLMTestCase(
            input=entry["question"],
            actual_output=actual_output,
            expected_output=entry["golden_answer"],
            additional_metadata={"id": entry["id"]},
        )
        test_cases.append((entry, tc, actual_output))

    if test_cases:
        tc_list = [tc for _, tc, _ in test_cases]
        eval_result = evaluate(
            test_cases=tc_list,
            metrics=[correctness],
            async_config=AsyncConfig(max_concurrent=10, throttle_value=0),
            display_config=DisplayConfig(print_results=False),
        )
        result_by_id = {
            tr.additional_metadata["id"]: tr for tr in eval_result.test_results
        }
        for entry, _tc, actual_output in test_cases:
            test_result = result_by_id[entry["id"]]
            metric = test_result.metrics_data[0]
            score = metric.score if metric.score is not None else 0.0
            passed = score >= THRESHOLD
            reason = metric.reason
            print(f"[{entry['id']}] score={score:.2f}  {'PASS' if passed else 'FAIL'}")
            results.append(
                {
                    **entry,
                    "predicted": actual_output,
                    "judge_score": score,
                    "judge_reason": reason,
                    "passed": passed,
                    "error": metric.error,
                }
            )

    return results


def run() -> None:
    eval_run_id = f"eval-{uuid.uuid4().hex[:8]}"
    print(f"==> Eval run ID:  {eval_run_id}")

    experiments = discover_experiments()
    judge = VLLMJudge()
    all_results: list[dict] = []
    all_passed = True
    now = datetime.now()

    for experiment_name, dataset_path in experiments:
        print()
        print(f"{'=' * 72}")
        print(f"  Experiment: {experiment_name}  ({dataset_path.name})")
        print(f"{'=' * 72}")

        results = run_experiment(experiment_name, dataset_path, judge, eval_run_id)
        print_summary(experiment_name, results)
        save_results(experiment_name, results, now)
        all_results.extend(results)
        if not all(r["passed"] for r in results):
            all_passed = False

    save_summary(all_results, now)

    if len(experiments) > 1:
        print()
        print(f"{'#' * 72}")
        print("  OVERALL SUMMARY (all experiments)")
        print(f"{'#' * 72}")
        print_summary("all", all_results)

    sys.exit(0 if all_passed else 1)


def print_summary(experiment_name: str, results: list[dict]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    print("\n" + "=" * 72)
    print(f"  [{experiment_name}]  {total} tests | {passed} passed | {failed} failed")
    print("-" * 72)
    print(f"{'ID':<50} {'SCORE':>6}  {'RESULT':>6}")
    print("-" * 72)
    for r in results:
        tag = "PASS" if r["passed"] else "FAIL"
        print(f"{r['id']:<50} {r['judge_score']:>6.2f}  {tag:>6}")
    print("=" * 72)
    print(
        f"Total: {total}  Passed: {passed}  Failed: {failed}  "
        f"Pass rate: {passed / total * 100:.1f}%"
    )


def save_results(experiment_name: str, results: list[dict], now: datetime) -> None:
    """Write eval results to a timestamped JSON file in preds/<dataset>/."""
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    output = {
        "eval_dataset": EVAL_DATASET,
        "experiment": experiment_name,
        "timestamp": now.isoformat(timespec="seconds"),
        "threshold": THRESHOLD,
        "model": os.getenv("OPENAI_MODEL", "unknown"),
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total * 100, 1) if total else 0.0,
        "results": results,
    }

    run_dir = PREDS_DIR / now.strftime("%Y-%m-%dT%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    for parent in [run_dir] + list(run_dir.parents):
        try:
            parent.chmod(0o777)
        except OSError:
            break
    path = run_dir / f"{experiment_name}.json"

    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    path.chmod(0o666)

    print(f"\nResults saved to: {path}")


def save_summary(all_results: list[dict], now: datetime) -> None:
    """Write a compact summary JSON with per-ID score and pass/fail."""
    total = len(all_results)
    passed = sum(1 for r in all_results if r["passed"])
    failed = total - passed

    summary = {
        "timestamp": now.isoformat(timespec="seconds"),
        "model": os.getenv("OPENAI_MODEL", "unknown"),
        "threshold": THRESHOLD,
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total * 100, 1) if total else 0.0,
        "results": [
            {
                "id": r["id"],
                "score": r["judge_score"],
                "result": "pass" if r["passed"] else "fail",
            }
            for r in all_results
        ],
    }

    run_dir = PREDS_DIR / now.strftime("%Y-%m-%dT%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "summary.json"

    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    path.chmod(0o666)

    print(f"\nSummary saved to: {path}")


if __name__ == "__main__":
    experiments = discover_experiments()
    print(f"==> Eval dataset: {EVAL_DATASET}")
    print(f"==> Seed SQL:     {SEED_SQL_PATH}")
    print(f"==> Experiments:  {len(experiments)} file(s)")
    for name, path in experiments:
        print(f"    - {name} ({path.name})")
    print()

    print("Saving live database snapshot to volume ... ", end="", flush=True)
    stmt_count = save_snapshot()
    print(f"done ({stmt_count} statements)")

    try:
        print("Loading eval seed data ... ", end="", flush=True)
        counts = load_seed(SEED_SQL_PATH)
        summary = ", ".join(f"{t}: {n}" for t, n in counts.items())
        print(f"done ({summary})")
        run()
    finally:
        print("\nRestoring live database from snapshot ... ", end="", flush=True)
        try:
            restored = restore_snapshot()
            summary = ", ".join(f"{t}: {n}" for t, n in restored.items())
            print(f"done ({summary})")
        except Exception as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            print(
                "WARNING: Live data was NOT restored. "
                "Manual restore may be needed from the snapshot file.",
                file=sys.stderr,
            )
