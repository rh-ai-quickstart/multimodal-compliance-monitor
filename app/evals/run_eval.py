"""Standalone evaluation script for the /api/chat endpoint.

Loads eval_dataset.json for the selected dataset, calls the running backend's
chat REST API for each entry, and uses DeepEval's GEval (Correctness) metric
with VLLMJudge to score the responses.

Before running, the live database is snapshotted to a volume-backed file.
Eval seed data is loaded, eval runs, and the live data is restored in a
``finally`` block (even on error).

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
import urllib.error
from datetime import datetime
from pathlib import Path

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from judge_model import VLLMJudge
from load_seed import save_snapshot, load_seed, restore_snapshot

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8888").rstrip("/")
CHAT_ENDPOINT = f"{BACKEND_URL}/api/chat"
THRESHOLD = 0.5

EVAL_DATASET = os.getenv("EVAL_DATASET", "ppe")
DATASET_PATH = Path(__file__).parent / "datasets" / EVAL_DATASET / "eval_dataset.json"
SEED_SQL_PATH = Path(__file__).parent / "db_seed_data.sql"
PREDS_DIR = Path(__file__).parent / "preds" / EVAL_DATASET


def load_dataset() -> list[dict]:
    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found: {DATASET_PATH}", file=sys.stderr)
        print(
            f"Available datasets: {', '.join(p.name for p in (Path(__file__).parent / 'datasets').iterdir() if p.is_dir())}",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(DATASET_PATH) as f:
        return json.load(f)


def call_chat(question: str, description: str, session_id: str) -> str:
    payload = json.dumps(
        {
            "question": question,
            "description": description,
            "session_id": session_id,
        }
    ).encode()
    req = urllib.request.Request(
        CHAT_ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["answer"]


async def _fetch_one(entry: dict) -> tuple[dict, str | None, str | None]:
    """Return (entry, actual_output, error_str)."""
    try:
        output = await asyncio.to_thread(
            call_chat,
            entry["question"],
            entry["description"],
            session_id=entry["id"],
        )
        return entry, output, None
    except Exception as exc:
        return entry, None, str(exc)


async def _fetch_all(dataset: list[dict]) -> list[tuple[dict, str | None, str | None]]:
    tasks = [_fetch_one(entry) for entry in dataset]
    return await asyncio.gather(*tasks)


def run() -> None:
    dataset = load_dataset()
    judge = VLLMJudge()

    correctness = GEval(
        name="Correctness",
        criteria=(
            "The actual output must state the correct number and factual "
            "details matching the expected output. Minor wording differences "
            "are acceptable."
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=judge,
        threshold=THRESHOLD,
    )

    print(f"Sending {len(dataset)} chat requests concurrently ...")
    fetch_results = asyncio.run(_fetch_all(dataset))

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
        )
        test_cases.append((entry, tc, actual_output))

    for entry, tc, actual_output in test_cases:
        correctness.measure(tc)
        passed = correctness.score >= THRESHOLD
        reason = getattr(correctness, "reason", None)
        print(
            f"[{entry['id']}] score={correctness.score:.2f}  {'PASS' if passed else 'FAIL'}"
        )
        results.append(
            {
                **entry,
                "predicted": actual_output,
                "judge_score": correctness.score,
                "judge_reason": reason,
                "passed": passed,
                "error": None,
            }
        )

    print_summary(results)
    save_results(results)
    sys.exit(0 if all(r["passed"] for r in results) else 1)


def print_summary(results: list[dict]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    print("\n" + "=" * 72)
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


def save_results(results: list[dict]) -> None:
    """Write eval results to a timestamped JSON file in preds/<dataset>/."""
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    now = datetime.now()
    output = {
        "eval_dataset": EVAL_DATASET,
        "timestamp": now.isoformat(timespec="seconds"),
        "threshold": THRESHOLD,
        "model": os.getenv("OPENAI_MODEL", "unknown"),
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total * 100, 1) if total else 0.0,
        "results": results,
    }

    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"results_{now.strftime('%Y-%m-%dT%H-%M-%S')}.json"
    path = PREDS_DIR / filename

    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {path}")


if __name__ == "__main__":
    print(f"==> Eval dataset: {EVAL_DATASET}")
    print(f"==> Seed SQL:     {SEED_SQL_PATH}")
    print(f"==> Questions:    {DATASET_PATH}")
    print()

    print("Saving live database snapshot to volume ... ", end="", flush=True)
    stmt_count = save_snapshot()
    print(f"done ({stmt_count} statements)")

    try:
        print("Loading eval seed data ... ", end="", flush=True)
        counts = load_seed(SEED_SQL_PATH)
        summary = ", ".join(f"{t}: {n}" for t, n in counts.items())
        print(f"done ({summary})")
        print()
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
