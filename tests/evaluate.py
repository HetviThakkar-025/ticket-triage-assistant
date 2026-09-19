#!/usr/bin/env python
"""Run every sample test case through the LIVE decision pipeline.

Usage:
    # terminal 1
    uvicorn src.api:app --reload
    # terminal 2
    python tests/evaluate.py

Only the natural-language `message` is sent to the API -- the structured
columns in sample_test_cases.json are used as the answer key, never as input.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = ROOT / "sample_test_cases.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8000", help="base URL of the API")
    parser.add_argument("--cases", default=str(CASES_PATH), help="path to the test-case JSON")
    parser.add_argument(
        "--sleep", type=float, default=0.0, help="seconds to wait between calls"
    )
    return parser.parse_args()


def authenticate(api: str) -> str:
    """Register a throwaway evaluation user and return its bearer token."""
    credentials = {
        "email": f"eval-{uuid.uuid4().hex[:10]}@example.com",
        "password": "evaluation-password",
    }
    response = requests.post(f"{api}/register", json=credentials, timeout=30)
    response.raise_for_status()
    response = requests.post(f"{api}/login", json=credentials, timeout=30)
    response.raise_for_status()
    return response.json()["access_token"]


def main() -> int:
    args = parse_args()
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))

    try:
        requests.get(f"{args.api}/health", timeout=10).raise_for_status()
    except requests.RequestException as exc:
        print(f"Cannot reach the API at {args.api}: {exc}")
        print("Start it first:  uvicorn src.api:app --reload")
        return 2

    token = authenticate(args.api)
    headers = {"Authorization": f"Bearer {token}"}

    correct = 0
    failures: list[str] = []

    print(f"\nEvaluating {len(cases)} sample cases against {args.api}\n")
    print("-" * 78)

    for case in cases:
        case_id = case.get("case_id", "?")
        expected = case["expected_action"]
        response = requests.post(
            f"{args.api}/tickets", json={"message": case["message"]}, headers=headers, timeout=120
        )
        response.raise_for_status()
        body = response.json()
        actual = body["action"]

        is_correct = actual == expected
        correct += is_correct
        mark = "PASS" if is_correct else "FAIL"

        print(f"[{mark}] {case_id}  {case['message']}")
        print(f"        expected : {expected}")
        print(f"        actual   : {actual}  (confidence {body['confidence']:.2f})")
        print(f"        sources  : {', '.join(body['sources']) or '-'}")
        print(f"        reason   : {body['reason']}")
        print("-" * 78)

        if not is_correct:
            failures.append(f"{case_id}: expected {expected}, got {actual}")
        if args.sleep:
            time.sleep(args.sleep)

    total = len(cases)
    accuracy = (correct / total * 100) if total else 0.0
    print(f"\n{total} test cases / Correct: {correct} / Accuracy: {accuracy:.1f}%\n")

    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
