# evaluation scripts

from __future__ import annotations

import json
from typing import Any, Dict, List

from openai import APIConnectionError

from agent import run_agent, run_agent_direct


TEST_CASES: List[Dict[str, Any]] = [
    {
        "name": "basic_feasible_case",
        "tasks": """Finish NLP homework | Friday 5pm | 180 | high
Email professor | tomorrow 12pm | 15 | high
Study for stats exam | Thursday 8pm | 120 | medium
""",
        "availability": """Wednesday: 6pm-10pm
Thursday: 1pm-5pm
Friday: 9am-12pm
""",
        "expected": {
            "should_succeed": True,
            "should_be_valid": True,
            "max_unscheduled_tasks": 0,
        },
    },
    {
        "name": "not_enough_time_case",
        "tasks": """Write paper draft | Thursday 5pm | 300 | high
Prepare presentation | Thursday 3pm | 180 | high
""",
        "availability": """Wednesday: 6pm-8pm
Thursday: 1pm-2pm
""",
        "expected": {
            "should_succeed": True,
            "should_be_valid": True,
            "min_unscheduled_tasks": 1,
        },
    },
    {
        "name": "multiple_small_tasks",
        "tasks": """Email TA | tomorrow 10am | 10 | high
Review notes | Friday 3pm | 45 | medium
Workout | Friday 8pm | 60 | low
""",
        "availability": """Thursday: 8am-11am
Friday: 5pm-9pm
""",
        "expected": {
            "should_succeed": True,
            "should_be_valid": True,
        },
    },
]

REFERENCE_DATETIME = "2026-04-08T09:00:00"


def score_case(agent_result: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, Any]:
    success = agent_result.get("success", False)
    schedule_result = agent_result.get("schedule_result") or {}
    validation_result = agent_result.get("validation_result") or {}

    is_valid = validation_result.get("is_valid", False)
    unscheduled_tasks = schedule_result.get("unscheduled_tasks", [])
    scheduled_blocks = schedule_result.get("scheduled_blocks", [])

    metrics = {
        "agent_success": success,
        "valid_schedule": is_valid,
        "scheduled_blocks_count": len(scheduled_blocks),
        "unscheduled_tasks_count": len(unscheduled_tasks),
        "passed": True,
        "fail_reasons": [],
    }

    if "should_succeed" in expected and success != expected["should_succeed"]:
        metrics["passed"] = False
        metrics["fail_reasons"].append(
            f"Expected success={expected['should_succeed']}, got {success}"
        )

    if "should_be_valid" in expected and is_valid != expected["should_be_valid"]:
        metrics["passed"] = False
        metrics["fail_reasons"].append(
            f"Expected valid_schedule={expected['should_be_valid']}, got {is_valid}"
        )

    if "max_unscheduled_tasks" in expected:
        if len(unscheduled_tasks) > expected["max_unscheduled_tasks"]:
            metrics["passed"] = False
            metrics["fail_reasons"].append(
                f"Expected unscheduled_tasks_count <= {expected['max_unscheduled_tasks']}, "
                f"got {len(unscheduled_tasks)}"
            )

    if "min_unscheduled_tasks" in expected:
        if len(unscheduled_tasks) < expected["min_unscheduled_tasks"]:
            metrics["passed"] = False
            metrics["fail_reasons"].append(
                f"Expected unscheduled_tasks_count >= {expected['min_unscheduled_tasks']}, "
                f"got {len(unscheduled_tasks)}"
            )

    return metrics


def run_evaluation(
    model: str = "gpt-4o-mini",
    max_steps: int = 8,
    use_direct_fallback: bool = True,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for case in TEST_CASES:
        print(f"\nRunning case: {case['name']}")

        try:
            agent_result = run_agent(
                raw_tasks_text=case["tasks"],
                raw_availability_text=case["availability"],
                model=model,
                max_steps=max_steps,
                verbose=False,
                reference_datetime=REFERENCE_DATETIME,
            )
        except APIConnectionError:
            if not use_direct_fallback:
                raise
            print("OpenAI API unavailable, falling back to direct tool evaluation.")
            agent_result = run_agent_direct(
                raw_tasks_text=case["tasks"],
                raw_availability_text=case["availability"],
                reference_datetime=REFERENCE_DATETIME,
            )

        metrics = score_case(agent_result, case["expected"])

        results.append(
            {
                "case_name": case["name"],
                "metrics": metrics,
                "agent_result": agent_result,
            }
        )

    return results


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_cases = len(results)
    passed_cases = sum(1 for r in results if r["metrics"]["passed"])
    successful_runs = sum(1 for r in results if r["metrics"]["agent_success"])
    valid_schedules = sum(1 for r in results if r["metrics"]["valid_schedule"])

    avg_scheduled_blocks = (
        sum(r["metrics"]["scheduled_blocks_count"] for r in results) / total_cases
        if total_cases > 0
        else 0.0
    )

    avg_unscheduled_tasks = (
        sum(r["metrics"]["unscheduled_tasks_count"] for r in results) / total_cases
        if total_cases > 0
        else 0.0
    )

    return {
        "total_cases": total_cases,
        "pass_rate": passed_cases / total_cases if total_cases > 0 else 0.0,
        "agent_success_rate": successful_runs / total_cases if total_cases > 0 else 0.0,
        "valid_schedule_rate": valid_schedules / total_cases if total_cases > 0 else 0.0,
        "avg_scheduled_blocks": avg_scheduled_blocks,
        "avg_unscheduled_tasks": avg_unscheduled_tasks,
    }


def print_results(results: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("PER-CASE RESULTS")
    print("=" * 80)

    for result in results:
        case_name = result["case_name"]
        metrics = result["metrics"]

        print(f"\nCase: {case_name}")
        print(json.dumps(metrics, indent=2))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    results = run_evaluation()
    summary = summarize_results(results)
    print_results(results, summary)
