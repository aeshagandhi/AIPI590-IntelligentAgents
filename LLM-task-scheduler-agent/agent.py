# custom agent loop

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from llm import call_llm
from parser import parse_llm_response
from tools import TOOLS, run_tool


MAX_STEPS = 8


def get_tool_descriptions() -> str:
    lines = []
    for tool_name, tool_meta in TOOLS.items():
        lines.append(f"- {tool_name}: {tool_meta['description']}")
    return "\n".join(lines)


def build_system_prompt() -> str:
    return f"""
You are a task scheduling agent.

Your job is to help the user convert a to-do list and availability into a schedule.

You may use tools to:
1. parse raw tasks
2. parse raw availability
3. build a schedule
4. validate a schedule

Available tools:
{get_tool_descriptions()}

You must ALWAYS respond with valid JSON only.
Do not include markdown fences.
Do not include any text before or after the JSON.

Valid response formats:

Tool call:
{{
  "type": "tool_call",
  "tool": "parse_tasks",
  "args": {{
    "raw_tasks_text": "Finish homework | Friday 5pm | 180 | high"
  }}
}}

Final response:
{{
  "type": "final",
  "message": "Here is your proposed schedule..."
}}

Behavior rules:
- If raw tasks have not yet been parsed, call parse_tasks first.
- If raw availability has not yet been parsed, call parse_availability first.
- After both are parsed, call build_schedule.
- After building a schedule, call validate_schedule.
- Once validation is complete, return a final answer.
- Use only the tool names provided.
- Never invent tool outputs.
- If a tool returns errors, explain them in the final answer instead of continuing indefinitely.
""".strip()


def build_user_prompt(
    raw_tasks_text: str,
    raw_availability_text: str,
    reference_datetime: Optional[str] = None,
) -> str:
    reference_datetime = reference_datetime or datetime.now().isoformat()

    return f"""
Reference datetime: {reference_datetime}

User task list:
{raw_tasks_text}

User availability:
{raw_availability_text}

Create a schedule using the available tools.
""".strip()


def _json_dumps_pretty(data: Any) -> str:
    return json.dumps(data, indent=2, default=str)


def run_agent(
    raw_tasks_text: str,
    raw_availability_text: str,
    model: str = "gpt-4o-mini",
    max_steps: int = MAX_STEPS,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the custom agent loop.
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": build_system_prompt()},
        {
            "role": "user",
            "content": build_user_prompt(
                raw_tasks_text=raw_tasks_text,
                raw_availability_text=raw_availability_text,
            ),
        },
    ]

    trace: List[Dict[str, Any]] = []
    latest_schedule_result: Optional[Dict[str, Any]] = None
    latest_validation_result: Optional[Dict[str, Any]] = None

    for step in range(1, max_steps + 1):
        raw_llm_output = call_llm(messages=messages, model=model, temperature=0.0)

        if verbose:
            print(f"\n=== STEP {step}: RAW LLM OUTPUT ===")
            print(raw_llm_output)

        try:
            parsed = parse_llm_response(raw_llm_output)
        except Exception as exc:
            return {
                "success": False,
                "error": f"Failed to parse LLM output at step {step}: {exc}",
                "trace": trace,
                "messages": messages,
            }

        trace.append(
            {
                "step": step,
                "llm_response": parsed,
            }
        )

        if parsed["type"] == "final":
            return {
                "success": True,
                "final_message": parsed["message"],
                "schedule_result": latest_schedule_result,
                "validation_result": latest_validation_result,
                "trace": trace,
                "messages": messages,
            }

        tool_name = parsed["tool"]
        tool_args = parsed["args"]

        try:
            tool_result = run_tool(tool_name, **tool_args)
        except Exception as exc:
            tool_result = {
                "error": str(exc),
                "tool_name": tool_name,
                "tool_args": tool_args,
            }

        if verbose:
            print(f"\n=== STEP {step}: TOOL CALL ===")
            print(tool_name)
            print(_json_dumps_pretty(tool_args))

            print(f"\n=== STEP {step}: TOOL RESULT ===")
            print(_json_dumps_pretty(tool_result))

        if tool_name == "build_schedule" and isinstance(tool_result, dict):
            latest_schedule_result = tool_result

        if tool_name == "validate_schedule" and isinstance(tool_result, dict):
            latest_validation_result = tool_result

        trace[-1]["tool_name"] = tool_name
        trace[-1]["tool_args"] = tool_args
        trace[-1]["tool_result"] = tool_result

        messages.append({"role": "assistant", "content": raw_llm_output})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Tool '{tool_name}' returned:\n"
                    f"{_json_dumps_pretty(tool_result)}\n\n"
                    "Decide the next step. Return JSON only."
                ),
            }
        )

    return {
        "success": False,
        "error": f"Agent exceeded max_steps={max_steps} without producing a final answer.",
        "schedule_result": latest_schedule_result,
        "validation_result": latest_validation_result,
        "trace": trace,
        "messages": messages,
    }


if __name__ == "__main__":
    raw_tasks = """
    Finish NLP homework | Friday 5pm | 180 | high
    Email professor | tomorrow 12pm | 15 | high
    Study for stats exam | Thursday 8pm | 120 | medium
    """

    raw_availability = """
    Wednesday: 6pm-10pm
    Thursday: 1pm-5pm
    Friday: 9am-12pm
    """

    result = run_agent(
        raw_tasks_text=raw_tasks,
        raw_availability_text=raw_availability,
        verbose=True,
    )

    print("\n=== FINAL RESULT ===")
    print(json.dumps(result, indent=2, default=str))