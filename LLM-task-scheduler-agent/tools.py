# tool definitions and execution

# 3 tools:
# parse_tasks
# parse_availability
# build_schedule


from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import List, Dict, Any

from dateutil import parser as date_parser

from scheduler import Task, TimeSlot, build_schedule, validate_schedule


DAY_NAME_TO_INDEX = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def _clean_line(line: str) -> str:
    return line.strip().lstrip("-").strip()


def _next_weekday(base_date: datetime, target_weekday: int) -> datetime:
    days_ahead = target_weekday - base_date.weekday()
    if days_ahead < 0:
        days_ahead += 7
    return base_date + timedelta(days=days_ahead)


def _parse_datetime_flexible(text: str, reference: datetime | None = None) -> datetime:
    """
    Parse strings like:
      - Friday 5pm
      - tomorrow 12pm
      - 2026-04-05 17:00
    """
    reference = reference or datetime.now()
    lower = text.strip().lower()

    if lower.startswith("tomorrow"):
        remainder = lower.replace("tomorrow", "", 1).strip()
        base = reference + timedelta(days=1)
        if remainder:
            parsed_time = date_parser.parse(remainder, default=base)
            return parsed_time.replace(year=base.year, month=base.month, day=base.day)
        return base.replace(hour=17, minute=0, second=0, microsecond=0)

    if lower.startswith("today"):
        remainder = lower.replace("today", "", 1).strip()
        base = reference
        if remainder:
            parsed_time = date_parser.parse(remainder, default=base)
            return parsed_time.replace(year=base.year, month=base.month, day=base.day)
        return base.replace(hour=17, minute=0, second=0, microsecond=0)

    for day_name, day_idx in DAY_NAME_TO_INDEX.items():
        if lower.startswith(day_name):
            remainder = lower[len(day_name):].strip()
            base_day = _next_weekday(reference, day_idx)
            if remainder:
                parsed_time = date_parser.parse(remainder, default=base_day)
                return parsed_time.replace(
                    year=base_day.year,
                    month=base_day.month,
                    day=base_day.day,
                )
            return base_day.replace(hour=17, minute=0, second=0, microsecond=0)

    return date_parser.parse(text, default=reference)


def parse_tasks(raw_tasks_text: str, reference: datetime | None = None) -> Dict[str, Any]:
    """
    Expected input format: one task per line
    Example:
      Finish NLP homework | Friday 5pm | 180 | high
      Email professor | tomorrow 12pm | 15 | high

    Priority is optional. If omitted, defaults to medium.
    """
    reference = reference or datetime.now()
    tasks: List[Task] = []
    errors: List[str] = []

    lines = [line for line in raw_tasks_text.splitlines() if line.strip()]
    for idx, raw_line in enumerate(lines, start=1):
        line = _clean_line(raw_line)
        parts = [part.strip() for part in line.split("|")]

        if len(parts) < 3:
            errors.append(
                f"Line {idx} must have at least 3 fields separated by '|': "
                f"name | deadline | duration_minutes [| priority]"
            )
            continue

        name = parts[0]
        deadline_text = parts[1]
        duration_text = parts[2]
        priority = parts[3].lower() if len(parts) >= 4 else "medium"

        try:
            deadline = _parse_datetime_flexible(deadline_text, reference=reference)
        except Exception as exc:
            errors.append(f"Line {idx}: could not parse deadline '{deadline_text}': {exc}")
            continue

        try:
            duration_minutes = int(duration_text)
            if duration_minutes <= 0:
                raise ValueError("duration must be positive")
        except Exception as exc:
            errors.append(f"Line {idx}: could not parse duration '{duration_text}': {exc}")
            continue

        tasks.append(
            Task(
                name=name,
                deadline=deadline,
                duration_minutes=duration_minutes,
                priority=priority,
            )
        )

    return {
        "tasks": [task.to_dict() for task in tasks],
        "errors": errors,
    }


def parse_availability(raw_availability_text: str, reference: datetime | None = None) -> Dict[str, Any]:
    """
    Expected input format: one slot per line
    Example:
      Wednesday: 6pm-10pm
      Thursday: 1pm-5pm
      Friday: 9am-12pm

    Each line becomes one slot.
    """
    reference = reference or datetime.now()
    slots: List[TimeSlot] = []
    errors: List[str] = []

    lines = [line for line in raw_availability_text.splitlines() if line.strip()]
    pattern = re.compile(r"^\s*([A-Za-z]+)\s*:\s*(.+?)\s*-\s*(.+?)\s*$")

    for idx, raw_line in enumerate(lines, start=1):
        line = _clean_line(raw_line)
        match = pattern.match(line)
        if not match:
            errors.append(
                f"Line {idx} must look like 'Wednesday: 6pm-10pm'. Got: {raw_line}"
            )
            continue

        day_name = match.group(1).lower()
        start_text = match.group(2).strip()
        end_text = match.group(3).strip()

        if day_name not in DAY_NAME_TO_INDEX:
            errors.append(f"Line {idx}: unknown day '{day_name}'.")
            continue

        base_day = _next_weekday(reference, DAY_NAME_TO_INDEX[day_name])

        try:
            start_dt = date_parser.parse(start_text, default=base_day).replace(
                year=base_day.year,
                month=base_day.month,
                day=base_day.day,
            )
            end_dt = date_parser.parse(end_text, default=base_day).replace(
                year=base_day.year,
                month=base_day.month,
                day=base_day.day,
            )
        except Exception as exc:
            errors.append(f"Line {idx}: could not parse time range: {exc}")
            continue

        if end_dt <= start_dt:
            errors.append(
                f"Line {idx}: end time must be after start time. Got {start_text}-{end_text}."
            )
            continue

        slots.append(TimeSlot(start=start_dt, end=end_dt))

    return {
        "slots": [slot.to_dict() for slot in slots],
        "errors": errors,
    }


def _tasks_from_dicts(task_dicts: List[dict]) -> List[Task]:
    tasks: List[Task] = []
    for item in task_dicts:
        tasks.append(
            Task(
                name=item["name"],
                deadline=datetime.fromisoformat(item["deadline"]),
                duration_minutes=int(item["duration_minutes"]),
                priority=item.get("priority", "medium"),
                metadata=item.get("metadata", {}),
            )
        )
    return tasks


def _slots_from_dicts(slot_dicts: List[dict]) -> List[TimeSlot]:
    slots: List[TimeSlot] = []
    for item in slot_dicts:
        slots.append(
            TimeSlot(
                start=datetime.fromisoformat(item["start"]),
                end=datetime.fromisoformat(item["end"]),
            )
        )
    return slots


def build_schedule_tool(task_dicts: List[dict], slot_dicts: List[dict]) -> Dict[str, Any]:
    tasks = _tasks_from_dicts(task_dicts)
    slots = _slots_from_dicts(slot_dicts)

    result = build_schedule(tasks, slots)
    return result.to_dict()


def validate_schedule_tool(task_dicts: List[dict], scheduled_blocks: List[dict]) -> Dict[str, Any]:
    tasks = _tasks_from_dicts(task_dicts)

    from scheduler import ScheduleBlock  # local import to avoid circular issues if needed

    blocks = [
        ScheduleBlock(
            task_name=block["task_name"],
            start=datetime.fromisoformat(block["start"]),
            end=datetime.fromisoformat(block["end"]),
            minutes_scheduled=int(block["minutes_scheduled"]),
        )
        for block in scheduled_blocks
    ]

    return validate_schedule(blocks, tasks)


TOOLS = {
    "parse_tasks": {
        "description": "Parse raw task text into structured tasks.",
        "function": parse_tasks,
    },
    "parse_availability": {
        "description": "Parse raw availability text into structured time slots.",
        "function": parse_availability,
    },
    "build_schedule": {
        "description": "Build a feasible schedule from parsed task_dicts and parsed slot_dicts.",
        "function": build_schedule_tool,
    },
    "validate_schedule": {
        "description": "Validate scheduled_blocks against parsed task_dicts for overlaps and deadline violations.",
        "function": validate_schedule_tool,
    },
}


def run_tool(tool_name: str, **kwargs) -> Dict[str, Any]:
    if tool_name not in TOOLS:
        raise ValueError(f"Unknown tool: {tool_name}")

    # normalize common LLM-produced argument names
    if tool_name == "build_schedule":
        if "tasks" in kwargs and "task_dicts" not in kwargs:
            kwargs["task_dicts"] = kwargs.pop("tasks")
        if "availability" in kwargs and "slot_dicts" not in kwargs:
            kwargs["slot_dicts"] = kwargs.pop("availability")
        if "slots" in kwargs and "slot_dicts" not in kwargs:
            kwargs["slot_dicts"] = kwargs.pop("slots")

    if tool_name == "validate_schedule":
        if "tasks" in kwargs and "task_dicts" not in kwargs:
            kwargs["task_dicts"] = kwargs.pop("tasks")
        if "schedule" in kwargs and "scheduled_blocks" not in kwargs:
            kwargs["scheduled_blocks"] = kwargs.pop("schedule")
        if "blocks" in kwargs and "scheduled_blocks" not in kwargs:
            kwargs["scheduled_blocks"] = kwargs.pop("blocks")

    tool_fn = TOOLS[tool_name]["function"]
    return tool_fn(**kwargs)