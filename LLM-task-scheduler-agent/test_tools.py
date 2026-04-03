from pprint import pprint
from datetime import datetime

from tools import (
    parse_tasks,
    parse_availability,
    build_schedule_tool,
    validate_schedule_tool,
)


def main() -> None:
    reference = datetime.now()

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

    print("\n=== PARSE TASKS ===")
    parsed_tasks = parse_tasks(raw_tasks, reference=reference)
    pprint(parsed_tasks)

    print("\n=== PARSE AVAILABILITY ===")
    parsed_slots = parse_availability(raw_availability, reference=reference)
    pprint(parsed_slots)

    if parsed_tasks["errors"]:
        print("\nTask parsing errors found. Exiting.")
        return

    if parsed_slots["errors"]:
        print("\nAvailability parsing errors found. Exiting.")
        return

    print("\n=== BUILD SCHEDULE ===")
    schedule = build_schedule_tool(
        task_dicts=parsed_tasks["tasks"],
        slot_dicts=parsed_slots["slots"],
    )
    pprint(schedule)

    print("\n=== VALIDATE SCHEDULE ===")
    validation = validate_schedule_tool(
        task_dicts=parsed_tasks["tasks"],
        scheduled_blocks=schedule["scheduled_blocks"],
    )
    pprint(validation)


if __name__ == "__main__":
    main()