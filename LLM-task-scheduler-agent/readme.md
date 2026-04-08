# LLM Task Scheduler Agent

Video: https://youtu.be/ZPixhFpnKWQ
## Context

This assignment focuses on designing and evaluating a custom LLM-based agent. The system takes a natural-language task list plus user availability and converts them into a structured, time-blocked schedule.

## Project Summary

Many people write to-do lists in an informal way:

```text
Finish NLP homework | Friday 5pm | 180 | high
Email professor | tomorrow 12pm | 15 | high
Study for stats exam | Thursday 8pm | 120 | medium
```

Turning that list into an actionable schedule requires several reasoning steps:

1. Parse tasks into structured data.
2. Interpret relative dates such as `tomorrow` or weekday names.
3. Parse available time windows.
4. Allocate tasks into feasible time slots before their deadlines.
5. Validate that the resulting schedule has no overlaps or deadline violations.

This repository implements that workflow as a lightweight custom agent loop without relying on an external agent framework.

## Objectives

The project was designed to demonstrate:

- A custom LLM agent loop
- Tool use and multi-step reasoning
- Structured input/output handling
- Schedule generation under time constraints
- Quantitative evaluation on benchmark cases
- A simple interactive frontend for demos

## Repository Structure

- [agent.py](/Users/aeshagandhi/Downloads/MIDS-Sp26/Agents/Assignments/LLM-task-scheduler-agent/agent.py): custom agent loop and deterministic direct-execution fallback
- [tools.py](/Users/aeshagandhi/Downloads/MIDS-Sp26/Agents/Assignments/LLM-task-scheduler-agent/tools.py): task parsing, availability parsing, schedule construction, and validation tool wrappers
- [scheduler.py](/Users/aeshagandhi/Downloads/MIDS-Sp26/Agents/Assignments/LLM-task-scheduler-agent/scheduler.py): core scheduling and validation logic
- [parser.py](/Users/aeshagandhi/Downloads/MIDS-Sp26/Agents/Assignments/LLM-task-scheduler-agent/parser.py): validation of LLM JSON responses
- [llm.py](/Users/aeshagandhi/Downloads/MIDS-Sp26/Agents/Assignments/LLM-task-scheduler-agent/llm.py): OpenAI API wrapper
- [app.py](/Users/aeshagandhi/Downloads/MIDS-Sp26/Agents/Assignments/LLM-task-scheduler-agent/app.py): Streamlit app
- [eval.py](/Users/aeshagandhi/Downloads/MIDS-Sp26/Agents/Assignments/LLM-task-scheduler-agent/eval.py): benchmark evaluation script
- [test_tools.py](/Users/aeshagandhi/Downloads/MIDS-Sp26/Agents/Assignments/LLM-task-scheduler-agent/test_tools.py): local tool-level smoke test
- [outline.md](/Users/aeshagandhi/Downloads/MIDS-Sp26/Agents/Assignments/LLM-task-scheduler-agent/outline.md): project proposal and planning notes
- [requirements.txt](/Users/aeshagandhi/Downloads/MIDS-Sp26/Agents/Assignments/LLM-task-scheduler-agent/requirements.txt): Python dependencies

## System Design

### 1. Input Layer

The user provides:

- A task list with one task per line
- Availability windows with one slot per line

Expected task format:

```text
Task Name | Deadline | Duration Minutes | Priority
```

Expected availability format:

```text
Day: start-end
```

Examples:

```text
Finish NLP homework | Friday 5pm | 180 | high
Email professor | tomorrow 12pm | 15 | high
```

```text
Wednesday: 6pm-10pm
Thursday: 1pm-5pm
Friday: 9am-12pm
```

### 2. Agent Loop

The agent loop in `agent.py` is intentionally simple and transparent:

1. Send the user request and system instructions to the LLM.
2. Require the model to respond with valid JSON only.
3. Parse the JSON response.
4. If the response is a tool call, execute the requested tool.
5. Feed the tool result back into the conversation.
6. Repeat until the model returns a final response.

This satisfies the assignment requirement of a custom loop that coordinates reasoning and tool use.

### 3. Tools

The project uses four tools:

- `parse_tasks`: converts raw text into structured tasks with names, deadlines, durations, and priorities
- `parse_availability`: converts raw availability text into structured time slots
- `build_schedule`: assigns task durations to available time windows before deadlines
- `validate_schedule`: checks for overlaps, deadline violations, and task completion

### 4. Scheduling Strategy

The scheduler uses a greedy heuristic:

- Tasks are sorted primarily by earliest deadline.
- Priority is used as a tie-breaker.
- Shorter tasks are preferred when deadline and priority are equal.
- Tasks may be split across multiple time slots if needed.
- Only time before each task’s deadline can be used.

This keeps the implementation understandable while still producing reasonable schedules for short planning horizons.

## Implementation Details

### Task Parsing

Task parsing is handled in `tools.py`. Each valid line is converted into a `Task` object with:

- `name`
- `deadline`
- `duration_minutes`
- `priority`

Relative deadlines such as `today`, `tomorrow`, and weekday names are supported using `python-dateutil` and a reference datetime.

### Availability Parsing

Availability is parsed into `TimeSlot` objects. Each slot stores:

- `start`
- `end`
- derived duration in minutes

Invalid formats or impossible time ranges are reported as parsing errors.

### Schedule Construction

The scheduling logic lives in `scheduler.py` and uses the following data structures:

- `Task`
- `TimeSlot`
- `ScheduleBlock`
- `SchedulingResult`

The scheduler:

1. sorts tasks by urgency
2. merges overlapping or adjacent availability slots
3. allocates time from the beginning of each usable slot
4. preserves unused slot time for future tasks
5. records any remaining unscheduled task minutes

### Validation

Validation checks:

- no overlapping scheduled blocks
- no scheduled block extending past its task deadline
- how many requested minutes were actually scheduled per task

This makes it possible to separate “the agent ran successfully” from “every task fit in the schedule.”

## Evaluation

The benchmark in `eval.py` evaluates the system on three scenarios:

- a feasible case where all tasks should fit
- an infeasible case where at least one task should remain unscheduled
- a mixed case with several small tasks

The scoring logic tracks:

- agent success rate
- valid schedule rate
- pass rate across benchmark cases
- average number of scheduled blocks
- average number of unscheduled tasks

### Important Evaluation Note

The benchmark uses relative date expressions like `tomorrow`, so `eval.py` now fixes the reference datetime to a specific value to keep results deterministic. If the OpenAI API is unavailable, the evaluation script falls back to a direct tool-execution path so the scheduling pipeline can still be tested locally.

## Current Results

Using the current codebase and the deterministic benchmark setup, the evaluation script reports:

- `pass_rate = 1.0`
- `agent_success_rate = 1.0`
- `valid_schedule_rate = 1.0`

This means all included benchmark scenarios currently pass under the fixed reference-date setup.

If live API access is available, `eval.py` will first attempt the full LLM-driven agent loop before falling back to the direct tool pipeline.

## Setup Instructions

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add environment variables

Create a `.env` file with your OpenAI API key:

```text
OPENAI_API_KEY=your_api_key_here
```

## How to Run

### Run the Streamlit app

```bash
streamlit run app.py
```

This launches the interactive UI where you can paste tasks and availability and generate a schedule.

### Run the benchmark evaluation

```bash
python eval.py
```

This runs the benchmark cases and prints per-case results plus summary statistics.

### Run the tool smoke test

```bash
python test_tools.py
```

This is useful for debugging parsing and scheduling behavior without involving the LLM loop.

## Example Workflow

Example input:

```text
Tasks
Finish NLP homework | Friday 5pm | 180 | high
Email professor | tomorrow 12pm | 15 | high
Study for stats exam | Thursday 8pm | 120 | medium
```

```text
Availability
Wednesday: 6pm-10pm
Thursday: 1pm-5pm
Friday: 9am-12pm
```

Typical system behavior:

1. Parse the tasks into structured objects.
2. Parse the available time windows.
3. Schedule the most urgent tasks first.
4. Split tasks across multiple slots if necessary.
5. Validate the finished plan.
6. Return the schedule and any unscheduled tasks.

## Assumptions

This implementation makes several simplifying assumptions:

- planning horizon is short, typically within the same week
- task durations are provided by the user
- no external calendar integration is used
- each availability line represents one free time slot
- scheduling uses a greedy heuristic rather than global optimization

These choices were intentional to keep the project focused on agent design, tool use, and evaluation.

## Limitations

- The benchmark is small and does not represent every scheduling scenario.
- The LLM agent depends on reliable JSON responses.
- Relative date parsing depends on a reference datetime.
- The scheduler does not model breaks, task dependencies, travel time, or personal preferences.
- The current heuristic may not always find the globally best schedule even when one exists.

## Possible Future Improvements

- expand the benchmark dataset with more diverse cases
- add support for recurring tasks and dependencies
- include user preferences such as morning versus evening work
- compare the LLM agent against a stronger heuristic baseline
- add richer natural-language explanations for scheduling decisions
- integrate with a real calendar API

## What This Project Demonstrates

This project demonstrates that a relatively small custom agent can coordinate multiple tools to solve a useful planning task. It also shows the importance of evaluation design: when benchmarks include relative dates, the reference datetime must be controlled so results remain meaningful and reproducible.

## Submission Notes

This repository includes:

- a custom LLM agent loop
- at least three tools
- a working Streamlit interface
- scheduling and validation logic
- an evaluation script with quantitative results
- clear modular source files for inspection

## Author

Aesha Gandhi  
AIPI 590: Intelligent Agents
