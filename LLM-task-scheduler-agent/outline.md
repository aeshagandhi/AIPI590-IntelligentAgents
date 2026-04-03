# Project Proposal: Smart Task-to-Schedule LLM Agent

## 1. Project Overview

This project builds an LLM-based agent that converts natural-language to-do lists into structured, time-blocked schedules. Users often write tasks informally (e.g., “finish assignment by Friday, study for exam this weekend”), but turning those into an actual plan requires manual effort.

The goal of this agent is to automate that process by:

* extracting tasks, deadlines, and durations from text
* reasoning about priorities and time constraints
* assigning tasks to available time slots
* producing a clear, executable schedule

The system will be implemented as an interactive **Streamlit application** where users can input tasks and receive a proposed schedule.

---

## 2. System Architecture

The system will consist of four main components:

### A. User Interface (Streamlit)

* Text input for user task list
* Optional inputs for availability (e.g., working hours, busy times)
* Display of:

  * parsed tasks
  * generated schedule
  * explanations of decisions

---

### B. Agent Loop (Core Requirement)

The agent will implement a custom loop (no frameworks) that:

1. receives the user request and current state
2. queries the LLM for the next action
3. determines whether to:

   * call a tool
   * or produce a final answer
4. executes tools and feeds results back into the loop
5. repeats until a schedule is produced

This enables multi-step reasoning such as:

* parsing tasks
* checking availability
* resolving conflicts
* refining the schedule

---

### C. LLM Reasoning Layer

The LLM will:

* interpret user input
* decide which tools to use
* reason about task prioritization and scheduling
* generate the final explanation

A structured output format (e.g., JSON tool calls) will be used to ensure reliable interaction between the LLM and tools.

---

### D. Tools (Minimum of 3)

The agent will use at least three tools:

#### 1. Task Parsing Tool

* Input: raw natural-language task list
* Output: structured tasks with fields such as:

  * task name
  * deadline
  * estimated duration
  * priority (if inferred)

#### 2. Availability Tool

* Input: user-defined working hours and/or busy times
* Output: available time slots for scheduling

#### 3. Scheduling Tool

* Input: tasks + available time slots
* Output: a feasible schedule that:

  * avoids conflicts
  * respects deadlines
  * prioritizes urgent tasks

#### (Optional) 4. Conflict Checker / Validator

* Verifies schedule correctness
* Ensures no overlaps and deadline violations

---

## 3. Agent Behavior

The agent will follow a planning-and-execution workflow:

1. Parse tasks from user input
2. Identify missing or ambiguous information (if any)
3. Retrieve available time slots
4. Assign tasks to time blocks using reasoning + scheduling logic
5. Validate and refine the schedule
6. Return final schedule with explanation

The agent may iterate through multiple tool calls before producing the final result.

---

## 4. User Experience

The Streamlit app will allow users to:

* Paste a to-do list in natural language
* Optionally specify availability (e.g., “free after 6pm”)
* View:

  * structured interpretation of tasks
  * generated schedule
  * explanation of how tasks were prioritized and placed

This creates a clear before/after transformation:

* **Input:** messy text
* **Output:** structured, actionable plan

---

## 5. Evaluation Plan (Quantitative)

To evaluate the agent’s performance, a benchmark set of task scenarios will be created with known constraints.

### Metrics

* **Task Completion Rate:**
  Percentage of tasks scheduled before their deadlines

* **Conflict Rate:**
  Number of overlapping scheduled tasks

* **Deadline Violation Rate:**
  Number of tasks scheduled after their deadline

* **Schedule Utilization:**
  Percentage of available time slots used effectively

* **Tool Usage Accuracy (optional):**
  Whether the agent selects appropriate tools for each step

---

### Baseline Comparison

The agent will be compared against a simple heuristic baseline, such as:

* Earliest Deadline First (EDF)
* Greedy scheduling without reasoning

This allows quantitative demonstration of whether the agent improves scheduling quality.

---

## 6. Deliverables

### A. Streamlit Application

* Fully functional UI for interacting with the agent
* Displays schedule and explanations

### B. LLM Agent Implementation

* Custom agent loop (no frameworks)
* Tool integration and execution logic

### C. Evaluation Results

* Benchmark dataset of task scenarios
* Quantitative metrics comparing agent vs baseline

### D. GitHub Repository

* Clean code structure (agent, tools, UI, evaluation)
* README with:

  * system overview
  * setup instructions
  * example usage

### E. Demo Video (≤ 6 minutes)

* Problem overview
* Agent architecture
* Live demo
* Evaluation results

---

## 7. Scope and Feasibility

To keep the project manageable:

* Scheduling will be limited to short time horizons (e.g., a few days or a week)
* Calendar integration will be simulated (no external API required)
* Task durations may be user-provided or estimated simply
* Scheduling logic will use straightforward heuristics combined with LLM reasoning

This ensures the system is fully implementable within the assignment timeline while still demonstrating meaningful agent capabilities.

---

## 8. Summary

This project builds an LLM-based planning agent that transforms unstructured task lists into structured schedules using multi-step reasoning and tool interaction. It demonstrates key agent capabilities including planning, tool use, and iterative refinement, while remaining practical, interactive, and easy to evaluate.

V