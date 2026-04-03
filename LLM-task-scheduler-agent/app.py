# streamlit UI

from __future__ import annotations

import json
import streamlit as st

from agent import run_agent


st.set_page_config(page_title="Task Scheduler Agent", page_icon="🗓️", layout="wide")


DEFAULT_TASKS = """Finish NLP homework | Friday 5pm | 180 | high
Email professor | tomorrow 12pm | 15 | high
Study for stats exam | Thursday 8pm | 120 | medium
"""

DEFAULT_AVAILABILITY = """Wednesday: 6pm-10pm
Thursday: 1pm-5pm
Friday: 9am-12pm
"""


def format_schedule_blocks(schedule_result: dict) -> list[dict]:
    blocks = schedule_result.get("scheduled_blocks", [])
    formatted = []
    for block in blocks:
        formatted.append(
            {
                "Task": block["task_name"],
                "Start": block["start"],
                "End": block["end"],
                "Minutes": block["minutes_scheduled"],
            }
        )
    return formatted


def main() -> None:
    st.title("🗓️ Smart Task-to-Schedule Agent")
    st.write(
        "Paste in tasks and availability, then let the agent parse, schedule, "
        "validate, and explain the result."
    )

    with st.sidebar:
        st.header("Settings")
        model = st.text_input("Model", value="gpt-4o-mini")
        max_steps = st.slider("Max agent steps", min_value=3, max_value=12, value=8)

        st.markdown("---")
        st.subheader("Expected input format")
        st.code(
            "Task Name | Deadline | Duration Minutes | Priority(optional)\n"
            "Example:\n"
            "Finish NLP homework | Friday 5pm | 180 | high"
        )
        st.code(
            "Day: start-end\n"
            "Example:\n"
            "Wednesday: 6pm-10pm"
        )

    col1, col2 = st.columns(2)

    with col1:
        raw_tasks_text = st.text_area(
            "Tasks",
            value=DEFAULT_TASKS,
            height=220,
        )

    with col2:
        raw_availability_text = st.text_area(
            "Availability",
            value=DEFAULT_AVAILABILITY,
            height=220,
        )

    run_clicked = st.button("Generate Schedule", type="primary")

    if run_clicked:
        with st.spinner("Running agent..."):
            result = run_agent(
                raw_tasks_text=raw_tasks_text,
                raw_availability_text=raw_availability_text,
                model=model,
                max_steps=max_steps,
                verbose=False,
            )

        if not result["success"]:
            st.error("Agent run failed.")
            st.code(result.get("error", "Unknown error"))
            if "trace" in result:
                with st.expander("Trace"):
                    st.json(result["trace"])
            return

        st.success("Agent completed successfully.")

        st.subheader("Final Answer")
        st.write(result["final_message"])

        schedule_result = result.get("schedule_result")
        validation_result = result.get("validation_result")

        if schedule_result:
            st.subheader("Schedule")
            formatted_blocks = format_schedule_blocks(schedule_result)
            if formatted_blocks:
                st.dataframe(formatted_blocks, use_container_width=True)
            else:
                st.info("No scheduled blocks were created.")

            unscheduled = schedule_result.get("unscheduled_tasks", [])
            if unscheduled:
                st.subheader("Unscheduled Tasks")
                st.dataframe(unscheduled, use_container_width=True)

            with st.expander("Raw Schedule JSON"):
                st.json(schedule_result)

        if validation_result:
            st.subheader("Validation")
            st.write(f"Valid schedule: **{validation_result.get('is_valid')}**")

            issues = validation_result.get("issues", [])
            if issues:
                st.warning("Issues found:")
                for issue in issues:
                    st.write(f"- {issue}")
            else:
                st.info("No validation issues found.")

            completion = validation_result.get("completion", [])
            if completion:
                st.dataframe(completion, use_container_width=True)

            with st.expander("Raw Validation JSON"):
                st.json(validation_result)

        with st.expander("Agent Trace"):
            for item in result.get("trace", []):
                st.markdown(f"### Step {item['step']}")
                st.write("LLM response:")
                st.json(item["llm_response"])

                if "tool_name" in item:
                    st.write(f"Tool called: `{item['tool_name']}`")
                    st.write("Tool args:")
                    st.json(item.get("tool_args", {}))
                    st.write("Tool result:")
                    st.json(item.get("tool_result", {}))

        with st.expander("Full Result JSON"):
            st.code(json.dumps(result, indent=2, default=str), language="json")


if __name__ == "__main__":
    main()