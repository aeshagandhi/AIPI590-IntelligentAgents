# parse structured LLM outputs
# takes raw llm output and converts into structure that can be used by agent loop to decide what to do next (call tool, or finish with final answer)
# convert LLM decision into executable instruction



from __future__ import annotations
import json
from typing import Any, Dict

ALLOWED_TYPES = {"tool_call", "final"}

def parse_llm_response(raw_text: str) -> Dict[str, Any]:
    
    """
    Parse LLM JSON response into structured dict and validate the response
    Expected format:
    Tool call:
    {
        "type": "tool_call",
        "tool": "parse_tasks",
        "args": {...}
    }
    
    Final:
    {
        "type": "final",
        "message": "..."
    }
    """
    
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid LLM response, not valid JSON: {exc}")
    if not isinstance(parsed, dict):
        raise ValueError(f"LLM output should be a JSON object/dict, got {type(parsed)}")
    
    response_type = parsed.get("type")
    if response_type not in ALLOWED_TYPES:
        raise ValueError(
            f"Invalid response type '{response_type}'."
            f"Expected one of: {ALLOWED_TYPES}"
        )
        
    if response_type == "tool_call":
        tool = parsed.get("tool")
        args = parsed.get("args", {})
        if not isinstance(tool, str) or not tool.strip():
            raise ValueError("Tool call must include a non-empty 'tool' string.")

        if not isinstance(args, dict):
            raise ValueError("'args' must be a JSON object for tool calls.")

        return {
            "type": "tool_call",
            "tool": tool,
            "args": args,
        }
        
    message = parsed.get("message")
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Final response cannot be empty")
    
    return {
        "type": "final",
        "message": message
    }