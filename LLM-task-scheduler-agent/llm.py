# model/API wrapper

from __future__ import annotations

from typing import List, Dict, Any

from openai import OpenAI

from dotenv import load_dotenv
import os

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "no valid open api key found"
        )
    return OpenAI(api_key=api_key)


def call_llm(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
) -> str:
    """
    Send chat messages to the LLM and return plain text output.
    """
    client = get_client()

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("LLM returned empty content.")

    return content