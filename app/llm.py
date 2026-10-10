"""OpenAI-compatible chat streaming and JSON extraction, shared by single runs and workflows."""

import json
import re
from typing import AsyncIterator, Optional

import httpx

DEFAULT_SYSTEM_PROMPT = """You are an image classification assistant.
Look at the image and respond ONLY with a JSON object in this exact shape:
{
  "category": "<single best category, e.g. landscape, portrait, food, document, screenshot, animal, vehicle, architecture, other>",
  "tags": ["<short descriptive tag>", "..."],
  "confidence": <number between 0 and 1>,
  "description": "<one sentence description of the image>"
}
Do not include any text outside the JSON."""

DEFAULT_USER_PROMPT = "Classify this image."


def headers(api_key: str) -> dict:
    h = {"Content-Type": "application/json"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def chat_payload(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_data_url: Optional[str],
    temperature: float,
    max_tokens: int,
    json_mode: bool,
) -> dict:
    content: list = [{"type": "text", "text": user_prompt}]
    if image_data_url:
        content.append({"type": "image_url", "image_url": {"url": image_data_url}})
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if model:
        payload["model"] = model
    if json_mode:
        # A schema accepting any object. LM Studio rejects {"type": "json_object"}, but it,
        # llama.cpp and OpenAI all accept json_schema, which constrains output to valid JSON.
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": {"type": "object"}},
        }
    return payload


async def stream_chat(base_url: str, api_key: str, payload: dict, timeout: float) -> AsyncIterator[dict]:
    """Yield token / reasoning / usage / timings / error events from a streaming chat completion."""
    url = base_url.rstrip("/") + "/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, json=payload, headers=headers(api_key)) as r:
                if r.status_code >= 400:
                    body = (await r.aread()).decode(errors="replace")
                    yield {"type": "error", "message": f"HTTP {r.status_code}: {body}"}
                    return
                got_content, thought, finish = False, False, None
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            got_content = True
                            yield {"type": "token", "text": delta["content"]}
                        # llama.cpp / LM Studio send reasoning_content; Ollama and vLLM send reasoning
                        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                        if reasoning:
                            thought = True
                            yield {"type": "reasoning", "text": reasoning}
                        finish = choice.get("finish_reason") or finish
                    if chunk.get("usage"):
                        yield {"type": "usage", "usage": chunk["usage"]}
                    if chunk.get("timings"):
                        yield {"type": "timings", "timings": chunk["timings"]}
                if finish == "length" and not got_content:
                    spent = "thinking" if thought else "before answering"
                    yield {
                        "type": "error",
                        "message": f"Model used all {payload.get('max_tokens')} max tokens {spent} and returned "
                        "no answer. Raise max tokens, or use a non-thinking model.",
                    }
    except Exception as e:  # noqa: BLE001
        yield {"type": "error", "message": f"{type(e).__name__}: {e}"}


def extract_json(text: str):
    """Pull the first JSON value out of model output (mirrors extractJson in app.js)."""
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, flags=re.S | re.I)
    if fence:
        t = fence.group(1).strip()
    try:
        return json.loads(t)
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch in "{[":
            try:
                return decoder.raw_decode(t[i:])[0]
            except ValueError:
                continue
    raise ValueError("No JSON object found in output.")
