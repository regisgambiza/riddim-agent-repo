"""
Thin client for llama-server's OpenAI-compatible /v1/chat/completions
endpoint, including tool-calling support. No business logic lives here --
this module only knows how to talk HTTP to the local model.
"""

import json
import sys
import time
from typing import Any

import requests

import config


class LlmClientError(RuntimeError):
    """Non-transient failure -- caller should treat this as a failed round."""
    pass


class LlmClient:
    def __init__(self, base_url: str, model_name: str = "local-model", timeout_s: int = 300):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_s = timeout_s

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = config.LLAMA_TEMPERATURE,
        max_tokens: int = config.LLAMA_MAX_TOKENS,
    ) -> dict[str, Any]:
        """
        Sends one chat-completions request, retrying transient server-side
        failures (e.g. llama-server's own JSON parse errors when a
        generation gets truncated by a full context window) a bounded
        number of times before raising. Returns the raw 'message' object
        from choices[0] (may include 'content' and/or 'tool_calls').
        """
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            # "auto" lets the model choose whether/which tool to call each
            # turn; the agent's own loop logic decides when to stop.
            payload["tool_choice"] = "auto"

        last_error: Exception | None = None
        for attempt in range(1, config.MAX_CHAT_RETRIES + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=self.timeout_s,
                )
            except requests.RequestException as e:
                last_error = e
            else:
                if resp.status_code == 200:
                    data = resp.json()
                    try:
                        return data["choices"][0]["message"]
                    except (KeyError, IndexError) as e:
                        last_error = LlmClientError(
                            f"unexpected response shape from llama-server: {data}"
                        )
                    else:
                        raise AssertionError("unreachable")
                elif resp.status_code >= 500:
                    # Transient server-side failure -- commonly llama-server's
                    # own "Failed to parse tool call arguments as JSON" when a
                    # generation got truncated by a full context window.
                    # Worth a retry: sampling is non-deterministic enough that
                    # a retry sometimes produces well-formed output.
                    last_error = LlmClientError(
                        f"llama-server returned {resp.status_code} (attempt "
                        f"{attempt}/{config.MAX_CHAT_RETRIES}): {resp.text[:1000]}"
                    )
                else:
                    # 4xx: our request was malformed -- retrying won't help.
                    raise LlmClientError(
                        f"llama-server returned {resp.status_code}: {resp.text[:2000]}"
                    )

            if attempt < config.MAX_CHAT_RETRIES:
                print(f"  [llm_client] {last_error} -- retrying...", file=sys.stderr)
                time.sleep(config.CHAT_RETRY_BACKOFF_S * attempt)

        raise LlmClientError(f"llama-server request failed after {config.MAX_CHAT_RETRIES} attempts: {last_error}")

    @staticmethod
    def parse_tool_call_args(tool_call: dict[str, Any]) -> dict[str, Any]:
        """
        tool_call["function"]["arguments"] is a JSON *string* per the
        OpenAI tool-calling wire format. Parses it defensively, since
        local models occasionally emit near-JSON.
        """
        raw = tool_call.get("function", {}).get("arguments", "{}")
        if isinstance(raw, dict):
            return raw
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise LlmClientError(f"model emitted invalid tool-call arguments JSON: {raw!r}") from e
