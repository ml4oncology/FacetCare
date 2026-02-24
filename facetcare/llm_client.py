from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .json_utils import parse_json_object_from_text, safe_json_loads
from .tools import ToolSpec


class LLMJsonClient:
    """Small wrapper around Chat Completions with JSON and tool-call recovery."""

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        if client is not None:
            self.client = client
        else:
            base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("MEDFLOW_OPENAI_BASE_URL")
            api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("MEDFLOW_OPENAI_API_KEY") or "sk-local"
            if base_url:
                self.client = OpenAI(base_url=base_url, api_key=api_key)
            else:
                self.client = OpenAI(api_key=api_key)
        self.model = model or os.getenv("OPENAI_MODEL") or os.getenv("MEDFLOW_MODEL") or "gpt-4o-mini"

    @staticmethod
    def _assistant_message_to_dict(msg: Any) -> Dict[str, Any]:
        out: Dict[str, Any] = {"role": "assistant", "content": getattr(msg, "content", None)}
        tc = getattr(msg, "tool_calls", None)
        if tc:
            out["tool_calls"] = [
                {
                    "id": t.id,
                    "type": t.type,
                    "function": {"name": t.function.name, "arguments": t.function.arguments},
                }
                for t in tc
            ]
        fc = getattr(msg, "function_call", None)
        if fc:
            out["function_call"] = {"name": fc.name, "arguments": fc.arguments}
        return out

    def _repair_json_via_model(self, *, system: str, user: str, bad_text: str, temperature: float) -> Dict[str, Any]:
        repair_system = (
            "You repair malformed JSON. Return exactly ONE valid JSON object. "
            "Do not add commentary. Preserve meaning and keys as much as possible."
        )
        repair_user = (
            "Original system prompt (for context):\n"
            f"{system}\n\n"
            "Original user prompt (for context):\n"
            f"{user[:4000]}\n\n"
            "Malformed model output to repair:\n"
            f"{bad_text[:8000]}"
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": repair_system}, {"role": "user", "content": repair_user}],
            temperature=temperature,
        )
        txt = resp.choices[0].message.content or ""
        return parse_json_object_from_text(txt)

    def json_object_no_tools(self, *, system: str, user: str, temperature: float = 0.0) -> Dict[str, Any]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
        )
        txt = resp.choices[0].message.content or ""
        try:
            return parse_json_object_from_text(txt)
        except Exception:
            return self._repair_json_via_model(system=system, user=user, bad_text=txt, temperature=0.0)

    def json_object_with_tools(
        self,
        *,
        system: str,
        user: str,
        tool_specs: List[ToolSpec],
        max_rounds: int = 10,
        temperature: float = 0.0,
        prefer_new_tools: bool = True,
    ) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last_text = ""
        tool_by_name = {t.name: t for t in tool_specs}

        for _ in range(max_rounds):
            if prefer_new_tools:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=[t.chat_tool for t in tool_specs],
                    temperature=temperature,
                )
            else:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    functions=[t.legacy_function for t in tool_specs],
                    temperature=temperature,
                )

            msg = resp.choices[0].message
            messages.append(self._assistant_message_to_dict(msg))

            tool_calls = getattr(msg, "tool_calls", None) or []
            function_call = getattr(msg, "function_call", None)

            if tool_calls:
                for tc in tool_calls:
                    call_id = tc.id
                    fname = tc.function.name
                    args = safe_json_loads(tc.function.arguments) or {}
                    spec = tool_by_name.get(fname)
                    if not spec:
                        result_payload = {"ok": False, "error": f"Unknown tool: {fname}", "args": args}
                    else:
                        try:
                            result_payload = {"ok": True, "result": spec.fn(**args)}
                        except Exception as e:
                            result_payload = {"ok": False, "error": str(e), "args": args}
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(result_payload, ensure_ascii=False)})
                continue

            if function_call:
                fname = function_call.name
                args = safe_json_loads(function_call.arguments) or {}
                spec = tool_by_name.get(fname)
                if not spec:
                    result_payload = {"ok": False, "error": f"Unknown function: {fname}", "args": args}
                else:
                    try:
                        result_payload = {"ok": True, "result": spec.fn(**args)}
                    except Exception as e:
                        result_payload = {"ok": False, "error": str(e), "args": args}
                messages.append({"role": "function", "name": fname, "content": json.dumps(result_payload, ensure_ascii=False)})
                continue

            last_text = getattr(msg, "content", "") or ""
            if not last_text.strip():
                continue
            try:
                return parse_json_object_from_text(last_text)
            except Exception:
                # Keep the tool transcript and ask the model to re-emit strict JSON.
                messages.append(
                    {
                        "role": "user",
                        "content": "Your last response was malformed JSON. Re-output the same answer as one valid JSON object only.",
                    }
                )
                continue

        if last_text.strip():
            return self._repair_json_via_model(system=system, user=user, bad_text=last_text, temperature=0.0)
        raise RuntimeError(f"Exceeded max_rounds={max_rounds}. Last raw text: {last_text[:500]!r}")
