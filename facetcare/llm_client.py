from __future__ import annotations

import json
import os
import sys
import time
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
        self.debug = str(os.getenv("FACETCARE_DEBUG_LLM", "")).strip().lower() in {"1", "true", "yes", "on"}

    def _debug_log(self, message: str) -> None:
        if self.debug:
            print(f"[facetcare.llm] {message}", file=sys.stderr, flush=True)

    def _preview_messages(self, messages: List[Dict[str, Any]], limit: int = 1500) -> str:
        parts: List[str] = []
        for i, msg in enumerate(messages):
            role = str(msg.get("role", "?"))
            content = msg.get("content", "")
            if isinstance(content, list):
                content = json.dumps(content, ensure_ascii=False)
            content = str(content or "").replace("\n", "\\n")
            parts.append(f"{i}:{role}:{content[:400]}")
        joined = " | ".join(parts)
        return joined[:limit]

    def _chat_create(self, *, messages: List[Dict[str, Any]], temperature: float, tools: Optional[List[Dict[str, Any]]] = None, functions: Optional[List[Dict[str, Any]]] = None) -> Any:
        started = time.time()
        self._debug_log(
            f"request start model={self.model!r} temp={temperature} messages={len(messages)} "
            f"tools={len(tools) if tools else 0} functions={len(functions) if functions else 0} "
            f"preview={self._preview_messages(messages)!r}"
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                functions=functions,
                temperature=temperature,
            )
        except Exception as e:
            elapsed = time.time() - started
            self._debug_log(f"request error after {elapsed:.2f}s: {type(e).__name__}: {e}")
            raise

        elapsed = time.time() - started
        try:
            content = resp.choices[0].message.content
        except Exception:
            content = None
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False)
        preview = str(content or "").replace("\n", "\\n")[:600]
        self._debug_log(f"request ok after {elapsed:.2f}s raw_preview={preview!r}")
        return resp


    def _medgemma_safe_messages(self, *, system: str, user: str) -> List[Dict[str, str]]:
        """
        Many MedGemma-compatible OpenAI endpoints reject a top-level system role and expect
        alternating user/assistant turns. Preserve instructions by folding the system text
        into a single user message.
        """
        sys_txt = (system or "").strip()
        usr_txt = (user or "").strip()
        if sys_txt:
            merged = f"[INSTRUCTIONS]\n{sys_txt}\n\n[REQUEST]\n{usr_txt}"
        else:
            merged = usr_txt
        return [{"role": "user", "content": merged}]

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
        resp = self._chat_create(
            messages=self._medgemma_safe_messages(system=repair_system, user=repair_user),
            temperature=temperature,
        )
        txt = resp.choices[0].message.content or ""
        return parse_json_object_from_text(txt)

    def json_object_no_tools(self, *, system: str, user: str, temperature: float = 0.0) -> Dict[str, Any]:
        resp = self._chat_create(
            messages=self._medgemma_safe_messages(system=system, user=user),
            temperature=temperature,
        )
        txt = resp.choices[0].message.content or ""
        try:
            return parse_json_object_from_text(txt)
        except Exception:
            self._debug_log(f"initial JSON parse failed, invoking repair. raw_text={str(txt)[:800]!r}")
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
        messages: List[Dict[str, Any]] = [dict(m) for m in self._medgemma_safe_messages(system=system, user=user)]
        last_text = ""
        tool_by_name = {t.name: t for t in tool_specs}

        for _ in range(max_rounds):
            if prefer_new_tools:
                resp = self._chat_create(
                    messages=messages,
                    tools=[t.chat_tool for t in tool_specs],
                    temperature=temperature,
                )
            else:
                resp = self._chat_create(
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
                self._debug_log(f"tool round produced malformed JSON, requesting strict re-output. raw_text={last_text[:800]!r}")
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
