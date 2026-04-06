from __future__ import annotations

import json
import re
from typing import Any, Dict


def _strip_code_fences(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def _strip_llm_channel_wrappers(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return s

    # Some GPT-OSS / llama.cpp configurations emit role/channel control tokens
    # around the actual assistant payload. Prefer the last assistant/final message body.
    message_matches = list(re.finditer(r"<\|message\|>", s))
    if not message_matches:
        return s

    last = message_matches[-1]
    body = s[last.end():]
    end_marker = body.find("<|end|>")
    if end_marker >= 0:
        body = body[:end_marker]
    return body.strip() or s


def _extract_balanced_block(s: str, start_idx: int) -> tuple[str, int]:
    opener = s[start_idx]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_str = False
    esc = False
    for i in range(start_idx, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return s[start_idx : i + 1], i + 1
    raise RuntimeError("Unbalanced braces while extracting block")


def coerce_first_json(text: str) -> str:
    if not text or not text.strip():
        raise RuntimeError("Empty model output")

    s = _strip_code_fences(text)
    # Prefer object, but allow array extraction for internal repair use.
    start_obj = s.find("{")
    start_arr = s.find("[")
    starts = [x for x in (start_obj, start_arr) if x >= 0]
    if not starts:
        raise RuntimeError(f"No JSON object found in output: {s[:500]!r}")
    start = min(starts)
    try:
        block, _ = _extract_balanced_block(s, start)
        if block.lstrip().startswith("{"):
            return block
        # If first JSON is array, try to find first object inside it.
        m = re.search(r"\{", block)
        if m:
            inner, _ = _extract_balanced_block(block, m.start())
            return inner
    except Exception:
        pass
    raise RuntimeError(f"Unbalanced braces; could not extract JSON from: {s[:2000]!r}")


def _unwrap_quoted_json_fragments(s: str) -> str:
    """Fix patterns like ... ,"{...}", ... produced by some models.

    This is not a full parser. It specifically detects a quote immediately
    wrapping a JSON object/array in a place where a value is expected, then
    removes the outer quotes.
    """
    out = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch != '"':
            out.append(ch)
            i += 1
            continue

        # check if next non-space begins with { or [
        j = i + 1
        while j < n and s[j].isspace():
            j += 1
        if j < n and s[j] in "[{":
            # attempt to extract balanced block starting at j
            try:
                block, end_idx = _extract_balanced_block(s, j)
                k = end_idx
                while k < n and s[k].isspace():
                    k += 1
                if k < n and s[k] == '"':
                    # remove only the wrapping quotes
                    out.append(block)
                    i = k + 1
                    continue
            except Exception:
                pass

        # normal string token fallback (best-effort copy)
        out.append('"')
        i += 1
        esc = False
        while i < n:
            out.append(s[i])
            if esc:
                esc = False
            elif s[i] == "\\":
                esc = True
            elif s[i] == '"':
                i += 1
                break
            i += 1
    return "".join(out)


def _light_json_repairs(s: str) -> str:
    s = _strip_llm_channel_wrappers(_strip_code_fences(s))
    # common smart quotes
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    # unwrap accidental quoted JSON objects in arrays or fields
    s = _unwrap_quoted_json_fragments(s)
    # remove trailing commas before closing braces/brackets
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # normalize booleans/null casing if model emitted Python-ish values
    s = re.sub(r"\bNone\b", "null", s)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    return s


def safe_json_loads(s: Any) -> Any:
    if s is None:
        return None
    if isinstance(s, (dict, list, int, float, bool)):
        return s
    if isinstance(s, str):
        ss = s.strip()
        if not ss:
            return None
        for cand in (ss, _light_json_repairs(ss)):
            try:
                return json.loads(cand)
            except Exception:
                continue
        try:
            return json.loads(coerce_first_json(_light_json_repairs(ss)))
        except Exception:
            return None
    return None


def parse_json_object_from_text(txt: str) -> Dict[str, Any]:
    if not txt or not txt.strip():
        raise RuntimeError("Empty model output text")

    last_err: str = "unknown"
    candidates = [txt, _light_json_repairs(txt)]

    # Optional third-party repair if installed
    try:
        from json_repair import repair_json  # type: ignore

        candidates.append(repair_json(txt, return_objects=False))
    except Exception:
        pass

    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
            if isinstance(obj, list):
                # If model wrapped the object in a list, take first object.
                for item in obj:
                    if isinstance(item, dict):
                        return item
            last_err = f"Expected JSON object, got {type(obj)}"
        except Exception as e:
            last_err = str(e)
            try:
                extracted = coerce_first_json(cand)
                obj = json.loads(extracted)
                if isinstance(obj, dict):
                    return obj
                last_err = f"Expected JSON object, got {type(obj)}"
            except Exception as e2:
                last_err = str(e2)
                continue

    raise RuntimeError(f"Could not parse JSON object from model output. Last error: {last_err}")
