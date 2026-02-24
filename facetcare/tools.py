from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

_DOB_RE = re.compile(r"\bDOB\s*:\s*(\d{4}-\d{2}-\d{2})\b")
_NOTE_TAG_RE = re.compile(r"<note_[^>]*_(\d{4}-\d{2}-\d{2})>")


def get_patient_age(dob: str, as_of: str) -> int:
    d0 = dt.date.fromisoformat(dob)
    d1 = dt.date.fromisoformat(as_of)
    years = d1.year - d0.year - ((d1.month, d1.day) < (d0.month, d0.day))
    return int(years)


def extract_dob_and_most_recent_note_date(longitudinal_notes: str) -> Tuple[Optional[str], Optional[str]]:
    dob = None
    m = _DOB_RE.search(longitudinal_notes or "")
    if m:
        dob = m.group(1)

    dates = _NOTE_TAG_RE.findall(longitudinal_notes or "")
    most_recent = None
    if dates:
        try:
            most_recent = max(dates)
        except Exception:
            most_recent = dates[-1]
    return dob, most_recent


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]
    fn: Callable[..., Any]

    @property
    def chat_tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @property
    def legacy_function(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


def default_tool_specs() -> List[ToolSpec]:
    params = {
        "type": "object",
        "properties": {
            "dob": {"type": "string", "description": "DOB in YYYY-MM-DD format."},
            "as_of": {"type": "string", "description": "As-of date in YYYY-MM-DD format (use most recent note date)."},
        },
        "required": ["dob", "as_of"],
        "additionalProperties": False,
    }
    return [
        ToolSpec(
            name="get_patient_age",
            description="Compute age in years given DOB and an as-of date.",
            parameters=params,
            fn=get_patient_age,
        )
    ]
