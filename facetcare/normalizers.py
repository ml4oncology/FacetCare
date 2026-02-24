from __future__ import annotations

from typing import Any, Dict, List, Optional


def ensure_list_str(x: Any) -> List[str]:
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    if x is None:
        return []
    s = str(x).strip()
    return [s] if s else []


def first_non_empty(*vals: Any, default: str = "") -> str:
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str) and v.strip():
            return v.strip()
        if not isinstance(v, str):
            sv = str(v).strip()
            if sv:
                return sv
    return default


def coerce_probability(x: Any, default: float = 0.01) -> float:
    try:
        p = float(x)
    except Exception:
        p = float(default)
    if p != p:
        p = float(default)
    if p > 1.0 and p <= 100.0:
        p = p / 100.0
    return max(0.0, min(1.0, p))


def coerce_int(x: Any, default: int, *, lo: Optional[int] = None, hi: Optional[int] = None) -> int:
    try:
        v = int(x)
    except Exception:
        v = int(default)
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def normalize_candidate_pool_strategy(x: Any) -> str:
    s = str(x or "").strip().lower()
    aliases = {
        "all": "all",
        "recent": "recent_notes_only",
        "recent_notes": "recent_notes_only",
        "recent_notes_only": "recent_notes_only",
        "recent_visits": "recent_notes_only",
        "keyword": "keyword_prefilter",
        "keywords": "keyword_prefilter",
        "keyword_prefilter": "keyword_prefilter",
        # common model mistakes; treat as broad pool and let selection handle priority
        "high_risk": "all",
        "highrisk": "all",
    }
    return aliases.get(s, "all")


def normalize_selection_method(x: Any, mode: Any = None) -> str:
    raw = str(x or mode or "").strip().lower()
    aliases = {
        "topk": "top_k",
        "top_k": "top_k",
        "ranked": "top_k",
        "threshold": "threshold",
        "threshold_then_top_k": "threshold_then_top_k",
        "first": "first_k",
        "first_k": "first_k",
        "firstk": "first_k",
    }
    return aliases.get(raw, "top_k")


def normalize_workflow_payload(obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {}

    if "workflow" in obj and isinstance(obj["workflow"], dict):
        obj = obj["workflow"]

    out: Dict[str, Any] = {}
    out["clinic_name"] = first_non_empty(obj.get("clinic_name"), obj.get("name"), default="Unknown clinic")
    out["clinic_type"] = first_non_empty(obj.get("clinic_type"), default="Clinic")

    nc = obj.get("note_capture") or {}
    if isinstance(nc, dict):
        out["note_capture"] = {
            "source": str(first_non_empty(nc.get("source"), nc.get("type"), default="EMR")),
            "format": str(first_non_empty(nc.get("format"), default="free_text")),
            "cadence": str(first_non_empty(nc.get("cadence"), default="encounter_based")),
        }
    else:
        out["note_capture"] = {"source": "EMR", "format": "free_text", "cadence": "encounter_based"}

    rc = obj.get("review_cadence") or {}
    if isinstance(rc, dict):
        out["review_cadence"] = {
            "who_reviews": str(first_non_empty(rc.get("who_reviews"), default="Clinical team")),
            "frequency": str(first_non_empty(rc.get("frequency"), rc.get("type"), default="weekly")),
            "trigger": str(first_non_empty(rc.get("trigger"), default="scheduled_review")),
        }
    else:
        out["review_cadence"] = {"who_reviews": "Clinical team", "frequency": "weekly", "trigger": "scheduled_review"}

    rp = obj.get("referral_pathway") or {}
    if isinstance(rp, dict):
        out["referral_pathway"] = {
            "internal": str(first_non_empty(rp.get("internal"), default="Internal clinician triage")),
            "external": str(first_non_empty(rp.get("external"), default="Refer to external service as needed")),
            "urgent_flags": ensure_list_str(rp.get("urgent_flags") or ["red flag symptoms", "abnormal imaging", "critical labs"]),
        }
    else:
        out["referral_pathway"] = {
            "internal": "Internal clinician triage",
            "external": "Refer to external service as needed",
            "urgent_flags": ["red flag symptoms", "abnormal imaging", "critical labs"],
        }

    out["staffing_constraints"] = ensure_list_str(obj.get("staffing_constraints") or ["Limited clinician review time"]) 
    out["goals"] = ensure_list_str(obj.get("goals") or ["Surface actionable patients for review"]) 

    gr = obj.get("guardrails") or {}
    if isinstance(gr, dict):
        out["guardrails"] = {
            "privacy": str(first_non_empty(gr.get("privacy"), default="Use minimum necessary PHI")),
            "safety": str(first_non_empty(gr.get("safety"), default="Decision support only, no diagnosis")),
            "logging": str(first_non_empty(gr.get("logging"), default="No PHI in logs")),
        }
    else:
        out["guardrails"] = {
            "privacy": "Use minimum necessary PHI",
            "safety": "Decision support only, no diagnosis",
            "logging": "No PHI in logs",
        }
    return out


def normalize_risk_payload(
    obj: Dict[str, Any],
    *,
    patient_id: str,
    target_condition: str,
    horizon_months: int,
    risk_level: str,
) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        obj = {}
    out: Dict[str, Any] = {
        "patient_id": patient_id,
        "target_condition": target_condition,
        "horizon_months": horizon_months,
        "risk_probability": coerce_probability(obj.get("risk_probability", 0.01), default=0.01),
        "risk_level": first_non_empty(obj.get("risk_level"), default=risk_level),
        "key_risk_factors": ensure_list_str(obj.get("key_risk_factors")),
        "key_protective_factors": ensure_list_str(obj.get("key_protective_factors")),
        "recommended_next_steps": ensure_list_str(obj.get("recommended_next_steps")),
        "safety_notes": ensure_list_str(obj.get("safety_notes")),
    }
    if not out["key_risk_factors"]:
        out["key_risk_factors"] = ["Longitudinal notes contain symptoms or patterns that merit review"]
    if not out["key_protective_factors"]:
        out["key_protective_factors"] = ["No clear protective factors captured in available notes"]
    if not out["recommended_next_steps"]:
        out["recommended_next_steps"] = [
            "Review symptom trajectory and recent testing",
            "Consider targeted follow-up based on clinic workflow",
            "Document safety netting and follow-up timeline",
        ]
    if not out["safety_notes"]:
        out["safety_notes"] = ["Decision support only. Use clinician judgment."]
    return out


def normalize_clinician_summary_payload(
    obj: Dict[str, Any],
    *,
    patient_id: str,
    target_condition: str,
    horizon_months: int,
) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        obj = {}
    out: Dict[str, Any] = {
        "patient_id": patient_id,
        "target_condition": target_condition,
        "horizon_months": horizon_months,
        "summary_for_chart": first_non_empty(
            obj.get("summary_for_chart"), obj.get("summary"), default=f"{patient_id}: Clinical review summary generated."
        ),
        "suggested_orders": ensure_list_str(obj.get("suggested_orders") or obj.get("orders")),
        "suggested_referrals": ensure_list_str(obj.get("suggested_referrals") or obj.get("referrals")),
        "safety_netting": ensure_list_str(obj.get("safety_netting") or obj.get("follow_up")),
    }
    if not out["suggested_orders"]:
        out["suggested_orders"] = ["Review recent labs and imaging if available"]
    if not out["suggested_referrals"]:
        out["suggested_referrals"] = ["Refer or discuss with specialty service if clinically indicated"]
    if not out["safety_netting"]:
        out["safety_netting"] = ["Provide return precautions and document follow-up interval"]
    return out
