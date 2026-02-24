from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .llm_client import LLMJsonClient
from . import prompts
from .normalizers import (
    coerce_int,
    coerce_probability,
    ensure_list_str,
    first_non_empty,
    normalize_candidate_pool_strategy,
    normalize_selection_method,
)
from .schemas import CandidatePoolConfig, ClinicPlanSchema, ProgramConstraints, RiskLevelPolicy, SelectionConfig, TaskSpec


TASK_CATALOG: Dict[str, Dict[str, Any]] = {
    "intake_workflow": {
        "depends_on": [],
        "synonyms_pos": ["intake", "clinic profile", "workflow capture", "setup"],
        "synonyms_neg": [],
    },
    "risk_assessment": {
        "depends_on": ["intake_workflow"],
        "synonyms_pos": ["risk", "high risk", "screening triage", "cancer risk"],
        "synonyms_neg": ["no risk scoring", "no risk assessment"],
    },
    "queue_prioritization": {
        "depends_on": ["intake_workflow"],
        "synonyms_pos": ["priority queue", "queue prioritization", "triage queue", "shortlist"],
        "synonyms_neg": ["no prioritization"],
    },
    "clinician_summary": {
        "depends_on": ["intake_workflow"],
        "synonyms_pos": ["clinician summary", "chart summary", "monthly summary", "review summary"],
        "synonyms_neg": ["no summary"],
    },
    "admin_referral": {
        "depends_on": ["intake_workflow"],
        "synonyms_pos": ["referral", "referral package", "specialty referral"],
        "synonyms_neg": ["no referral"],
    },
    "referral_letter": {
        "depends_on": ["admin_referral"],
        "synonyms_pos": ["referral letter", "consult letter"],
        "synonyms_neg": ["no referral letter"],
    },
    "referral_intake_checklist": {
        "depends_on": ["intake_workflow"],
        "synonyms_pos": ["intake checklist", "referral intake", "pre-referral checklist"],
        "synonyms_neg": [],
    },
    "followup_gap_detection": {
        "depends_on": ["intake_workflow"],
        "synonyms_pos": ["follow-up gap", "missed follow-up", "care gaps"],
        "synonyms_neg": [],
    },
    "lab_trend_summary": {
        "depends_on": ["intake_workflow"],
        "synonyms_pos": ["lab trend", "labs summary", "yearly lab summary", "results summary"],
        "synonyms_neg": [],
    },
    "care_plan_reconciliation": {
        "depends_on": ["intake_workflow"],
        "synonyms_pos": ["care plan reconciliation", "reconcile plan", "care plan review"],
        "synonyms_neg": [],
    },
    "patient_instructions": {
        "depends_on": ["clinician_summary"],
        "synonyms_pos": ["patient instructions", "patient note", "patient message", "after visit"],
        "synonyms_neg": [],
    },
    "results_summary": {
        "depends_on": ["intake_workflow"],
        "synonyms_pos": ["results summary", "imaging summary", "test summary"],
        "synonyms_neg": [],
    },
    "transcription": {
        "depends_on": [],
        "synonyms_pos": ["transcription", "dictation", "audio"],
        "synonyms_neg": [],
    },
    "differential_diagnosis": {
        "depends_on": ["clinician_summary"],
        "synonyms_pos": ["differential", "what else could this be"],
        "synonyms_neg": ["no differential"],
    },
    "guideline_comparison": {
        "depends_on": ["clinician_summary"],
        "synonyms_pos": ["guideline", "compare guidelines"],
        "synonyms_neg": ["no guideline"],
    },
}

_RE_TOPK = re.compile(r"\b(?:top|review|shortlist)\s*(\d{1,3})\b", re.IGNORECASE)
_RE_CADENCE = {
    "daily": re.compile(r"\bdaily\b", re.IGNORECASE),
    "weekly": re.compile(r"\bweekly\b", re.IGNORECASE),
    "monthly": re.compile(r"\bmonthly\b", re.IGNORECASE),
    "yearly": re.compile(r"\b(yearly|annual|annually)\b", re.IGNORECASE),
    "per_visit": re.compile(r"\bper[\s-]?visit\b|\beach visit\b", re.IGNORECASE),
}
_RE_DEDUP_DAYS = re.compile(r"\b(?:dedup|do not repeat|no repeats|repeat)\b.*?\b(\d{1,4})\s*days?\b", re.IGNORECASE)
_RE_DEDUP_MONTHS = re.compile(r"\b(?:dedup|do not repeat|no repeats|repeat)\b.*?\b(\d{1,3})\s*months?\b", re.IGNORECASE)
_RE_THRESHOLD = re.compile(r"\bthreshold\b.*?\b(0?\.\d+|\d+\.?\d*)\b", re.IGNORECASE)


def _text_has_any(text: str, phrases: List[str]) -> bool:
    t = (text or "").lower()
    return any(p.lower() in t for p in phrases)


def _infer_enabled_tasks_from_text(text: str) -> Dict[str, bool]:
    enabled: Dict[str, bool] = {}
    for task_name, meta in TASK_CATALOG.items():
        pos_hit = _text_has_any(text, meta.get("synonyms_pos", []))
        neg_hit = _text_has_any(text, meta.get("synonyms_neg", []))
        enabled[task_name] = bool(pos_hit and not neg_hit)
    enabled["intake_workflow"] = True
    return enabled


def _infer_review_limit(text: str, default: int = 5) -> int:
    m = _RE_TOPK.search(text or "")
    if not m:
        return default
    return coerce_int(m.group(1), default, lo=1, hi=200)


def _infer_cadence(text: str, default: str = "weekly") -> str:
    for k, rx in _RE_CADENCE.items():
        if rx.search(text or ""):
            return k
    return default


def _infer_dedup_days(text: str, default: int = 180) -> int:
    m = _RE_DEDUP_DAYS.search(text or "")
    if m:
        return coerce_int(m.group(1), default, lo=0, hi=3650)
    m2 = _RE_DEDUP_MONTHS.search(text or "")
    if m2:
        return coerce_int(m2.group(1), default // 30 if default else 6, lo=0, hi=120) * 30
    return default


def _infer_threshold(text: str) -> Optional[float]:
    m = _RE_THRESHOLD.search(text or "")
    if not m:
        return None
    return coerce_probability(m.group(1), default=0.1)


def _infer_default_source_task(enabled_by_text: Dict[str, bool], tasks: List[TaskSpec]) -> Optional[str]:
    enabled_names = {t.name for t in tasks if t.enabled}
    if "risk_assessment" in enabled_names:
        return "risk_assessment"
    if "queue_prioritization" in enabled_names:
        return "queue_prioritization"
    if enabled_by_text.get("followup_gap_detection"):
        return "followup_gap_detection"
    return None


def _normalize_task_list(raw_tasks: Any, clinic_description: str) -> List[TaskSpec]:
    enabled_by_text = _infer_enabled_tasks_from_text(clinic_description)
    tasks: List[TaskSpec] = []

    if isinstance(raw_tasks, list):
        for t in raw_tasks:
            if not isinstance(t, dict):
                continue
            name = str(t.get("name", "")).strip()
            if name not in TASK_CATALOG:
                continue
            enabled = bool(t.get("enabled", True))
            if enabled_by_text.get(name, False):
                enabled = True
            if _text_has_any(clinic_description, TASK_CATALOG[name].get("synonyms_neg", [])):
                enabled = False
            params = t.get("params") if isinstance(t.get("params"), dict) else {}
            tasks.append(TaskSpec(name=name, enabled=enabled, params=params, depends_on=list(TASK_CATALOG[name]["depends_on"])))

    if not tasks:
        for name in TASK_CATALOG:
            tasks.append(
                TaskSpec(name=name, enabled=bool(enabled_by_text.get(name, False)), params={}, depends_on=list(TASK_CATALOG[name]["depends_on"]))
            )

    # always include intake; do not force risk workflows
    have = {t.name for t in tasks}
    if "intake_workflow" not in have:
        tasks.append(TaskSpec(name="intake_workflow", enabled=True, params={}, depends_on=[]))
    else:
        for t in tasks:
            if t.name == "intake_workflow":
                t.enabled = True

    # de-duplicate and refresh dependencies while preserving first appearance
    deduped: Dict[str, TaskSpec] = {}
    ordered: List[TaskSpec] = []
    for t in tasks:
        if t.name in deduped:
            # merge latest enabled/params into existing task without moving its order
            deduped[t.name].enabled = t.enabled
            deduped[t.name].params = t.params
            continue
        t.depends_on = list(TASK_CATALOG[t.name]["depends_on"])
        deduped[t.name] = t
        ordered.append(t)
    return ordered


def normalize_plan_payload(obj: Dict[str, Any], clinic_description: str) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        obj = {}
    for k in ("plan", "clinic_plan", "clinicPlan", "output"):
        if k in obj and isinstance(obj[k], dict):
            obj = obj[k]
            break

    tasks = _normalize_task_list(obj.get("tasks"), clinic_description)
    enabled_by_text = _infer_enabled_tasks_from_text(clinic_description)

    out: Dict[str, Any] = {
        "plan_version": str(first_non_empty(obj.get("plan_version"), default="2.0")),
        "clinic_description": clinic_description,
    }

    tc = obj.get("target_condition")
    out["target_condition"] = str(tc).strip() if tc is not None and str(tc).strip() else None

    hm = obj.get("horizon_months")
    try:
        out["horizon_months"] = int(hm) if hm is not None and str(hm).strip() else None
    except Exception:
        out["horizon_months"] = None
    if out["horizon_months"] is not None:
        out["horizon_months"] = max(1, min(120, int(out["horizon_months"])))

    c = obj.get("constraints") if isinstance(obj.get("constraints"), dict) else {}
    constraints = ProgramConstraints()
    constraints.cadence = str(first_non_empty(c.get("cadence"), default=constraints.cadence))
    if constraints.cadence not in {"daily", "weekly", "per_visit", "monthly", "yearly"}:
        constraints.cadence = "weekly"
    constraints.review_limit = coerce_int(c.get("review_limit"), constraints.review_limit, lo=1, hi=200)
    constraints.dedup_days = coerce_int(c.get("dedup_days"), constraints.dedup_days, lo=0, hi=3650)

    cp = c.get("candidate_pool") if isinstance(c.get("candidate_pool"), dict) else {}
    constraints.candidate_pool = CandidatePoolConfig(
        strategy=normalize_candidate_pool_strategy(cp.get("strategy", constraints.candidate_pool.strategy)),
        max_candidates=coerce_int(cp.get("max_candidates"), 0, lo=1, hi=5000) if cp.get("max_candidates") not in (None, "", "null") else None,
        keywords=ensure_list_str(cp.get("keywords", constraints.candidate_pool.keywords)),
    )

    sel = c.get("selection") if isinstance(c.get("selection"), dict) else {}
    method = normalize_selection_method(sel.get("method"), sel.get("mode"))
    threshold = sel.get("threshold")
    threshold_val = None if threshold in (None, "", "null") else coerce_probability(threshold, default=0.1)
    source_task = sel.get("source_task") if isinstance(sel.get("source_task"), str) else None
    constraints.selection = SelectionConfig(
        source_task=source_task,
        method=method,
        k=coerce_int(sel.get("k"), constraints.review_limit, lo=1, hi=200),
        threshold=threshold_val,
    )

    rlp = c.get("risk_level_policy") if isinstance(c.get("risk_level_policy"), dict) else {}
    constraints.risk_level_policy = RiskLevelPolicy(
        low_lt=coerce_probability(rlp.get("low_lt", constraints.risk_level_policy.low_lt), default=constraints.risk_level_policy.low_lt),
        moderate_lt=coerce_probability(
            rlp.get("moderate_lt", constraints.risk_level_policy.moderate_lt), default=constraints.risk_level_policy.moderate_lt
        ),
    )

    # Text-based overrides are often more reliable for UX than model-generated JSON.
    constraints.review_limit = _infer_review_limit(clinic_description, default=constraints.review_limit)
    constraints.cadence = _infer_cadence(clinic_description, default=constraints.cadence)
    constraints.dedup_days = _infer_dedup_days(clinic_description, default=constraints.dedup_days)
    inferred_threshold = _infer_threshold(clinic_description)
    if inferred_threshold is not None and constraints.selection.method != "first_k":
        constraints.selection.threshold = inferred_threshold
        if constraints.selection.method == "top_k":
            constraints.selection.method = "threshold_then_top_k"

    constraints.selection.k = constraints.review_limit
    if not constraints.selection.source_task:
        constraints.selection.source_task = _infer_default_source_task(enabled_by_text, tasks)
        if constraints.selection.source_task is None:
            constraints.selection.method = "first_k"

    out["constraints"] = constraints.model_dump()
    out["tasks"] = [t.model_dump() for t in tasks]
    if isinstance(obj.get("workflow"), dict):
        out["workflow"] = obj["workflow"]
    return out


def build_clinic_plan_from_description(
    *,
    llm: LLMJsonClient,
    clinic_description: str,
    default_target_condition: Optional[str] = None,
    default_horizon_months: Optional[int] = None,
) -> ClinicPlanSchema:
    system, user = prompts.clinic_plan_prompt(
        clinic_description=clinic_description,
        default_target_condition=default_target_condition,
        default_horizon_months=default_horizon_months,
        valid_task_names=TASK_CATALOG.keys(),
    )

    obj = llm.json_object_no_tools(system=system, user=user, temperature=0.0)
    if default_target_condition and not obj.get("target_condition"):
        # Only apply defaults if the plan includes risk scoring or explicitly mentions risk.
        raw_tasks = obj.get("tasks") if isinstance(obj.get("tasks"), list) else []
        if any(isinstance(t, dict) and t.get("name") == "risk_assessment" and t.get("enabled", True) for t in raw_tasks):
            obj["target_condition"] = default_target_condition
    if default_horizon_months and not obj.get("horizon_months") and obj.get("target_condition"):
        obj["horizon_months"] = default_horizon_months

    normalized = normalize_plan_payload(obj, clinic_description=clinic_description)
    return ClinicPlanSchema.model_validate(normalized)
