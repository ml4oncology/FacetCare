from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, redirect, render_template_string, request, url_for, make_response

from facetcare.llm_client import LLMJsonClient
from facetcare.plan_builder import TASK_CATALOG, build_clinic_plan_from_description
from facetcare.runner import TaskRunner
from facetcare.schemas import ClinicPlanSchema, PatientRecord, ReviewBundleSchema, TaskSpec
from facetcare.tasks import TaskContext, default_task_registry
from facetcare.output_cache import JSONTaskOutputCache

from openai import OpenAI

ENDPOINT = "http://192.168.0.2:881/v1/"
CLIENT = OpenAI(base_url=ENDPOINT, api_key="")
MODEL = "gpt-4.1-mini"  # Replace with model name if needed.

app = Flask(__name__)

# Re-entrant lock prevents deadlocks when route handlers call helpers that also lock.
_lock = threading.RLock()
current_plan: Optional[ClinicPlanSchema] = None
current_results = None
run_state: Dict[str, Any] = {"status": "idle", "message": "No run yet."}
patients_cache: List[PatientRecord] = []
reviewed_state_by_run: Dict[str, Dict[str, bool]] = {}
_task_output_cache = JSONTaskOutputCache(Path(__file__).resolve().parent / "task_output_cache.json")


# Python-side task groupings used for adaptive plan UI and labels
TARGET_CONTEXT_TASKS = {
    "risk_assessment",
    "clinician_summary",
    "admin_referral",
    "patient_instructions",
    "results_summary",
    "referral_letter",
    "differential_diagnosis",
    "guideline_comparison",
}
RISK_TASKS = {"risk_assessment"}


def _plan_has_any_task(plan: Optional[ClinicPlanSchema], names: set[str]) -> bool:
    if plan is None:
        return False
    try:
        return any(bool(getattr(t, "enabled", True)) and str(getattr(t, "name", "")) in names for t in (plan.tasks or []))
    except Exception:
        return False


def _run_button_label(plan: Optional[ClinicPlanSchema]) -> str:
    if plan is None:
        return "Run Workflow"
    try:
        enabled = {str(t.name) for t in (plan.tasks or []) if bool(getattr(t, "enabled", True))}
    except Exception:
        enabled = set()

    if "risk_assessment" in enabled:
        return "Run Screening"
    if "followup_gap_detection" in enabled:
        return "Run Follow-up Review"
    if enabled & {"results_summary", "lab_trend_summary"}:
        return "Run Results Review"
    if enabled & {"differential_diagnosis", "guideline_comparison"}:
        return "Run Clinical Review"
    if enabled & {"referral_letter", "admin_referral", "referral_intake_checklist"}:
        return "Run Referral Workflow"
    return "Run Workflow"


def _cache_summary() -> dict:
    try:
        return _task_output_cache.stats()
    except Exception:
        return {"entries": 0, "hits": 0, "misses": 0, "path": str(Path(__file__).resolve().parent / "task_output_cache.json")}


def _risk_prob(bundle) -> float:
    try:
        return float(bundle.risk.risk_probability) if getattr(bundle, "risk", None) else -1.0
    except Exception:
        return -1.0


def _risk_level(bundle) -> str:
    try:
        return str(bundle.risk.risk_level).lower() if getattr(bundle, "risk", None) else "none"
    except Exception:
        return "none"


def _bundle_artifact_flags(bundle) -> dict:
    extras = set(((getattr(bundle, "extra_outputs", None) or {}) or {}).keys())
    return {
        "risk": getattr(bundle, "risk", None) is not None,
        "clinician_summary": getattr(bundle, "clinician_summary", None) is not None,
        "admin_referral": getattr(bundle, "admin_referral", None) is not None,
        "referral_letter": "referral_letter" in extras,
        "patient_instructions": "patient_instructions" in extras,
        "lab_trend_summary": "lab_trend_summary" in extras,
        "followup_gap_detection": "followup_gap_detection" in extras,
        "extra_count": len(extras),
    }


def _truncate_text(text: str, n: int = 280) -> str:
    txt = (text or "").strip()
    return txt if len(txt) <= n else (txt[: n - 1] + "…")



def _infer_default_enabled_tasks(plan: ClinicPlanSchema) -> List[str]:
    desc = (plan.clinic_description or "").lower()
    chosen: List[str] = []

    def add(name: str) -> None:
        if name in TASK_CATALOG and name not in chosen:
            chosen.append(name)

    add("intake_workflow")
    if any(k in desc for k in ["risk", "screen", "screening", "cancer", "triage"]):
        add("risk_assessment")
        add("queue_prioritization")
        add("clinician_summary")
    if any(k in desc for k in ["follow-up", "followup", "missed", "care gap", "pending"]):
        add("followup_gap_detection")
        add("clinician_summary")
    if "referral" in desc:
        add("admin_referral")
        add("referral_letter")
        add("referral_intake_checklist")
    if any(k in desc for k in ["patient instruction", "patient instructions", "patient message", "after visit"]):
        add("patient_instructions")
    if any(k in desc for k in ["lab", "result", "imaging", "trend"]):
        add("results_summary")
        add("lab_trend_summary")
    if len(chosen) == 1:
        add("clinician_summary")
    return chosen


def _ensure_plan_tasks_catalog(plan: ClinicPlanSchema) -> ClinicPlanSchema:
    registry_order = list(TASK_CATALOG.keys())
    existing = {t.name: t for t in (plan.tasks or [])}
    if existing:
        ordered_names = [t.name for t in plan.tasks if t.name in TASK_CATALOG] + [n for n in registry_order if n not in existing]
        default_enabled: List[str] = []
    else:
        ordered_names = list(registry_order)
        default_enabled = _infer_default_enabled_tasks(plan)
    rebuilt: List[TaskSpec] = []
    for name in ordered_names:
        meta = TASK_CATALOG.get(name, {})
        prev = existing.get(name)
        enabled = bool(prev.enabled) if prev is not None else (name in default_enabled)
        params = dict(prev.params) if (prev is not None and isinstance(prev.params, dict)) else {}
        depends = list(meta.get("depends_on", []))
        rebuilt.append(TaskSpec(name=name, enabled=enabled, params=params, depends_on=depends))
    return plan.model_copy(update={"tasks": rebuilt})

def _app_state_path() -> Path:
    return Path(__file__).resolve().parent / "facetcare_app_state.json"

def _results_run_key(results: Any) -> str:
    if results is None:
        return "none"
    run_date = getattr(results, "run_date", "unknown") or "unknown"
    clinic_name = getattr(results, "clinic_name", "clinic") or "clinic"
    target = None
    try:
        target = getattr(getattr(results, "plan_snapshot", None), "target_condition", None)
    except Exception:
        target = None
    return f"{run_date}|{clinic_name}|{target or 'unknown'}"


def _review_map_for_results(results: Any, *, create: bool = False) -> Dict[str, bool]:
    global reviewed_state_by_run
    key = _results_run_key(results)
    if create and key not in reviewed_state_by_run:
        reviewed_state_by_run[key] = {}
    return reviewed_state_by_run.get(key, {})


def _is_reviewed_patient(results: Any, patient_id: str) -> bool:
    return bool(_review_map_for_results(results).get(patient_id, False))


def _set_reviewed_patient(results: Any, patient_id: str, reviewed: bool) -> None:
    if results is None or not patient_id:
        return
    with _lock:
        rm = _review_map_for_results(results, create=True)
        rm[patient_id] = bool(reviewed)
        _persist_state_locked()


def _output_completion_score(bundle: Any) -> int:
    flags = _bundle_artifact_flags(bundle)
    return int(sum(1 for k, v in flags.items() if k != "extra_count" and v))



def _persist_state_locked() -> None:
    """Persist current plan/results/run status to disk. Call while holding _lock."""
    payload: Dict[str, Any] = {
        "current_plan": current_plan.model_dump() if current_plan is not None else None,
        "current_results": current_results.model_dump() if current_results is not None else None,
        "run_state": dict(run_state or {}),
    }
    path = _app_state_path()
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        # Non-fatal for demo app
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _persist_state() -> None:
    with _lock:
        _persist_state_locked()


def _load_persisted_state() -> None:
    global current_plan, current_results, run_state
    path = _app_state_path()
    if not path.exists():
        return
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return

    if not isinstance(raw, dict):
        return

    cp = raw.get("current_plan")
    cr = raw.get("current_results")
    rs = raw.get("run_state")

    loaded_plan = None
    if isinstance(cp, dict):
        try:
            loaded_plan = _ensure_plan_tasks_catalog(ClinicPlanSchema.model_validate(cp))
        except Exception:
            loaded_plan = None

    loaded_results = None
    if isinstance(cr, dict):
        try:
            loaded_results = ReviewBundleSchema.model_validate(cr)
        except Exception:
            loaded_results = None

    loaded_status = None
    if isinstance(rs, dict):
        if rs.get("status") == "running":
            rs = {"status": "idle", "message": "Previous run was interrupted. Last saved results are loaded if available."}
        loaded_status = rs

    with _lock:
        current_plan = loaded_plan
        current_results = loaded_results
        if loaded_status is not None:
            run_state = loaded_status

def _make_llm_client() -> LLMJsonClient:
    return LLMJsonClient(client=CLIENT, model=MODEL)


def _make_runner() -> TaskRunner:
    llm = _make_llm_client()
    ctx = TaskContext(llm=llm)
    return TaskRunner(ctx=ctx, task_registry=default_task_registry(), output_cache=_task_output_cache)


def _sample_patients_path() -> Path:
    root = Path(__file__).resolve().parent
    for candidate in [root / "sample_patients_100.jsonl", root / "sample_patients.jsonl", root / "sample_patients.json"]:
        if candidate.exists():
            return candidate
    return root / "sample_patients_100.jsonl"


def _generate_demo_patients(n: int = 100) -> List[PatientRecord]:
    rng = random.Random(42)
    records: List[PatientRecord] = []
    symptom_pool = [
        "epigastric pain",
        "weight loss",
        "new-onset diabetes",
        "jaundice denied",
        "back pain",
        "dyspepsia",
        "fatigue",
        "decreased appetite",
        "bloating",
        "nausea",
    ]
    chronic_pool = ["HTN", "T2DM", "hyperlipidemia", "GERD", "none documented"]
    for i in range(1, n + 1):
        pid = f"P{i:04d}"
        age = rng.randint(38, 84)
        sex = rng.choice(["F", "M"])
        chronic = rng.sample(chronic_pool, k=2)
        symptoms = rng.sample(symptom_pool, k=3)
        note = (
            f"Family medicine progress notes for {pid}. Age {age} {sex}. "
            f"PMH: {', '.join(chronic)}. "
            f"Recent visits mention {', '.join(symptoms)}. "
            f"Follow-up plan discussed, labs/imaging may be pending. "
            f"Patient advised to return if symptoms worsen."
        )
        if i % 7 == 0:
            note += " Family history of GI malignancy mentioned."
        if i % 9 == 0:
            note += " CT abdomen ordered but completion unclear."
        if i % 11 == 0:
            note += " Recurrent abnormal LFT wording appears in note summaries."
        records.append(PatientRecord(patient_id=pid, longitudinal_notes=note))
    return records


def load_sample_patients() -> List[PatientRecord]:
    global patients_cache
    with _lock:
        if patients_cache:
            return list(patients_cache)

    path = _sample_patients_path()
    patients: List[PatientRecord] = []
    if path.exists():
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    patients.append(PatientRecord.model_validate(obj))
                except Exception:
                    continue
        elif path.suffix == ".json":
            try:
                arr = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(arr, list):
                    for obj in arr:
                        try:
                            patients.append(PatientRecord.model_validate(obj))
                        except Exception:
                            continue
            except Exception:
                pass

    if not patients:
        patients = _generate_demo_patients(100)
        try:
            with path.open("w", encoding="utf-8") as f:
                for p in patients:
                    f.write(json.dumps(p.model_dump(), ensure_ascii=False) + "\n")
        except Exception:
            pass

    with _lock:
        patients_cache = list(patients)
    return patients


_load_persisted_state()


def _plan_to_json_text(plan: ClinicPlanSchema) -> str:
    return json.dumps(plan.model_dump(), indent=2, ensure_ascii=False)


def _json_or_default(s: str, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else default
    except Exception:
        return default


def _taskspec_index(plan: ClinicPlanSchema) -> Dict[str, Any]:
    return {t.name: t for t in plan.tasks}


def _task_depends_on(plan: ClinicPlanSchema, task_name: str) -> List[str]:
    idx = _taskspec_index(plan)
    t = idx.get(task_name)
    if t and t.depends_on:
        return list(t.depends_on or [])
    return list((TASK_CATALOG.get(task_name) or {}).get("depends_on") or [])


MANUAL_TASK_PREREQS: Dict[str, List[str]] = {
    # These improve single-click generation quality from the results page by seeding richer context.
    "clinician_summary": ["risk_assessment"],
    "admin_referral": ["risk_assessment", "clinician_summary"],
    "referral_letter": ["risk_assessment", "clinician_summary", "admin_referral"],
    "patient_instructions": ["risk_assessment", "clinician_summary"],
    "differential_diagnosis": ["risk_assessment", "clinician_summary"],
    "guideline_comparison": ["risk_assessment", "clinician_summary"],
}




def _existing_task_output_from_state(state: Dict[str, Any], task_name: str, patient_id: str) -> Any:
    # Common canonical key
    val = (state.get(f"{task_name}_by_patient") or {}).get(patient_id)
    if val is not None:
        return val
    # Backward-compatible aliases used by some task implementations
    aliases = {
        "queue_prioritization": "queue_priority_by_patient",
        "followup_gap_detection": "followup_gap_by_patient",
        "referral_intake_checklist": "referral_intake_by_patient",
        "lab_trend_summary": "lab_trend_by_patient",
        "care_plan_reconciliation": "care_plan_recon_by_patient",
    }
    alias_key = aliases.get(task_name)
    if alias_key:
        return (state.get(alias_key) or {}).get(patient_id)
    if task_name == "intake_workflow":
        return state.get("workflow")
    return None


def _run_single_task_action(patient_id: str, task_name: str, *, force_refresh: bool = False) -> Optional[str]:
    global current_results
    with _lock:
        plan = current_plan
        results = current_results
    if plan is None or results is None:
        return "No plan/results available."

    patient_lookup = {p.patient_id: p for p in load_sample_patients()}
    patient = patient_lookup.get(patient_id)
    if patient is None:
        return f"Patient {patient_id} not found."

    bundle = None
    for b in results.selected:
        if b.patient_id == patient_id:
            bundle = b
            break
    if bundle is None:
        return f"Patient {patient_id} is not in the selected list for this run."

    runner = _make_runner()
    state: Dict[str, Any] = {}
    # Seed state from existing bundle to avoid recomputing everything.
    if plan.workflow is not None:
        state["workflow"] = plan.workflow
    if bundle.risk is not None:
        state.setdefault("risk_by_patient", {})[patient_id] = bundle.risk
    if bundle.clinician_summary is not None:
        state.setdefault("clinician_summary_by_patient", {})[patient_id] = bundle.clinician_summary
    if bundle.admin_referral is not None:
        state.setdefault("admin_referral_by_patient", {})[patient_id] = bundle.admin_referral

    # Rehydrate some extra outputs when present
    for extra_name, payload in (bundle.extra_outputs or {}).items():
        if isinstance(payload, dict):
            state.setdefault(f"{extra_name}_by_patient", {})[patient_id] = payload

    visiting: set[str] = set()

    def ensure_task(name: str, *, force: bool = False) -> Any:
        if not force:
            cached = _existing_task_output_from_state(state, name, patient_id)
            if cached is not None:
                return cached
        if name in visiting:
            return _existing_task_output_from_state(state, name, patient_id)
        visiting.add(name)
        try:
            for dep in _task_depends_on(plan, name):
                ensure_task(dep, force=False)
            if not force:
                cached = _existing_task_output_from_state(state, name, patient_id)
                if cached is not None:
                    return cached
            out = runner._run_task_for_patient(name, plan, patient, state, use_cache=(not force))
            if out is not None:
                state.setdefault(f"{name}_by_patient", {})[patient_id] = out
            return out
        finally:
            visiting.discard(name)

    if task_name not in runner.task_registry:
        return f"Task {task_name} is not available."

    for prereq in MANUAL_TASK_PREREQS.get(task_name, []):
        if prereq in runner.task_registry and prereq != task_name:
            ensure_task(prereq, force=False)

    out = ensure_task(task_name, force=force_refresh)
    if out is None:
        return f"Task {task_name} returned no output."

    # Write back into bundle fields / extras
    if task_name == "risk_assessment":
        bundle.risk = out
    elif task_name == "clinician_summary":
        bundle.clinician_summary = out
    elif task_name == "admin_referral":
        bundle.admin_referral = out
    else:
        if bundle.extra_outputs is None:
            bundle.extra_outputs = {}
        bundle.extra_outputs[task_name] = out.model_dump() if hasattr(out, "model_dump") else out

    with _lock:
        current_results = results
        _persist_state_locked()
    return None


@app.route("/")
def home():
    with _lock:
        has_state = (current_plan is not None) or (current_results is not None)
    return redirect(url_for("dashboard" if has_state else "setup_plan"))


@app.route("/dashboard")
def dashboard():
    global current_plan
    with _lock:
        plan = _ensure_plan_tasks_catalog(current_plan) if current_plan is not None else None
        if plan is not None and current_plan is not None and len(plan.tasks) != len(current_plan.tasks):
            current_plan = plan
            _persist_state_locked()
        results = current_results
        status = dict(run_state)
    active_tasks = []
    if plan is not None:
        active_tasks = [t.name for t in (plan.tasks or []) if t.enabled]
    run_label = _run_button_label(plan) if plan is not None else "Run Workflow"
    with _lock:
        has_plan = current_plan is not None
        run_label = _run_button_label(current_plan) if current_plan is not None else "Run Workflow"

    tpl = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>FacetCare — Dashboard</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet"/>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet"/>
  <style>
    body { background-color: #f0f4f8; font-family: 'Segoe UI', sans-serif; }
    .brand-bar { background: linear-gradient(90deg, #1a3c5e 0%, #2563a8 100%); padding: 1.1rem 2rem; color: white; }
    .brand-title-link { color: white; text-decoration: none; font-size: 1.6rem; font-weight: 700; letter-spacing: 0.03em; }
    .brand-title-link:hover { color: #eef6ff; }
    .brand-subtitle { font-size: 0.88rem; opacity: 0.82; margin-top: 0.1rem; }
    .card-panel { border: none; border-radius: 12px; box-shadow: 0 4px 24px rgba(30,60,100,0.10); background: white; }
    .kpi { font-size: 1.35rem; font-weight: 700; color: #1a3c5e; }
    .muted { color: #6b7a8d; font-size: 0.86rem; }
    .pill { display:inline-block; padding:0.2rem 0.55rem; border-radius:999px; background:#e8f0fb; color:#1a4080; font-size:0.75rem; margin:0.1rem 0.2rem 0.1rem 0; }
  </style>
</head>
<body>
<div class="brand-bar d-flex align-items-center gap-3">
  <i class="bi bi-heart-pulse-fill fs-3"></i>
  <div>
    <a class="brand-title-link" href="{{ url_for('dashboard') }}">FacetCare</a>
    <div class="brand-subtitle">AI-Assisted Clinical Workflow Orchestration</div>
  </div>
  <div class="ms-auto d-flex align-items-center gap-2">
    <a href="{{ url_for('dashboard') }}" class="btn btn-sm btn-light text-primary fw-semibold"><i class="bi bi-grid-1x2 me-1"></i>Dashboard</a>
    <a href="{{ url_for('setup_plan') }}" class="btn btn-sm btn-outline-light"><i class="bi bi-sliders me-1"></i>Setup</a>
    <a href="{{ url_for('plan_page') }}" class="btn btn-sm btn-outline-light"><i class="bi bi-diagram-3 me-1"></i>Plan</a>
    {% if plan %}<a href="{{ url_for('run_plan') }}" class="btn btn-sm btn-outline-light"><i class="bi bi-play-fill me-1"></i>{{ run_label }}</a>{% endif %}
    <a href="{{ url_for('results_page') }}" class="btn btn-sm btn-outline-light"><i class="bi bi-table me-1"></i>Results</a>
  </div>
</div>
<div class="container py-4" style="max-width:980px;">
  <div class="row g-3 mb-3">
    <div class="col-md-3"><div class="card-panel p-3"><div class="muted">Status</div><div class="kpi" style="font-size:1.05rem;">{{ status.get('status','idle')|capitalize }}</div><div class="muted">{{ status.get('message','No run yet.') }}</div></div></div>
    <div class="col-md-3"><div class="card-panel p-3"><div class="muted">Plan</div><div class="kpi">{{ 'Ready' if plan else 'None' }}</div><div class="muted">{{ plan.clinic_name if plan else 'Create a plan in Setup' }}</div></div></div>
    <div class="col-md-3"><div class="card-panel p-3"><div class="muted">Selected Patients</div><div class="kpi">{{ results.selected|length if results else 0 }}</div><div class="muted">Latest run shortlist size</div></div></div>
    <div class="col-md-3"><div class="card-panel p-3"><div class="muted">Run Date</div><div class="kpi" style="font-size:1.05rem;">{{ results.run_date if results else 'N/A' }}</div><div class="muted">Most recent workflow run</div></div></div>
  </div>

  <div class="row g-3">
    <div class="col-lg-7">
      <div class="card-panel p-4 mb-3">
        <h5 class="fw-bold mb-2" style="color:#1a3c5e;">Quick start</h5>
        <ol class="mb-0" style="padding-left:1.15rem;">
          <li class="mb-1">Configure the clinic workflow in <a href="{{ url_for('setup_plan') }}">Setup</a>.</li>
          <li class="mb-1">Review tasks and constraints in <a href="{{ url_for('plan_page') }}">Plan</a>.</li>
          <li class="mb-1">Run the workflow, then review outputs in <a href="{{ url_for('results_page') }}">Results</a>.</li>
        </ol>
      </div>
      <div class="card-panel p-4">
        <h5 class="fw-bold mb-2" style="color:#1a3c5e;">Plan summary</h5>
        {% if not plan %}
          <p class="text-muted mb-0">No saved plan yet.</p>
        {% else %}
          <div class="mb-2"><strong>Workflow:</strong> {{ plan.clinic_name or 'Unnamed workflow' }}</div>
          <div class="mb-2"><strong>Run mode:</strong> {{ run_label }}</div>
          <div class="mb-2"><strong>Review limit:</strong> {{ plan.constraints.review_limit }} &nbsp; <strong>Dedup:</strong> {{ plan.constraints.dedup_days }} days</div>
          {% if plan.target_condition %}<div class="mb-2"><strong>Target condition:</strong> {{ plan.target_condition }}</div>{% endif %}
          {% if plan.horizon_months %}<div class="mb-2"><strong>Horizon:</strong> {{ plan.horizon_months }} months</div>{% endif %}
          <div class="muted mb-1">Active tasks</div>
          <div>
            {% for t in active_tasks %}<span class="pill">{{ t }}</span>{% endfor %}
            {% if not active_tasks %}<span class="text-muted">No active tasks.</span>{% endif %}
          </div>
        {% endif %}
      </div>
    </div>
    <div class="col-lg-5">
      <div class="card-panel p-4 mb-3">
        <h5 class="fw-bold mb-2" style="color:#1a3c5e;">Latest run</h5>
        {% if results %}
          <div class="mb-2"><strong>Clinic:</strong> {{ results.clinic_name }}</div>
          <div class="mb-2"><strong>Selected:</strong> {{ results.selected|length }} / {{ (results.selected|length) + (results.not_selected_count or 0) }}</div>
          <div class="mb-2"><strong>Not selected:</strong> {{ results.not_selected_count }}</div>
          {% if results.selected %}
            <div class="muted mb-1">Top patients</div>
            <ul class="mb-0 ps-3">
              {% for b in results.selected[:5] %}
                <li>{{ b.patient_id }}{% if b.risk %} · {{ '%.3f'|format(b.risk.risk_probability) }} {{ b.risk.risk_level }}{% endif %}</li>
              {% endfor %}
            </ul>
          {% endif %}
        {% else %}
          <p class="text-muted mb-0">No results yet.</p>
        {% endif %}
      </div>
      <div class="card-panel p-4">
        <h5 class="fw-bold mb-2" style="color:#1a3c5e;">Next actions</h5>
        <div class="d-grid gap-2">
          <a class="btn btn-primary" href="{{ url_for('plan_page') }}"><i class="bi bi-pencil-square me-1"></i>Edit Plan</a>
          {% if plan %}<a class="btn btn-success" href="{{ url_for('run_plan') }}"><i class="bi bi-play-fill me-1"></i>{{ run_label }}</a>{% endif %}
          <a class="btn btn-outline-secondary" href="{{ url_for('results_page') }}"><i class="bi bi-table me-1"></i>Open Results</a>
        </div>
      </div>
    </div>
  </div>
</div>
</body>
</html>
    """
    return render_template_string(tpl, plan=plan, results=results, status=status, active_tasks=active_tasks, run_label=run_label)


@app.route("/setup", methods=["GET", "POST"])
def setup_plan():
    global current_plan
    with _lock:
        _load_persisted_state()
        has_plan = current_plan is not None
        run_label = _run_button_label(current_plan) if current_plan is not None else "Run Workflow"

    default_desc = (
        "Family medicine clinic. Weekly clinician review of a shortlist of patients. "
        "Goal is to identify patients who may be at higher risk of pancreatic cancer and prepare referral-ready summaries. "
        "Do not repeat the same patient within 30 days. No diagnosis, decision support only."
    )
    msg = ""
    if request.method == "POST":
        desc = request.form.get("clinic_description", "").strip() or default_desc
        default_target = request.form.get("default_target_condition", "").strip() or None
        horizon_raw = request.form.get("default_horizon_months", "").strip()
        default_horizon = int(horizon_raw) if horizon_raw.isdigit() else None
        setup_max_candidates_raw = request.form.get("setup_max_candidates", "").strip()
        setup_candidate_strategy = request.form.get("setup_candidate_pool_strategy", "").strip()
        setup_manual_include_raw = request.form.get("setup_manual_include_patient_ids", "")
        try:
            plan = build_clinic_plan_from_description(
                llm=_make_llm_client(),
                clinic_description=desc,
                default_target_condition=default_target,
                default_horizon_months=default_horizon,
            )
            cp = plan.constraints.candidate_pool
            if setup_candidate_strategy:
                cp.strategy = setup_candidate_strategy
            if setup_max_candidates_raw.isdigit():
                cp.max_candidates = max(1, int(setup_max_candidates_raw))
            setup_manual_ids = [x.strip() for x in re.split(r"[,\n]+", setup_manual_include_raw or "") if x.strip()]
            if setup_manual_ids:
                cp.include_patient_ids = setup_manual_ids

            plan = _ensure_plan_tasks_catalog(plan)
            with _lock:
                current_plan = plan
                _persist_state_locked()
            msg = "Plan generated. Review and edit it on the Plan page."
            return redirect(url_for("plan_page"))
        except Exception as e:
            msg = f"Plan generation failed: {e}"

    tpl = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>FacetCare — Clinic Setup</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" />
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet" />
  <style>
    body { background-color: #f0f4f8; font-family: 'Segoe UI', sans-serif; }
    .brand-bar { background: linear-gradient(90deg, #1a3c5e 0%, #2563a8 100%); padding: 1.1rem 2rem; color: white; }
    .brand-title-link { color:white; text-decoration:none; font-size: 1.6rem; font-weight: 700; letter-spacing: 0.03em; }
    .brand-title-link:hover { color:#eef6ff; }
    .brand-bar .brand-subtitle { font-size: 0.88rem; opacity: 0.82; margin-top: 0.1rem; }
    .card-setup { border: none; border-radius: 12px; box-shadow: 0 4px 24px rgba(30,60,100,0.10); }
    .section-label { font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; color: #6b7a8d; margin-bottom: 0.3rem; }
    .field-hint { font-size: 0.82rem; color: #7a8898; margin-top: 0.3rem; }
    .example-pill { display: inline-block; cursor: pointer; background: #e8f0fb; color: #1a4080; border-radius: 20px; padding: 0.25rem 0.75rem; font-size: 0.8rem; margin: 0.2rem 0.15rem 0.2rem 0; border: 1px solid #c5d8f5; transition: background 0.15s; }
    .example-pill:hover { background: #cfe0f7; }
    .btn-generate { background: linear-gradient(90deg, #1a3c5e, #2563a8); color: white; border: none; border-radius: 8px; padding: 0.65rem 2.2rem; font-size: 1rem; font-weight: 600; letter-spacing: 0.02em; }
    .btn-generate:hover { opacity: 0.9; color: white; }
    .alert-info-soft { background: #eaf2ff; border: 1px solid #c5d8f5; border-radius: 8px; color: #1a3c5e; font-size: 0.88rem; }
    .step-badge { display: inline-flex; align-items: center; justify-content: center; background: #2563a8; color: white; border-radius: 50%; width: 24px; height: 24px; font-size: 0.78rem; font-weight: 700; margin-right: 0.5rem; flex-shrink: 0; }
  </style>
</head>
<body>
<div class="brand-bar d-flex align-items-center gap-3">
  <i class="bi bi-heart-pulse-fill fs-3"></i>
  <div>
    <a class="brand-title-link" href="{{ url_for('dashboard') }}">FacetCare</a>
    <div class="brand-subtitle">AI-Assisted Clinical Workflow Orchestration</div>
  </div>
  <div class="ms-auto d-flex gap-2 flex-wrap">
    <a href="{{ url_for('dashboard') }}" class="btn btn-sm btn-outline-light"><i class="bi bi-grid-1x2 me-1"></i>Dashboard</a>
    <a href="{{ url_for('setup_plan') }}" class="btn btn-sm btn-light text-primary fw-semibold"><i class="bi bi-sliders me-1"></i>Setup</a>
    <a href="{{ url_for('plan_page') }}" class="btn btn-sm btn-outline-light"><i class="bi bi-diagram-3 me-1"></i>Plan</a>
    {% if has_plan %}<a href="{{ url_for('run_plan') }}" class="btn btn-sm btn-outline-light"><i class="bi bi-play-fill me-1"></i>{{ run_label }}</a>{% endif %}
    <a href="{{ url_for('results_page') }}" class="btn btn-sm btn-outline-light"><i class="bi bi-table me-1"></i>Results</a>
  </div>
</div>
<div class="container py-5" style="max-width:780px;">
  <div class="alert alert-info-soft d-flex gap-2 align-items-start mb-4 px-4 py-3">
    <i class="bi bi-info-circle-fill mt-1 flex-shrink-0" style="color:#2563a8;"></i>
    <div><strong>How FacetCare works:</strong> Describe your clinic's goals in plain language below. FacetCare will use AI to build a screening plan, scan longitudinal notes, identify elevated-risk patients, and generate clinician-ready outputs for human review.</div>
  </div>
  {% if msg %}
  <div class="alert alert-danger d-flex align-items-center gap-2 mb-4" role="alert"><i class="bi bi-exclamation-triangle-fill"></i><div>{{ msg }}</div></div>
  {% endif %}
  <div class="card card-setup p-4 p-md-5">
    <h2 class="fw-bold mb-1" style="color:#1a3c5e;">Configure Your Screening Workflow</h2>
    <p class="text-muted mb-4" style="font-size:0.92rem;">Fill in the fields below to generate a tailored screening plan for your clinic. All fields except the description have sensible defaults and can be left blank.</p>
    <form method="POST" action="/setup">
      <div class="mb-4">
        <div class="d-flex align-items-center mb-1"><span class="step-badge">1</span><label class="fw-semibold fs-6 mb-0" for="clinic_description" style="color:#1a3c5e;">Describe Your Clinic's Goals</label></div>
        <p class="field-hint mb-2">Describe your clinic, target condition, cadence, shortlist size, and constraints in plain language.</p>
        <div class="mb-2">
          <span class="section-label">Quick examples, click to use:</span><br/>
          <span class="example-pill" onclick="setExample('pancreatic')">🔬 Pancreatic cancer risk</span>
          <span class="example-pill" onclick="setExample('colorectal')">🔬 Colorectal cancer risk</span>
          <span class="example-pill" onclick="setExample('followup')">📋 Follow-up gaps</span>
          <span class="example-pill" onclick="setExample('referral')">📨 Referral triage</span>
        </div>
        <textarea class="form-control" id="clinic_description" name="clinic_description" rows="6" style="font-size:0.93rem; border-radius:8px;">{{ default_desc }}</textarea>
        <div class="field-hint"><i class="bi bi-lightbulb"></i> Tip: Include cadence, shortlist size, dedup window, and guardrails like decision support only.</div>
      </div>
      <div class="mb-4">
        <div class="d-flex align-items-center mb-1"><span class="step-badge">2</span><label class="fw-semibold fs-6 mb-0" style="color:#1a3c5e;">Optional Screening Parameters</label></div>
        <p class="field-hint mb-3">These override defaults extracted from your description. Leave blank to let FacetCare infer them.</p>
        <div class="row g-3">
          <div class="col-md-6">
            <label class="form-label section-label" for="default_target_condition">Target Condition</label>
            <input type="text" class="form-control" id="default_target_condition" name="default_target_condition" placeholder="e.g. pancreatic_cancer" style="border-radius:8px;" />
            <div class="field-hint">Use underscores, no spaces.</div>
          </div>
          <div class="col-md-6">
            <label class="form-label section-label" for="default_horizon_months">Risk Horizon (months)</label>
            <input type="number" class="form-control" id="default_horizon_months" name="default_horizon_months" min="1" max="120" placeholder="36" style="border-radius:8px;" />
            <div class="field-hint">Timeframe for risk assessment, 1 to 120 months.</div>
          </div>
        </div>
      </div>

      <div class="mb-3 mt-3">
        <div class="d-flex align-items-center mb-1">
          <span class="step-badge">3</span>
          <label class="fw-semibold fs-6 mb-0" style="color:#1a3c5e;">Advanced Mode</label>
        </div>
        <p class="field-hint mb-2">Optional workflow controls. Hidden by default to keep setup simple.</p>
        <button class="btn btn-sm btn-outline-primary" type="button" data-bs-toggle="collapse" data-bs-target="#setupAdvancedCollapse" aria-expanded="false" aria-controls="setupAdvancedCollapse">
          <i class="bi bi-sliders2 me-1"></i>Show advanced options
        </button>
        <div class="collapse mt-2" id="setupAdvancedCollapse">
          <div class="border rounded-3 p-3" style="background:#fafcff;border-color:#dbe8fb !important;">
            <div class="row g-3">
              <div class="col-md-4">
                <label class="form-label section-label" for="setup_max_candidates">Max Patients to Process</label>
                <input type="number" class="form-control" id="setup_max_candidates" name="setup_max_candidates" min="1" max="5000" placeholder="e.g. 100" style="border-radius:8px;" />
                <div class="field-hint">Caps the candidate pool before patient-level tasks run.</div>
              </div>
              <div class="col-md-4">
                <label class="form-label section-label" for="setup_candidate_pool_strategy">Candidate Pool Strategy</label>
                <select class="form-select" id="setup_candidate_pool_strategy" name="setup_candidate_pool_strategy" style="border-radius:8px;">
                  <option value="">Use AI defaults</option>
                  <option value="all">All patients</option>
                  <option value="keyword_prefilter">Keyword prefilter</option>
                  <option value="recent_notes_only">Recent notes only</option>
                </select>
                <div class="field-hint">You can fine-tune this later in the Plan page.</div>
              </div>
              <div class="col-md-12">
                <label class="form-label section-label" for="setup_manual_include_patient_ids">Manual Include-List (optional)</label>
                <textarea class="form-control" id="setup_manual_include_patient_ids" name="setup_manual_include_patient_ids" rows="3" placeholder="P0001, P0007, P0042 or one ID per line" style="border-radius:8px;"></textarea>
                <div class="field-hint">If provided, FacetCare restricts processing to these patient IDs. Useful for manual review batches and demos.</div>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="d-grid mt-4"><button type="submit" class="btn btn-generate btn-lg"><i class="bi bi-cpu me-2"></i>Generate Screening Plan</button></div>
      <p class="text-center text-muted mt-3" style="font-size:0.82rem;">FacetCare will analyze your description and build a structured plan. You can review and adjust it before any patients are screened.</p>
    </form>
  </div>
</div>
<script>
const examples = {
  pancreatic: {
    desc: "Family medicine clinic in rural Ontario. Weekly clinician review, shortlist top 5 patients. Goal: identify patients at elevated risk of pancreatic cancer based on longitudinal EMR notes. Generate risk assessment, clinician chart summary, admin referral payload, and patient instructions. Do not repeat patients within 6 months. No diagnosis, decision support only.",
    target: "pancreatic_cancer",
    horizon: 36
  },
  colorectal: {
    desc: "Community family practice. Monthly review of top 5 patients overdue for colorectal cancer screening or showing relevant risk factors in their notes. Generate clinician summaries and referral letters for patients at moderate or high risk. Decision support only.",
    target: "colorectal_cancer",
    horizon: 36
  },
  followup: {
    desc: "Family medicine clinic. Weekly review of patients with potential follow-up gaps: missed appointments, abnormal lab results not acted on, pending imaging. Generate follow-up gap reports and suggested care plan actions. No repeated patients within 30 days.",
    target: "",
    horizon: ""
  },
  referral: {
    desc: "Rural family practice. Weekly session to prioritize and triage referral queue. For each top-5 patient, generate a structured referral intake checklist and referral letter. Highlight semi-urgent and urgent cases. Decision support only.",
    target: "",
    horizon: ""
  }
};

function setExample(key) {
  const ex = examples[key];
  if (!ex) return;
  document.getElementById('clinic_description').value = ex.desc || '';
  const targetEl = document.getElementById('default_target_condition');
  const horizonEl = document.getElementById('default_horizon_months');
  if (targetEl) targetEl.value = ex.target ?? '';
  if (horizonEl) horizonEl.value = ex.horizon ?? '';
}
</script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    """
    return render_template_string(tpl, default_desc=default_desc, msg=msg, has_plan=has_plan, run_label=run_label)


@app.route("/plan", methods=["GET", "POST"])
def plan_page():
    global current_plan
    with _lock:
        plan = _ensure_plan_tasks_catalog(current_plan) if current_plan is not None else None
        if plan is not None and current_plan is not None and len(plan.tasks) != len(current_plan.tasks):
            current_plan = plan
            _persist_state_locked()
    if plan is None:
        return redirect(url_for("setup_plan"))

    msg = ""
    if request.method == "POST":
        data = plan.model_dump()
        data["target_condition"] = (request.form.get("target_condition", "").strip() or None)
        hm_raw = request.form.get("horizon_months", "").strip()
        data["horizon_months"] = int(hm_raw) if hm_raw.isdigit() else None

        c = data.get("constraints", {})
        c["cadence"] = request.form.get("cadence", c.get("cadence", "weekly"))
        try:
            c["review_limit"] = max(1, int(request.form.get("review_limit", c.get("review_limit", 5))))
        except Exception:
            pass
        try:
            c["dedup_days"] = max(0, int(request.form.get("dedup_days", c.get("dedup_days", 30))))
        except Exception:
            pass
        cp = c.get("candidate_pool", {})
        cp["strategy"] = request.form.get("candidate_pool_strategy", cp.get("strategy", "all"))
        cp["max_candidates"] = int(request.form["max_candidates"]) if request.form.get("max_candidates", "").isdigit() else None
        cp["keywords"] = [k.strip() for k in request.form.get("candidate_pool_keywords", "").split(",") if k.strip()]
        cp["include_patient_ids"] = [x.strip() for x in re.split(r"[,\n]+", request.form.get("manual_include_patient_ids", "") or "") if x.strip()]
        c["candidate_pool"] = cp
        sel = c.get("selection", {})
        sel["source_task"] = request.form.get("selection_source_task", "").strip() or None
        sel["method"] = request.form.get("selection_method", sel.get("method", "top_k"))
        sel["k"] = max(1, int(request.form.get("selection_k", sel.get("k", 5)) or 5))
        thr_raw = request.form.get("selection_threshold", "").strip()
        sel["threshold"] = float(thr_raw) if thr_raw else None
        c["selection"] = sel
        rlp = c.get("risk_level_policy", {})
        try:
            rlp["low_lt"] = float(request.form.get("risk_low_lt", rlp.get("low_lt", 0.1)))
            rlp["moderate_lt"] = float(request.form.get("risk_moderate_lt", rlp.get("moderate_lt", 0.2)))
        except Exception:
            pass
        c["risk_level_policy"] = rlp
        data["constraints"] = c

        # task editing
        task_map = {t["name"]: t for t in data.get("tasks", [])}
        has_task_payload = any(k.startswith(("enabled__", "params__", "order__")) for k in request.form.keys())
        if has_task_payload:
            ordered_names = []
            for name, task_obj in task_map.items():
                task_obj["enabled"] = request.form.get(f"enabled__{name}") == "on"
                params_txt = request.form.get(f"params__{name}", "{}").strip() or "{}"
                try:
                    parsed = json.loads(params_txt)
                    task_obj["params"] = parsed if isinstance(parsed, dict) else {}
                except Exception:
                    msg = f"Invalid JSON for task params: {name}. Changes still saved for other fields."
                try:
                    order_val = int(request.form.get(f"order__{name}", "999"))
                except Exception:
                    order_val = 999
                ordered_names.append((order_val, name))
            ordered_names.sort(key=lambda x: (x[0], x[1]))
            data["tasks"] = [task_map[n] for _, n in ordered_names]
        else:
            data["tasks"] = data.get("tasks", [])

        try:
            updated = _ensure_plan_tasks_catalog(ClinicPlanSchema.model_validate(data))
            with _lock:
                current_plan = updated
                _persist_state_locked()
            plan = updated
            if not msg:
                msg = "Plan updated."
        except Exception as e:
            msg = f"Validation error while saving plan: {e}"

    plan_json = _plan_to_json_text(plan)
    if not (plan.tasks or []):
        plan = _ensure_plan_tasks_catalog(plan)
        with _lock:
            current_plan = plan
            _persist_state_locked()

    task_rows = []
    task_dep_map = {t.name: list(t.depends_on or []) for t in plan.tasks}
    ui_flags = {
        "show_target_fields": _plan_has_any_task(plan, TARGET_CONTEXT_TASKS),
        "show_risk_fields": _plan_has_any_task(plan, RISK_TASKS),
        "run_label": _run_button_label(plan),
    }
    run_label = ui_flags["run_label"]
    constraints_payload = plan.constraints.model_dump()
    for t in plan.tasks:
        task_rows.append(
            {
                "name": t.name,
                "enabled": t.enabled,
                "depends_on": ", ".join(t.depends_on or []),
                "params_text": json.dumps(t.params or {}, ensure_ascii=False, indent=2),
            }
        )

    tpl = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>FacetCare — Review &amp; Configure Plan</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet"/>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet"/>
  <style>
    body { background-color: #f0f4f8; font-family: 'Segoe UI', sans-serif; }
    .brand-bar { background: linear-gradient(90deg, #1a3c5e 0%, #2563a8 100%); padding: 1.1rem 2rem; color: white; }
    .brand-title-link { color:white; text-decoration:none; font-size: 1.6rem; font-weight: 700; letter-spacing: 0.03em; }
    .brand-title-link:hover { color:#eef6ff; }
    .brand-bar .brand-subtitle { font-size: 0.88rem; opacity: 0.82; margin-top: 0.1rem; }
    .card-panel { border: none; border-radius: 12px; box-shadow: 0 4px 24px rgba(30,60,100,0.10); background: white; }
    .section-label { font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; color: #6b7a8d; margin-bottom: 0.3rem; }
    .field-hint { font-size: 0.82rem; color: #7a8898; margin-top: 0.25rem; }
    .step-badge { display: inline-flex; align-items: center; justify-content: center; background: #2563a8; color: white; border-radius: 50%; width: 24px; height: 24px; font-size: 0.78rem; font-weight: 700; margin-right: 0.5rem; flex-shrink: 0; }
    .btn-primary-facetcare { background: linear-gradient(90deg, #1a3c5e, #2563a8); color: white; border: none; border-radius: 8px; padding: 0.6rem 1.8rem; font-weight: 600; }
    .btn-primary-facetcare:hover { opacity: 0.9; color: white; }
    .btn-run { background: linear-gradient(90deg, #145a32, #1e8449); color: white; border: none; border-radius: 8px; padding: 0.6rem 1.8rem; font-weight: 600; }
    .btn-run:hover { opacity: 0.9; color: white; }
    .btn-run:disabled { background: #adb5bd; cursor: not-allowed; opacity: 0.7; }
    .task-zone { min-height: 80px; border-radius: 10px; border: 2px dashed #c5d8f5; padding: 0.75rem; background: #f7faff; transition: background 0.2s; }
    .task-zone.drag-over { background: #dceeff; border-color: #2563a8; }
    .task-card { background: white; border-radius: 8px; border: 1px solid #dde6f0; box-shadow: 0 2px 6px rgba(30,60,100,0.07); padding: 0.65rem 0.9rem; margin-bottom: 0.5rem; cursor: grab; display: flex; align-items: flex-start; gap: 0.6rem; transition: box-shadow 0.15s; user-select: none; }
    .task-card:active { cursor: grabbing; box-shadow: 0 6px 18px rgba(30,60,100,0.18); }
    .task-card.dragging { opacity: 0.45; }
    .task-icon { font-size: 1.2rem; margin-top: 0.05rem; flex-shrink: 0; }
    .task-name { font-weight: 600; font-size: 0.9rem; color: #1a3c5e; }
    .task-desc { font-size: 0.78rem; color: #6b7a8d; margin-top: 0.1rem; }
    .task-badge-dep { display: inline-block; font-size: 0.72rem; background: #e8f0fb; color: #1a4080; border-radius: 12px; padding: 0.1rem 0.5rem; margin-top: 0.25rem; margin-right: 0.2rem; }
    .params-toggle { font-size: 0.75rem; color: #2563a8; cursor: pointer; margin-top: 0.3rem; text-decoration: underline dotted; }
    .params-box { display: none; margin-top: 0.4rem; }
    .params-box textarea { font-size: 0.78rem; font-family: monospace; border-radius: 6px; }
    .dep-error { font-size: 0.75rem; color: #c0392b; background: #fdf0ef; border: 1px solid #f5c6c6; border-radius: 6px; padding: 0.25rem 0.6rem; margin-top: 0.3rem; display: none; }
    .unsaved-banner { display: none; background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px; padding: 0.5rem 1rem; font-size: 0.86rem; color: #7d5a00; align-items: center; gap: 0.5rem; }
  </style>
</head>
<body>
<div class="brand-bar d-flex align-items-center gap-3">
  <i class="bi bi-heart-pulse-fill fs-3"></i>
  <div>
    <a class="brand-title-link" href="{{ url_for('dashboard') }}">FacetCare</a>
    <div class="brand-subtitle">AI-Assisted Clinical Workflow Orchestration</div>
  </div>
  <div class="ms-auto d-flex gap-2 flex-wrap">
    <a href="{{ url_for('dashboard') }}" class="btn btn-sm btn-outline-light"><i class="bi bi-grid-1x2 me-1"></i>Dashboard</a>
    <a href="{{ url_for('setup_plan') }}" class="btn btn-sm btn-outline-light"><i class="bi bi-sliders me-1"></i>Setup</a>
    <a href="{{ url_for('plan_page') }}" class="btn btn-sm btn-light text-primary fw-semibold"><i class="bi bi-diagram-3 me-1"></i>Plan</a>
    <a href="{{ url_for('run_plan') }}" class="btn btn-sm btn-outline-light"><i class="bi bi-play-fill me-1"></i>{{ run_label }}</a>
    <a href="{{ url_for('results_page') }}" class="btn btn-sm btn-outline-light"><i class="bi bi-table me-1"></i>Results</a>
  </div>
</div>
<div class="container py-4" style="max-width:860px;">
  <div class="unsaved-banner d-flex mb-3" id="unsavedBanner"><i class="bi bi-exclamation-triangle-fill"></i><span>You have unsaved changes. Please click <strong>Save Plan</strong> before running.</span></div>
  {% if msg %}
  <div class="alert {% if 'error' in msg.lower() or 'invalid' in msg.lower() or 'failed' in msg.lower() or 'validation' in msg.lower() %}alert-danger{% else %}alert-success{% endif %} d-flex align-items-center gap-2 mb-3"><i class="bi {% if 'error' in msg.lower() or 'invalid' in msg.lower() or 'failed' in msg.lower() or 'validation' in msg.lower() %}bi-exclamation-triangle-fill{% else %}bi-check-circle-fill{% endif %}"></i><div>{{ msg }}</div></div>
  {% endif %}
  <form method="POST" action="/plan" id="planForm">
    <div class="card-panel p-4 mb-4">
      <div class="d-flex align-items-center mb-3"><span class="step-badge">1</span><h5 class="fw-bold mb-0" style="color:#1a3c5e;">Workflow Parameters</h5></div>
      <p class="text-muted mb-3" style="font-size:0.88rem;">Edit core workflow settings. Additional candidate-pool and selection controls appear below.</p>
      <div class="row g-3">
        <div class="col-md-6"><label class="form-label section-label">Target Condition <span class="text-danger">*</span></label><input type="text" class="form-control tracked" name="target_condition" value="{{ plan.target_condition or '' }}" placeholder="e.g. pancreatic_cancer" style="border-radius:8px;"/><div class="field-hint">Use underscores, no spaces.</div></div>
        <div class="col-md-3"><label class="form-label section-label">Risk Horizon (months) <span class="text-danger">*</span></label><input type="number" class="form-control tracked" name="horizon_months" value="{{ plan.horizon_months or '' }}" min="1" max="120" style="border-radius:8px;"/><div class="field-hint">Timeframe for risk assessment.</div></div>
        <div class="col-md-3"><label class="form-label section-label">Review Cadence <span class="text-danger">*</span></label><select class="form-select tracked" name="cadence" style="border-radius:8px;">{% for opt in ['daily','weekly','monthly','per_visit','yearly'] %}<option value="{{ opt }}" {% if plan.constraints.cadence == opt %}selected{% endif %}>{{ opt|capitalize }}</option>{% endfor %}</select><div class="field-hint">How often the workflow runs.</div></div>
        <div class="col-md-3"><label class="form-label section-label">Patients per Review <span class="text-danger">*</span></label><input type="number" class="form-control tracked" name="review_limit" value="{{ plan.constraints.review_limit }}" min="1" max="200" style="border-radius:8px;"/><div class="field-hint">Shortlist size per run.</div></div>
        <div class="col-md-3"><label class="form-label section-label">Deduplication Window (days)</label><input type="number" class="form-control tracked" name="dedup_days" value="{{ plan.constraints.dedup_days }}" min="0" style="border-radius:8px;"/><div class="field-hint">Avoid repeated patients too soon.</div></div>
        <div class="col-md-3"><label class="form-label section-label">Selection Method</label><select class="form-select tracked" name="selection_method" style="border-radius:8px;">{% for opt in ['top_k','threshold','threshold_then_top_k','first_k','random_k'] %}<option value="{{ opt }}" {% if plan.constraints.selection.method == opt %}selected{% endif %}>{{ opt.replace('_',' ')|title }}</option>{% endfor %}</select><div class="field-hint">How candidates are chosen from scores.</div></div>
        <div class="col-md-3"><label class="form-label section-label">Risk Threshold Low (&lt;)</label><input type="number" class="form-control tracked" name="risk_low_lt" step="0.01" min="0" max="1" value="{{ plan.constraints.risk_level_policy.low_lt }}" style="border-radius:8px;"/><div class="field-hint">Below this is low risk.</div></div>
        <div class="col-md-3"><label class="form-label section-label">Risk Threshold Moderate (&lt;)</label><input type="number" class="form-control tracked" name="risk_moderate_lt" step="0.01" min="0" max="1" value="{{ plan.constraints.risk_level_policy.moderate_lt }}" style="border-radius:8px;"/><div class="field-hint">Between low and this is moderate risk.</div></div>
      </div>
    <div id="advancedControlsMount"></div>
    <div id="taskHiddenInputs"></div>
    <div class="card-panel p-4 mb-4">
      <div class="d-flex align-items-center mb-1"><span class="step-badge">2</span><h5 class="fw-bold mb-0" style="color:#1a3c5e;">Workflow Tasks</h5></div>
      <p class="text-muted mb-3" style="font-size:0.88rem;">Drag tasks between Active and Available. Dependency checks prevent invalid combinations.</p>
      <div class="row g-3">
        <div class="col-md-6">
          <div class="d-flex align-items-center mb-2 gap-2"><i class="bi bi-check-circle-fill text-success"></i><span class="fw-semibold" style="color:#145a32;">Active Tasks</span><span class="badge bg-success ms-1" id="activeCount">0</span></div>
          <div class="field-hint mb-2">These tasks run for selected patients.</div>
          <div class="task-zone" id="activeZone" ondragover="onDragOver(event)" ondrop="onDrop(event, 'active')" ondragleave="onDragLeave(event)"><div id="activeEmpty" class="text-center text-muted py-3" style="font-size:0.82rem; display:none;"><i class="bi bi-inbox fs-4 d-block mb-1"></i>No active tasks. Drag tasks here.</div></div>
        </div>
        <div class="col-md-6">
          <div class="d-flex align-items-center mb-2 gap-2"><i class="bi bi-circle text-secondary"></i><span class="fw-semibold" style="color:#6b7a8d;">Available (Inactive) Tasks</span><span class="badge bg-secondary ms-1" id="inactiveCount">0</span></div>
          <div class="field-hint mb-2">Drag a task to Active to include it.</div>
          <div class="task-zone" id="inactiveZone" ondragover="onDragOver(event)" ondrop="onDrop(event, 'inactive')" ondragleave="onDragLeave(event)"><div id="inactiveEmpty" class="text-center text-muted py-3" style="font-size:0.82rem; display:none;"><i class="bi bi-check-all fs-4 d-block mb-1"></i>All tasks are active.</div></div>
        </div>
      </div>
    </div>
    <div class="card-panel p-4 d-flex flex-wrap align-items-center justify-content-between gap-3">
      <div><div class="fw-semibold" style="color:#1a3c5e;">Ready to proceed?</div><div class="field-hint">Save your plan first, then run the workflow.</div></div>
      <div class="d-flex gap-2 flex-wrap"><button type="submit" class="btn btn-primary-facetcare" id="saveBtn"><i class="bi bi-floppy me-1"></i>Save Plan</button><a href="/run" class="btn btn-run" id="runBtn"><i class="bi bi-play-fill me-1"></i>{{ ui_flags.run_label }}</a></div>
    </div>
  </form>
</div>
<script>
const TASK_META = {
  "intake_workflow": { icon: "bi-hospital", desc: "Profiles the clinic workflow from the clinic setup description." },
  "risk_assessment": { icon: "bi-graph-up-arrow", desc: "Scores risk for the target condition from longitudinal notes." },
  "queue_prioritization": { icon: "bi-sort-down", desc: "Prioritizes patients for review using task outputs." },
  "clinician_summary": { icon: "bi-file-earmark-text", desc: "Generates a clinician-ready summary and suggested next steps." },
  "admin_referral": { icon: "bi-send", desc: "Builds a structured referral payload for admin workflow." },
  "patient_instructions": { icon: "bi-person-lines-fill", desc: "Drafts plain-language patient instructions." },
  "results_summary": { icon: "bi-clipboard2-data", desc: "Summarizes relevant labs, imaging, and trends." },
  "transcription": { icon: "bi-mic", desc: "Placeholder audio-to-text task in this demo." },
  "referral_letter": { icon: "bi-envelope-paper", desc: "Drafts a referral letter." },
  "differential_diagnosis": { icon: "bi-search", desc: "Lists differential diagnoses with reasoning." },
  "guideline_comparison": { icon: "bi-journals", desc: "Compares the case with selected guidelines." },
  "followup_gap_detection": { icon: "bi-calendar-x", desc: "Finds follow-up gaps and missed actions." },
  "referral_intake_checklist": { icon: "bi-card-checklist", desc: "Creates an intake checklist for the receiving service." },
  "lab_trend_summary": { icon: "bi-activity", desc: "Summarizes lab trends over time." },
  "care_plan_reconciliation": { icon: "bi-arrow-repeat", desc: "Flags completed, changed, or unresolved care plan items." }
};
const DEPENDS_ON = {{ task_dep_map | tojson }};
const CONSTRAINTS_PAYLOAD = {{ constraints_payload | tojson }};

const INITIAL_TASKS = [
  {% for t in task_rows %}
  { name: {{ t.name | tojson }}, enabled: {{ 'true' if t.enabled else 'false' }}, params: {{ t.params_text | tojson }}, order: {{ loop.index }} },
  {% endfor %}
];

let taskState = {};
let dragSrc = null;
let planSaved = true;
let baselineSnapshot = "";

const TARGET_CONTEXT_TASKS = new Set(["risk_assessment","clinician_summary","admin_referral","patient_instructions","results_summary","referral_letter","differential_diagnosis","guideline_comparison"]);
const RISK_TASKS = new Set(["risk_assessment"]);

function labelizeTask(name) {
  return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function injectAdvancedControls() {
  const mount = document.getElementById('advancedControlsMount');
  if (!mount) return;
  const cp = (CONSTRAINTS_PAYLOAD && CONSTRAINTS_PAYLOAD.candidate_pool) || {};
  const sel = (CONSTRAINTS_PAYLOAD && CONSTRAINTS_PAYLOAD.selection) || {};
  const includeIds = Array.isArray(cp.include_patient_ids) ? cp.include_patient_ids.join('\n') : '';
  mount.innerHTML = `
    <div class="card-panel p-4 mb-4">
      <div class="d-flex align-items-center justify-content-between gap-2 flex-wrap mb-2">
        <div class="d-flex align-items-center">
          <span class="step-badge">3</span>
          <h5 class="fw-bold mb-0" style="color:#1a3c5e;">Advanced Workflow Controls</h5>
        </div>
        <button class="btn btn-sm btn-outline-primary" type="button" data-bs-toggle="collapse" data-bs-target="#planAdvancedCollapse" aria-expanded="false" aria-controls="planAdvancedCollapse">
          <i class="bi bi-sliders2 me-1"></i>Show advanced controls
        </button>
      </div>
      <p class="text-muted mb-2" style="font-size:0.88rem;">Optional controls for pre-filtering, manual review batches, and non-screening workflows.</p>
      <div class="collapse" id="planAdvancedCollapse">
        <div class="border rounded-3 p-3" style="background:#fafcff;border-color:#dbe8fb !important;">
          <div class="row g-3">
            <div class="col-md-4">
              <label class="form-label section-label">Candidate Pool Strategy</label>
              <select class="form-select tracked" name="candidate_pool_strategy" style="border-radius:8px;">
                <option value="all" ${cp.strategy === 'all' ? 'selected' : ''}>All patients</option>
                <option value="recent_notes_only" ${cp.strategy === 'recent_notes_only' ? 'selected' : ''}>Recent notes only</option>
                <option value="keyword_prefilter" ${cp.strategy === 'keyword_prefilter' ? 'selected' : ''}>Keyword prefilter</option>
              </select>
              <div class="field-hint">Choose which patient pool to evaluate before ranking or selection.</div>
            </div>
            <div class="col-md-4">
              <label class="form-label section-label">Max Candidates</label>
              <input type="number" class="form-control tracked" name="max_candidates" min="1" value="${cp.max_candidates ?? ''}" style="border-radius:8px;" />
              <div class="field-hint">Maximum patients to process before shortlist selection.</div>
            </div>
            <div class="col-md-4">
              <label class="form-label section-label">Selection Source Task</label>
              <input type="text" class="form-control tracked" name="selection_source_task" value="${sel.source_task ?? ''}" placeholder="e.g. followup_gap_detection" style="border-radius:8px;" />
              <div class="field-hint">Optional task output used for ranking or selection.</div>
            </div>
            <div class="col-md-8">
              <label class="form-label section-label">Candidate Pool Keywords</label>
              <input type="text" class="form-control tracked" name="candidate_pool_keywords" value="${(cp.keywords || []).join(', ')}" placeholder="e.g. missed appointment, abnormal lab, pending imaging" style="border-radius:8px;" />
              <div class="field-hint">Comma-separated keywords used only for keyword prefilter.</div>
            </div>
            <div class="col-md-2">
              <label class="form-label section-label">Selection K</label>
              <input type="number" class="form-control tracked" name="selection_k" min="1" value="${sel.k ?? 5}" style="border-radius:8px;" />
              <div class="field-hint">Patients to shortlist.</div>
            </div>
            <div class="col-md-2">
              <label class="form-label section-label">Selection Threshold</label>
              <input type="number" class="form-control tracked" name="selection_threshold" step="0.01" min="0" max="1" value="${sel.threshold ?? ''}" style="border-radius:8px;" />
              <div class="field-hint">Used by threshold methods.</div>
            </div>
            <div class="col-md-12">
              <label class="form-label section-label">Manual Include-List</label>
              <textarea class="form-control tracked" name="manual_include_patient_ids" rows="3" placeholder="P0001, P0007, P0042 or one ID per line" style="border-radius:8px;">${includeIds}</textarea>
              <div class="field-hint">If provided, FacetCare restricts this run to the listed patient IDs. Useful for follow-up gap review batches or manual audit lists.</div>
            </div>
          </div>
        </div>
      </div>
    </div>`;
}

function buildHiddenInputs() {
  const container = document.getElementById('taskHiddenInputs');
  container.innerHTML = '';
  Object.entries(taskState).forEach(([name, state], idx) => {
    const add = (n, v) => {
      const inp = document.createElement('input');
      inp.type = 'hidden';
      inp.name = n;
      inp.value = v;
      container.appendChild(inp);
    };
    if (state.enabled) add(`enabled__${name}`, 'on');
    add(`params__${name}`, state.params || '{}');
    add(`order__${name}`, state.order || idx + 1);
  });
}

function serializeFormState() {
  const form = document.getElementById('planForm');
  if (!form) return '';
  buildHiddenInputs();
  const fd = new FormData(form);
  const pairs = [];
  for (const [k, v] of fd.entries()) pairs.push([k, String(v)]);
  pairs.sort((a, b) => (a[0] + '::' + a[1]).localeCompare(b[0] + '::' + b[1]));
  return JSON.stringify(pairs);
}

function applyDirtyState(isDirty) {
  planSaved = !isDirty;
  const banner = document.getElementById('unsavedBanner');
  if (banner) banner.style.display = isDirty ? 'flex' : 'none';
  updateRunButton();
}

function refreshDirtyState() {
  applyDirtyState(serializeFormState() !== baselineSnapshot);
}

function getActiveNames() {
  return Object.entries(taskState)
    .filter(([, v]) => v.enabled)
    .sort((a, b) => a[1].order - b[1].order)
    .map(([k]) => k);
}

function getInactiveNames() {
  return Object.entries(taskState)
    .filter(([, v]) => !v.enabled)
    .sort((a, b) => a[1].order - b[1].order)
    .map(([k]) => k);
}

function setFieldVisibility(fieldName, show) {
  const el = document.querySelector(`[name="${fieldName}"]`);
  if (!el) return;
  const wrapper = el.closest('.col-md-2, .col-md-3, .col-md-4, .col-md-6, .col-md-8, .col-md-12') || el.parentElement;
  if (wrapper) wrapper.style.display = show ? '' : 'none';
}

function updateAdaptiveUi() {
  const active = new Set(getActiveNames());
  const needsRisk = [...active].some(n => RISK_TASKS.has(n));
  const needsTarget = [...active].some(n => TARGET_CONTEXT_TASKS.has(n));
  const strategy = document.querySelector('[name="candidate_pool_strategy"]')?.value || 'all';
  const selMethod = document.querySelector('[name="selection_method"]')?.value || 'top_k';

  setFieldVisibility('target_condition', needsTarget);
  setFieldVisibility('horizon_months', needsTarget);
  setFieldVisibility('risk_low_lt', needsRisk);
  setFieldVisibility('risk_moderate_lt', needsRisk);
  setFieldVisibility('selection_threshold', selMethod.includes('threshold'));
  setFieldVisibility('candidate_pool_keywords', strategy === 'keyword_prefilter');

  const headings = document.querySelectorAll('.card-panel h5');
  if (headings.length) headings[0].textContent = needsRisk ? 'Screening Parameters' : 'Workflow Parameters';

  const runBtn = document.getElementById('runBtn');
  if (runBtn) {
    let label = 'Run Plan';
    if (needsRisk) label = 'Run Screening';
    else if (active.has('followup_gap_detection')) label = 'Run Follow-up Review';
    else if (active.has('queue_prioritization')) label = 'Run Queue Review';
    runBtn.innerHTML = `<i class="bi bi-play-fill me-1"></i>${label}`;
  }
}

function renderAllTasks() {
  const az = document.getElementById('activeZone');
  const iz = document.getElementById('inactiveZone');
  az.querySelectorAll('.task-card').forEach(el => el.remove());
  iz.querySelectorAll('.task-card').forEach(el => el.remove());

  const active = getActiveNames();
  const inactive = getInactiveNames();
  active.forEach(name => az.appendChild(makeCard(name, true)));
  inactive.forEach(name => iz.appendChild(makeCard(name, false)));

  document.getElementById('activeCount').textContent = active.length;
  document.getElementById('inactiveCount').textContent = inactive.length;
  document.getElementById('activeEmpty').style.display = active.length ? 'none' : 'block';
  document.getElementById('inactiveEmpty').style.display = inactive.length ? 'none' : 'block';

  buildHiddenInputs();
  updateAdaptiveUi();
}

function makeCard(name, enabled) {
  const meta = TASK_META[name] || { icon: 'bi-gear', desc: 'Custom task.' };
  const deps = DEPENDS_ON[name] || [];
  const unmet = deps.filter(d => !taskState[d] || !taskState[d].enabled);

  const card = document.createElement('div');
  card.className = 'task-card' + (enabled && unmet.length > 0 ? ' border-warning' : '');
  card.draggable = true;
  card.dataset.name = name;

  const depsHtml = deps.length ? deps.map(d => `<span class="task-badge-dep"><i class="bi bi-arrow-return-right"></i> ${d}</span>`).join('') : '';
  const unmetHtml = (enabled && unmet.length > 0) ? `<div class="dep-error" style="display:block;"><i class="bi bi-exclamation-triangle-fill me-1"></i>Requires: ${unmet.join(', ')}</div>` : '';
  const paramsDisplay = taskState[name].params && taskState[name].params !== '{}' ? taskState[name].params : '{}';

  card.innerHTML = `
    <i class="bi ${meta.icon} task-icon" style="color:#2563a8;"></i>
    <div class="flex-grow-1">
      <div class="d-flex align-items-center gap-2"><span class="task-name">${labelizeTask(name)}</span><span class="badge ${enabled ? 'bg-success' : 'bg-secondary'}" style="font-size:0.68rem;">${enabled ? 'Active' : 'Inactive'}</span></div>
      <div class="task-desc">${meta.desc}</div>${depsHtml ? `<div class="mt-1">${depsHtml}</div>` : ''}${unmetHtml}
      <div class="params-toggle" onclick="toggleParams('${name}')"><i class="bi bi-sliders2 me-1"></i>Advanced parameters</div>
      <div class="params-box" id="params-${name}"><textarea class="form-control" rows="3" placeholder='{"key": "value"}' onchange="updateParams('${name}', this.value)" oninput="markUnsaved()">${paramsDisplay}</textarea><div class="field-hint">Optional JSON key-value pairs for this task. Leave as <code>{}</code> unless needed.</div></div>
    </div>
    <i class="bi bi-grip-vertical text-secondary" style="font-size:1.1rem; margin-top:2px; cursor:grab;"></i>`;

  card.addEventListener('dragstart', e => {
    dragSrc = name;
    setTimeout(() => card.classList.add('dragging'), 0);
    e.dataTransfer.effectAllowed = 'move';
  });
  card.addEventListener('dragend', () => card.classList.remove('dragging'));
  return card;
}

function toggleParams(name) {
  const box = document.getElementById('params-' + name);
  box.style.display = box.style.display === 'block' ? 'none' : 'block';
}

function updateParams(name, val) {
  taskState[name].params = val;
  buildHiddenInputs();
  refreshDirtyState();
}

function onDragOver(e) { e.preventDefault(); e.currentTarget.classList.add('drag-over'); }
function onDragLeave(e) { e.currentTarget.classList.remove('drag-over'); }

function onDrop(e, zone) {
  e.preventDefault();
  e.currentTarget.classList.remove('drag-over');
  if (!dragSrc) return;
  const name = dragSrc;
  const toActive = zone === 'active';

  if (toActive) {
    const unmet = (DEPENDS_ON[name] || []).filter(d => !taskState[d] || !taskState[d].enabled);
    if (unmet.length) {
      showDepToast(name, unmet);
      dragSrc = null;
      return;
    }
  }

  if (!toActive) {
    const dependents = Object.entries(DEPENDS_ON)
      .filter(([k, deps]) => deps.includes(name) && taskState[k] && taskState[k].enabled)
      .map(([k]) => k);
    if (dependents.length) {
      showDepToast(name, [], dependents);
      dragSrc = null;
      return;
    }
  }

  taskState[name].enabled = toActive;
  const siblings = toActive ? getActiveNames().filter(n => n !== name) : getInactiveNames().filter(n => n !== name);
  taskState[name].order = siblings.length + 1;
  dragSrc = null;
  renderAllTasks();
  refreshDirtyState();
}

function showDepToast(name, unmet, dependents) {
  const existing = document.getElementById('depToast');
  if (existing) existing.remove();
  const label = labelizeTask(name);
  let msg;
  if (unmet && unmet.length) msg = `<strong>${label}</strong> requires: <strong>${unmet.join(', ')}</strong>.`;
  else msg = `Cannot deactivate <strong>${label}</strong> because these active tasks depend on it: <strong>${dependents.join(', ')}</strong>.`;
  const toast = document.createElement('div');
  toast.id = 'depToast';
  toast.className = 'alert alert-warning d-flex align-items-center gap-2 position-fixed';
  toast.style.cssText = 'bottom:1.5rem;left:50%;transform:translateX(-50%);z-index:9999;min-width:340px;max-width:580px;box-shadow:0 4px 18px rgba(0,0,0,0.15);';
  toast.innerHTML = `<i class="bi bi-exclamation-triangle-fill"></i><div>${msg}</div>`;
  document.body.appendChild(toast);
  setTimeout(() => { if (toast.parentNode) toast.remove(); }, 4000);
}

function markUnsaved() {
  updateAdaptiveUi();
  refreshDirtyState();
}

function updateRunButton() {
  const btn = document.getElementById('runBtn');
  if (!btn) return;
  if (!planSaved) {
    btn.classList.add('disabled');
    btn.setAttribute('aria-disabled', 'true');
    btn.onclick = e => {
      e.preventDefault();
      document.getElementById('unsavedBanner').scrollIntoView({ behavior: 'smooth' });
    };
  } else {
    btn.classList.remove('disabled');
    btn.removeAttribute('aria-disabled');
    btn.onclick = null;
  }
}

function init() {
  INITIAL_TASKS.forEach(t => {
    taskState[t.name] = { enabled: t.enabled, params: t.params, order: t.order };
  });
  injectAdvancedControls();
  renderAllTasks();
  baselineSnapshot = serializeFormState();
  applyDirtyState(false);
}

document.getElementById('planForm').addEventListener('change', markUnsaved);
document.getElementById('planForm').addEventListener('input', markUnsaved);
document.getElementById('planForm').addEventListener('submit', () => {
  buildHiddenInputs();
  baselineSnapshot = serializeFormState();
  applyDirtyState(false);
});

init();
</script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    """
    return render_template_string(tpl, plan=plan, plan_json=plan_json, task_rows=task_rows, task_dep_map=task_dep_map, ui_flags=ui_flags, constraints_payload=constraints_payload, msg=msg, run_label=run_label)


def _background_run(plan: ClinicPlanSchema):
    global current_results, run_state
    try:
        with _lock:
            run_state = {"status": "running", "message": "Run in progress...", "started_at": time.time()}
            _persist_state_locked()
        patients = load_sample_patients()
        runner = _make_runner()
        bundle = runner.run_weekly_review(patients=patients, plan=plan, run_date=date.today().isoformat())
        with _lock:
            current_results = bundle
            run_state = {"status": "done", "message": f"Run completed. Selected {len(bundle.selected)} patients."}
            _persist_state_locked()
    except Exception as e:
        with _lock:
            run_state = {"status": "error", "message": str(e)}
            _persist_state_locked()


@app.route("/run")
def run_plan():
    with _lock:
        plan = _ensure_plan_tasks_catalog(current_plan) if current_plan is not None else None
        if plan is not None and current_plan is not None and len(plan.tasks) != len(current_plan.tasks):
            current_plan = plan
            _persist_state_locked()
        status = run_state.get("status")
    if plan is None:
        return redirect(url_for("setup_plan"))
    if status == "running":
        return redirect(url_for("results_page"))
    t = threading.Thread(target=_background_run, args=(plan,), daemon=True)
    t.start()
    return redirect(url_for("results_page"))


@app.route("/results", methods=["GET", "POST"])
def results_page():
    global current_results, current_plan
    msg = ""
    msg_level = "success"

    task_map = {
        "generate_clinician_summary": "clinician_summary",
        "generate_admin_referral": "admin_referral",
        "generate_referral_letter": "referral_letter",
        "generate_patient_instructions": "patient_instructions",
        "generate_lab_trend_summary": "lab_trend_summary",
        "generate_followup_gap": "followup_gap_detection",
    }

    if request.method == "POST":
        action = (request.form.get("action", "") or "").strip()
        patient_id = (request.form.get("patient_id", "") or "").strip()
        force_refresh = request.form.get("force_refresh") == "1"

        with _lock:
            results_for_action = current_results

        if action == "bulk_apply":
            bulk_action = (request.form.get("bulk_action", "") or "").strip()
            selected_ids = [x.strip() for x in request.form.getlist("selected_patient_ids") if x.strip()]
            seen = set()
            selected_ids = [x for x in selected_ids if not (x in seen or seen.add(x))]
            if not selected_ids:
                msg = "Select at least one patient in the queue first."
                msg_level = "warning"
            elif results_for_action is None:
                msg = "No results available yet."
                msg_level = "warning"
            elif bulk_action in {"mark_reviewed", "mark_unreviewed"}:
                reviewed_flag = bulk_action == "mark_reviewed"
                for pid in selected_ids:
                    _set_reviewed_patient(results_for_action, pid, reviewed_flag)
                msg = f"Updated review status for {len(selected_ids)} patient(s)."
            elif bulk_action.startswith("run:"):
                task_name = bulk_action.split(":", 1)[1]
                ok = 0
                errs: List[str] = []
                for pid in selected_ids:
                    err = _run_single_task_action(pid, task_name, force_refresh=force_refresh)
                    if err:
                        errs.append(f"{pid}: {err}")
                    else:
                        ok += 1
                if errs:
                    msg_level = "warning" if ok else "danger"
                    preview = "; ".join(errs[:3])
                    more = f" (+{len(errs)-3} more)" if len(errs) > 3 else ""
                    msg = f"Bulk task {task_name}: {ok} succeeded, {len(errs)} failed. {preview}{more}"
                else:
                    msg = f"Bulk task {task_name} completed for {ok} patient(s)."
            else:
                msg = "Choose a bulk action to apply."
                msg_level = "warning"

        elif action in {"mark_reviewed", "mark_unreviewed", "toggle_reviewed"}:
            if results_for_action is None or not patient_id:
                msg = "No patient selected."
                msg_level = "warning"
            else:
                if action == "toggle_reviewed":
                    new_flag = not _is_reviewed_patient(results_for_action, patient_id)
                else:
                    new_flag = action == "mark_reviewed"
                _set_reviewed_patient(results_for_action, patient_id, new_flag)
                msg = f"{patient_id} marked as {'reviewed' if new_flag else 'not reviewed'}."

        elif action and patient_id:
            task_name = task_map.get(action)
            if task_name:
                err = _run_single_task_action(patient_id, task_name, force_refresh=force_refresh)
                if err:
                    msg = err
                    msg_level = "danger"
                else:
                    msg = f"Ran {task_name} for {patient_id}."
            else:
                msg = "Unknown action."
                msg_level = "warning"

    with _lock:
        plan = _ensure_plan_tasks_catalog(current_plan) if current_plan is not None else None
        if plan is not None and current_plan is not None and len(plan.tasks) != len(current_plan.tasks):
            current_plan = plan
            _persist_state_locked()
        results = current_results
        status = dict(run_state)

    auto_refresh = status.get("status") == "running"

    view_mode = (request.args.get("view") or "all").lower().strip()
    if view_mode not in {"all", "reviewed", "unreviewed"}:
        view_mode = "all"
    q = (request.args.get("q") or "").strip().lower()

    reviewed_map = _review_map_for_results(results) if results else {}
    cache_stats = _cache_summary()

    selected_bundles = list(getattr(results, "selected", []) or [])
    selected_bundles.sort(key=lambda b: (_is_reviewed_patient(results, b.patient_id), -_risk_prob(b), b.patient_id) if results else (False, 0, ""))

    reviewed_count = 0
    filtered_bundles = []
    for b in selected_bundles:
        is_rev = bool(reviewed_map.get(b.patient_id, False))
        if is_rev:
            reviewed_count += 1
        if view_mode == "reviewed" and not is_rev:
            continue
        if view_mode == "unreviewed" and is_rev:
            continue
        if q and q not in b.patient_id.lower():
            continue
        filtered_bundles.append(b)

    patient_lookup = {p.patient_id: p for p in load_sample_patients()}

    tpl = """
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\"/>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>
  <title>FacetCare — Results</title>
  {% if auto_refresh %}<meta http-equiv=\"refresh\" content=\"3\">{% endif %}
  <link href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css\" rel=\"stylesheet\"/>
  <link href=\"https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css\" rel=\"stylesheet\"/>
  <style>
    body { background-color: #f0f4f8; font-family: 'Segoe UI', sans-serif; }
    .brand-bar { background: linear-gradient(90deg, #1a3c5e 0%, #2563a8 100%); padding: 1.1rem 2rem; color: white; }
    .brand-title { font-size: 1.6rem; font-weight: 700; letter-spacing: 0.03em; }
    .brand-subtitle { font-size: 0.88rem; opacity: 0.85; }
    .card-panel { border: none; border-radius: 12px; box-shadow: 0 4px 24px rgba(30,60,100,0.10); background: white; }
    .metric-card { border-radius: 12px; border: 1px solid #dbe7f5; background: linear-gradient(180deg, #ffffff, #f8fbff); }
    .metric-label { color: #6b7a8d; font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }
    .metric-value { color: #1a3c5e; font-size: 1.25rem; font-weight: 700; }
    .queue-table th { font-size: 0.76rem; text-transform: uppercase; color: #6b7a8d; letter-spacing: 0.05em; }
    .queue-table td { vertical-align: middle; font-size: 0.88rem; }
    .patient-card { border: 1px solid #dde6f0; border-radius: 12px; background: #fff; }
    .patient-card.reviewed { border-left: 6px solid #198754; }
    .patient-card.unreviewed { border-left: 6px solid #f0ad4e; }
    .artifact-pill { display:inline-block; padding:0.15rem 0.5rem; border-radius:999px; font-size:0.72rem; margin-right:0.25rem; margin-bottom:0.25rem; border:1px solid #cfe0f7; background:#eaf2ff; color:#1a4080; }
    .artifact-pill.done { border-color:#cde9d6; background:#e9f8ee; color:#17633a; }
    .mono-pre { background:#f8fafc; border:1px solid #e3eaf2; border-radius:8px; padding:0.65rem; white-space:pre-wrap; font-size:0.80rem; }
    .section-title { color:#1a3c5e; font-weight:700; }
    .queue-toolbar .form-select, .queue-toolbar .form-control { border-radius: 8px; }
    .btn-facet { background: linear-gradient(90deg, #1a3c5e, #2563a8); color: white; border: none; }
    .btn-facet:hover { color: white; opacity: 0.92; }
    .small-muted { color:#6b7a8d; font-size:0.82rem; }
    .sticky-tools { position: sticky; top: 0.5rem; z-index: 20; }
  </style>
</head>
<body>
<div class=\"brand-bar d-flex align-items-center gap-3\">
  <i class=\"bi bi-heart-pulse-fill fs-3\"></i>
  <div>
    <a class=\"brand-title\" href=\"{{ url_for('dashboard') }}\" style=\"color:white;text-decoration:none;\">FacetCare</a>
    <div class=\"brand-subtitle\">AI-Assisted Clinical Workflow Orchestration</div>
  </div>
  <div class=\"ms-auto d-flex gap-2 flex-wrap\">
    <a href=\"{{ url_for('dashboard') }}\" class=\"btn btn-sm btn-outline-light\"><i class=\"bi bi-grid-1x2 me-1\"></i>Dashboard</a>
    <a href=\"{{ url_for('setup_plan') }}\" class=\"btn btn-sm btn-outline-light\"><i class=\"bi bi-sliders me-1\"></i>Setup</a>
    <a href=\"{{ url_for('plan_page') }}\" class=\"btn btn-sm btn-outline-light\"><i class=\"bi bi-diagram-3 me-1\"></i>Plan</a>
    {% if plan %}<a href=\"{{ url_for('run_plan') }}\" class=\"btn btn-sm btn-outline-light\"><i class=\"bi bi-play-fill me-1\"></i>{{ run_label }}</a>{% endif %}
    <a href=\"{{ url_for('results_page') }}\" class=\"btn btn-sm btn-light text-primary fw-semibold\"><i class=\"bi bi-table me-1\"></i>Results</a>
  </div>
</div>

<div class=\"container py-4\" style=\"max-width:1100px;\">

  {% if msg %}
  <div class=\"alert alert-{{ msg_level }} d-flex align-items-center gap-2 mb-3\" role=\"alert\">
    <i class=\"bi {% if msg_level in ['danger','warning'] %}bi-exclamation-triangle-fill{% else %}bi-check-circle-fill{% endif %}\"></i>
    <div>{{ msg }}</div>
  </div>
  {% endif %}

  <div class=\"card-panel p-3 p-md-4 mb-3\">
    <div class=\"d-flex flex-wrap align-items-center justify-content-between gap-2\">
      <div>
        <div class=\"section-title\">Run status</div>
        <div class=\"small-muted\">{{ status.get('status') }} • {{ status.get('message') }}</div>
        {% if auto_refresh %}<div class=\"small-muted\"><i class=\"bi bi-arrow-repeat me-1\"></i>Auto-refresh every 3 seconds while the run is active.</div>{% endif %}
      </div>
      {% if plan %}
      <div class=\"small-muted text-md-end\">
        <div><strong>Target:</strong> {{ plan.target_condition or 'N/A' }}{% if plan.horizon_months %} • {{ plan.horizon_months }} months{% endif %}</div>
        <div><strong>Cadence:</strong> {{ plan.constraints.cadence }} • <strong>Review limit:</strong> {{ plan.constraints.review_limit }}</div>
      </div>
      {% endif %}
    </div>
  </div>

  <div class=\"row g-3 mb-3\">
    <div class=\"col-6 col-md-3\"><div class=\"metric-card p-3\"><div class=\"metric-label\">Selected</div><div class=\"metric-value\">{{ (results.selected|length) if results else 0 }}</div></div></div>
    <div class=\"col-6 col-md-3\"><div class=\"metric-card p-3\"><div class=\"metric-label\">Reviewed</div><div class=\"metric-value\">{{ reviewed_count if results else 0 }}</div></div></div>
    <div class=\"col-6 col-md-3\"><div class=\"metric-card p-3\"><div class=\"metric-label\">Unreviewed</div><div class=\"metric-value\">{{ ((results.selected|length) - reviewed_count) if results else 0 }}</div></div></div>
    <div class=\"col-6 col-md-3\"><div class=\"metric-card p-3\"><div class=\"metric-label\">Cache</div><div class=\"metric-value\">{{ cache_stats.get('entries',0) }}</div><div class=\"small-muted\">H {{ cache_stats.get('hits',0) }} • M {{ cache_stats.get('misses',0) }}</div></div></div>
  </div>

  {% if results %}

  <div class=\"card-panel p-3 p-md-4 mb-3 sticky-tools\">
    <div class=\"d-flex flex-wrap align-items-end justify-content-between gap-2 mb-2 queue-toolbar\">
      <form method=\"get\" class=\"d-flex flex-wrap align-items-end gap-2\">
        <div>
          <label class=\"small-muted d-block mb-1\">Filter</label>
          <select name=\"view\" class=\"form-select form-select-sm\">
            <option value=\"all\" {% if view_mode == 'all' %}selected{% endif %}>All</option>
            <option value=\"unreviewed\" {% if view_mode == 'unreviewed' %}selected{% endif %}>Unreviewed</option>
            <option value=\"reviewed\" {% if view_mode == 'reviewed' %}selected{% endif %}>Reviewed</option>
          </select>
        </div>
        <div>
          <label class=\"small-muted d-block mb-1\">Search patient</label>
          <input type=\"text\" name=\"q\" value=\"{{ request.args.get('q','') }}\" class=\"form-control form-control-sm\" placeholder=\"e.g. P0007\"/>
        </div>
        <button type=\"submit\" class=\"btn btn-sm btn-outline-primary\"><i class=\"bi bi-funnel me-1\"></i>Apply</button>
        <a href=\"{{ url_for('results_page') }}\" class=\"btn btn-sm btn-outline-secondary\">Reset</a>
      </form>
      <div class=\"small-muted\">Showing {{ filtered_bundles|length }} of {{ results.selected|length }} selected patients</div>
    </div>

    <form method=\"post\" id=\"bulkForm\">
      <input type=\"hidden\" name=\"action\" value=\"bulk_apply\"/>
      <div class=\"row g-2 align-items-end mb-2\">
        <div class=\"col-md-4\">
          <label class=\"small-muted d-block mb-1\">Bulk action on selected rows</label>
          <select name=\"bulk_action\" class=\"form-select form-select-sm\">
            <option value=\"\">Choose action...</option>
            <option value=\"mark_reviewed\">Mark reviewed</option>
            <option value=\"mark_unreviewed\">Mark not reviewed</option>
            <option value=\"run:clinician_summary\">Generate clinician summary</option>
            <option value=\"run:admin_referral\">Generate admin referral</option>
            <option value=\"run:referral_letter\">Generate referral letter</option>
            <option value=\"run:patient_instructions\">Generate patient instructions</option>
            <option value=\"run:lab_trend_summary\">Generate lab trend summary</option>
            <option value=\"run:followup_gap_detection\">Run follow-up gap detection</option>
          </select>
        </div>
        <div class=\"col-md-3\">
          <label class=\"small-muted d-block mb-1\">Options</label>
          <div class=\"form-check small\">
            <input class=\"form-check-input\" type=\"checkbox\" name=\"force_refresh\" value=\"1\" id=\"bulkForce\"/>
            <label class=\"form-check-label\" for=\"bulkForce\">Force refresh if output exists</label>
          </div>
        </div>
        <div class=\"col-md-5 d-flex flex-wrap gap-2 justify-content-md-end\">
          <button type=\"button\" class=\"btn btn-sm btn-outline-secondary\" onclick=\"toggleAllRows(true)\">Select all visible</button>
          <button type=\"button\" class=\"btn btn-sm btn-outline-secondary\" onclick=\"toggleAllRows(false)\">Clear</button>
          <button type=\"submit\" class=\"btn btn-sm btn-facet\"><i class=\"bi bi-lightning-charge me-1\"></i>Apply</button>
        </div>
      </div>

      <div class=\"table-responsive\">
        <table class=\"table table-sm align-middle queue-table\">
          <thead>
            <tr>
              <th style=\"width:36px;\"><input type=\"checkbox\" class=\"form-check-input\" onclick=\"toggleAllRows(this.checked)\"/></th>
              <th>Patient</th>
              <th>Reviewed</th>
              <th>Risk</th>
              <th>Artifacts</th>
              <th>Queue</th>
            </tr>
          </thead>
          <tbody>
            {% for b in filtered_bundles %}
            {% set flags = _bundle_artifact_flags(b) %}
            {% set reviewed = reviewed_map.get(b.patient_id, False) %}
            <tr>
              <td><input type=\"checkbox\" class=\"form-check-input row-check\" name=\"selected_patient_ids\" value=\"{{ b.patient_id }}\"/></td>
              <td>
                <a href=\"#patient-{{ b.patient_id }}\" class=\"fw-semibold text-decoration-none\">{{ b.patient_id }}</a>
                <div class=\"small-muted\">{{ _truncate_text(b.selection_reason or '', 70) }}</div>
              </td>
              <td>
                {% if reviewed %}
                  <span class=\"badge bg-success\">Reviewed</span>
                {% else %}
                  <span class=\"badge bg-warning text-dark\">Pending</span>
                {% endif %}
              </td>
              <td>
                {% if b.risk %}
                  <div><span class=\"badge {% if (b.risk.risk_level or '').lower() == 'high' %}bg-danger{% elif (b.risk.risk_level or '').lower() == 'moderate' %}bg-warning text-dark{% else %}bg-secondary{% endif %}\">{{ b.risk.risk_level }}</span></div>
                  <div class=\"small-muted\">{{ '%.3f'|format(b.risk.risk_probability) }}</div>
                {% else %}
                  <span class=\"badge bg-light text-dark border\">Not scored</span>
                {% endif %}
              </td>
              <td>
                {% for label, key in [('Risk','risk'),('Summary','clinician_summary'),('Referral','admin_referral'),('Letter','referral_letter'),('Instructions','patient_instructions')] %}
                  <span class=\"artifact-pill {% if flags.get(key) %}done{% endif %}\">{{ label }}</span>
                {% endfor %}
              </td>
              <td>
                <a href=\"#patient-{{ b.patient_id }}\" class=\"btn btn-sm btn-outline-primary\">Open</a>
              </td>
            </tr>
            {% else %}
            <tr><td colspan=\"6\" class=\"text-center text-muted py-3\">No patients match the current filter.</td></tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    </form>
  </div>

  <div class=\"d-flex align-items-center justify-content-between mb-2\">
    <div class=\"section-title\">Patient review cards</div>
    <div class=\"small-muted\">Open a card, generate missing artifacts, then mark reviewed.</div>
  </div>

  {% for b in filtered_bundles %}
  {% set reviewed = reviewed_map.get(b.patient_id, False) %}
  {% set flags = _bundle_artifact_flags(b) %}
  {% set note = patient_lookup.get(b.patient_id) %}
  <div id=\"patient-{{ b.patient_id }}\" class=\"card-panel patient-card {{ 'reviewed' if reviewed else 'unreviewed' }} p-3 p-md-4 mb-3\">
    <div class=\"d-flex flex-wrap align-items-start justify-content-between gap-2\">
      <div>
        <div class=\"d-flex align-items-center gap-2 flex-wrap\">
          <h5 class=\"mb-0 section-title\">{{ b.patient_id }}</h5>
          {% if reviewed %}<span class=\"badge bg-success\">Reviewed</span>{% else %}<span class=\"badge bg-warning text-dark\">Needs review</span>{% endif %}
          {% if b.risk %}
            <span class=\"badge {% if (b.risk.risk_level or '').lower() == 'high' %}bg-danger{% elif (b.risk.risk_level or '').lower() == 'moderate' %}bg-warning text-dark{% else %}bg-secondary{% endif %}\">{{ b.risk.risk_level }}</span>
            <span class=\"badge bg-light text-dark border\">{{ '%.3f'|format(b.risk.risk_probability) }}</span>
          {% endif %}
        </div>
        <div class=\"small-muted mt-1\">{{ b.selection_reason }}</div>
      </div>
      <div class=\"d-flex gap-1 flex-wrap\">
        <form method=\"post\" class=\"d-inline\">
          <input type=\"hidden\" name=\"patient_id\" value=\"{{ b.patient_id }}\"/>
          <input type=\"hidden\" name=\"action\" value=\"{{ 'mark_unreviewed' if reviewed else 'mark_reviewed' }}\"/>
          <button class=\"btn btn-sm {% if reviewed %}btn-outline-warning{% else %}btn-outline-success{% endif %}\" type=\"submit\">{{ 'Mark not reviewed' if reviewed else 'Mark reviewed' }}</button>
        </form>
        <a href=\"#top\" class=\"btn btn-sm btn-outline-secondary\" onclick=\"window.scrollTo({top:0,behavior:'smooth'}); return false;\">Top</a>
      </div>
    </div>

    <div class=\"mt-2\">
      {% for label, key in [('Risk','risk'),('Summary','clinician_summary'),('Admin referral','admin_referral'),('Referral letter','referral_letter'),('Patient instructions','patient_instructions'),('Lab trend','lab_trend_summary'),('Follow-up gap','followup_gap_detection')] %}
        <span class=\"artifact-pill {% if flags.get(key) %}done{% endif %}\">{{ label }}</span>
      {% endfor %}
    </div>

    <details class=\"mt-2\">
      <summary class=\"small-muted\">Source note preview</summary>
      <div class=\"mono-pre mt-2\">{{ (note.longitudinal_notes if note else 'Note not found in sample dataset.') }}</div>
    </details>

    <div class=\"card mt-3 border-0\" style=\"background:#f8fbff;\">
      <div class=\"card-body py-2 px-2 px-md-3\">
        <div class=\"small-muted mb-2\">Generate or refresh artifacts for this patient</div>
        <div class=\"d-flex flex-wrap gap-2\">
          {% for action_name, label in [
            ('generate_clinician_summary','Clinician summary'),
            ('generate_admin_referral','Admin referral'),
            ('generate_referral_letter','Referral letter'),
            ('generate_patient_instructions','Patient instructions'),
            ('generate_lab_trend_summary','Lab trend summary'),
            ('generate_followup_gap','Follow-up gap')
          ] %}
          <form method=\"post\" class=\"d-inline-flex align-items-center gap-1\">
            <input type=\"hidden\" name=\"patient_id\" value=\"{{ b.patient_id }}\"/>
            <input type=\"hidden\" name=\"action\" value=\"{{ action_name }}\"/>
            <button class=\"btn btn-sm btn-outline-primary\" type=\"submit\">{{ label }}</button>
            <label class=\"form-check-label small text-muted ms-1\" style=\"font-size:0.72rem;\">
              <input class=\"form-check-input me-1\" type=\"checkbox\" name=\"force_refresh\" value=\"1\"/>refresh
            </label>
          </form>
          {% endfor %}
        </div>
      </div>
    </div>

    {% if b.risk %}
    <div class=\"mt-3\">
      <div class=\"fw-semibold\" style=\"color:#1a3c5e;\">Risk assessment</div>
      <div class=\"small-muted\">{{ b.risk.target_condition }} • {{ b.risk.horizon_months }} months</div>
      <div class=\"mono-pre mt-1\">{{ b.risk | tojson(indent=2) }}</div>
    </div>
    {% endif %}

    {% if b.clinician_summary %}
    <div class=\"mt-3\">
      <div class=\"fw-semibold\" style=\"color:#1a3c5e;\">Clinician summary</div>
      <div class=\"mono-pre mt-1\">{{ b.clinician_summary | tojson(indent=2) }}</div>
    </div>
    {% endif %}

    {% if b.admin_referral %}
    <div class=\"mt-3\">
      <div class=\"fw-semibold\" style=\"color:#1a3c5e;\">Admin referral</div>
      <div class=\"mono-pre mt-1\">{{ b.admin_referral | tojson(indent=2) }}</div>
    </div>
    {% endif %}

    {% if b.extra_outputs %}
    <details class=\"mt-3\">
      <summary class=\"fw-semibold\" style=\"color:#1a3c5e;\">Additional outputs ({{ b.extra_outputs|length }})</summary>
      <div class=\"mono-pre mt-2\">{{ b.extra_outputs | tojson(indent=2) }}</div>
    </details>
    {% endif %}
  </div>
  {% endfor %}

  {% else %}
  <div class=\"card-panel p-4 text-center\">
    <div class=\"section-title mb-1\">No results yet</div>
    <div class=\"small-muted mb-3\">Generate a plan, run the workflow, then come back here to review the queue.</div>
    <a href=\"{{ url_for('plan_page') }}\" class=\"btn btn-facet\"><i class=\"bi bi-play-fill me-1\"></i>Open plan and run</a>
  </div>
  {% endif %}
</div>

<script>
function toggleAllRows(flag) {
  document.querySelectorAll('.row-check').forEach(cb => { cb.checked = !!flag; });
}
</script>
<script src=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js\"></script>
</body>
</html>
    """
    resp = make_response(
        render_template_string(
            tpl,
            plan=plan,
            results=results,
            status=status,
            auto_refresh=auto_refresh,
            msg=msg,
            msg_level=msg_level,
            reviewed_map=reviewed_map,
            reviewed_count=reviewed_count,
            filtered_bundles=filtered_bundles,
            patient_lookup=patient_lookup,
            view_mode=view_mode,
            cache_stats=cache_stats,
            _bundle_artifact_flags=_bundle_artifact_flags,
            _truncate_text=_truncate_text,
            request=request,
            run_label=_run_button_label(plan) if plan else "Run Workflow",
        )
    )
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp


# Backward-compatible alias used in some earlier patches
@app.route("/edit_plan", methods=["GET", "POST"])
def edit_plan_alias():
    return plan_page()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
