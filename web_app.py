from __future__ import annotations

import json
import os
import random
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, redirect, render_template_string, request, url_for, make_response

from medflow.llm_client import LLMJsonClient
from medflow.plan_builder import build_clinic_plan_from_description
from medflow.runner import TaskRunner
from medflow.schemas import ClinicPlanSchema, PatientRecord
from medflow.tasks import TaskContext, default_task_registry

from openai import OpenAI

ENDPOINT = "http://node159:61424/v1"
CLIENT = OpenAI(base_url=ENDPOINT, api_key="")
MODEL = "medgemma-1.5-4b-it"  # Replace with model name if needed.

app = Flask(__name__)

_lock = threading.Lock()
current_plan: Optional[ClinicPlanSchema] = None
current_results = None
run_state: Dict[str, Any] = {"status": "idle", "message": "No run yet."}
patients_cache: List[PatientRecord] = []


def _make_llm_client() -> LLMJsonClient:
    return LLMJsonClient(client=CLIENT, model=MODEL)


def _make_runner() -> TaskRunner:
    llm = _make_llm_client()
    ctx = TaskContext(llm=llm)
    return TaskRunner(ctx=ctx, task_registry=default_task_registry())


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
    if t:
        return list(t.depends_on or [])
    return []


def _run_single_task_action(patient_id: str, task_name: str) -> Optional[str]:
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

    def ensure_task(name: str) -> Any:
        if name in visiting:
            return None
        visiting.add(name)
        for dep in _task_depends_on(plan, name):
            ensure_task(dep)
        out = runner._run_task_for_patient(name, plan, patient, state)
        if out is not None:
            state.setdefault(f"{name}_by_patient", {})[patient_id] = out
        visiting.remove(name)
        return out

    if task_name not in runner.task_registry:
        return f"Task {task_name} is not available."
    out = ensure_task(task_name)
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
    return None


@app.route("/")
def home():
    return redirect(url_for("setup_plan"))


@app.route("/setup", methods=["GET", "POST"])
def setup_plan():
    global current_plan
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
        try:
            plan = build_clinic_plan_from_description(
                llm=_make_llm_client(),
                clinic_description=desc,
                default_target_condition=default_target,
                default_horizon_months=default_horizon,
            )
            with _lock:
                current_plan = plan
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
  <title>MedFlow — Clinic Setup</title>
  <link
    href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
    rel="stylesheet"
  />
  <link
    href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css"
    rel="stylesheet"
  />
  <style>
    body { background-color: #f0f4f8; font-family: 'Segoe UI', sans-serif; }
    .brand-bar {
      background: linear-gradient(90deg, #1a3c5e 0%, #2563a8 100%);
      padding: 1.1rem 2rem;
      color: white;
    }
    .brand-bar .brand-title {
      font-size: 1.6rem;
      font-weight: 700;
      letter-spacing: 0.03em;
    }
    .brand-bar .brand-subtitle {
      font-size: 0.88rem;
      opacity: 0.82;
      margin-top: 0.1rem;
    }
    .card-setup {
      border: none;
      border-radius: 12px;
      box-shadow: 0 4px 24px rgba(30,60,100,0.10);
    }
    .section-label {
      font-size: 0.78rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.07em;
      color: #6b7a8d;
      margin-bottom: 0.3rem;
    }
    .field-hint {
      font-size: 0.82rem;
      color: #7a8898;
      margin-top: 0.3rem;
    }
    .example-pill {
      display: inline-block;
      cursor: pointer;
      background: #e8f0fb;
      color: #1a4080;
      border-radius: 20px;
      padding: 0.25rem 0.75rem;
      font-size: 0.8rem;
      margin: 0.2rem 0.15rem 0.2rem 0;
      border: 1px solid #c5d8f5;
      transition: background 0.15s;
    }
    .example-pill:hover { background: #cfe0f7; }
    .btn-generate {
      background: linear-gradient(90deg, #1a3c5e, #2563a8);
      color: white;
      border: none;
      border-radius: 8px;
      padding: 0.65rem 2.2rem;
      font-size: 1rem;
      font-weight: 600;
      letter-spacing: 0.02em;
    }
    .btn-generate:hover { opacity: 0.9; color: white; }
    .alert-info-soft {
      background: #eaf2ff;
      border: 1px solid #c5d8f5;
      border-radius: 8px;
      color: #1a3c5e;
      font-size: 0.88rem;
    }
    .step-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      background: #2563a8;
      color: white;
      border-radius: 50%;
      width: 24px;
      height: 24px;
      font-size: 0.78rem;
      font-weight: 700;
      margin-right: 0.5rem;
      flex-shrink: 0;
    }
  </style>
</head>
<body>

<!-- Brand bar -->
<div class="brand-bar d-flex align-items-center gap-3">
  <i class="bi bi-heart-pulse-fill fs-3"></i>
  <div>
    <div class="brand-title">MedFlow</div>
    <div class="brand-subtitle">AI-Assisted Clinical Risk Screening &amp; Workflow Orchestration</div>
  </div>
</div>

<div class="container py-5" style="max-width:780px;">

  <!-- How it works banner -->
  <div class="alert alert-info-soft d-flex gap-2 align-items-start mb-4 px-4 py-3">
    <i class="bi bi-info-circle-fill mt-1 flex-shrink-0" style="color:#2563a8;"></i>
    <div>
      <strong>How MedFlow works:</strong>
      Describe your clinic's goals in plain language below. MedFlow will use AI to automatically
      build a weekly screening plan, scan your patient roster's longitudinal notes, identify
      patients at elevated risk, and generate clinician-ready summaries, referral packages,
      and patient instructions — all for human review before any action is taken.
    </div>
  </div>

  <!-- Error/message alert -->
  {% if msg %}
  <div class="alert alert-danger d-flex align-items-center gap-2 mb-4" role="alert">
    <i class="bi bi-exclamation-triangle-fill"></i>
    <div>{{ msg }}</div>
  </div>
  {% endif %}

  <!-- Main card -->
  <div class="card card-setup p-4 p-md-5">
    <h2 class="fw-bold mb-1" style="color:#1a3c5e;">Configure Your Screening Workflow</h2>
    <p class="text-muted mb-4" style="font-size:0.92rem;">
      Fill in the fields below to generate a tailored screening plan for your clinic.
      All fields except the description have sensible defaults — you can leave them blank
      to accept MedFlow's recommendations.
    </p>

    <form method="POST" action="/setup">

      <!-- Step 1: Clinic description -->
      <div class="mb-4">
        <div class="d-flex align-items-center mb-1">
          <span class="step-badge">1</span>
          <label class="fw-semibold fs-6 mb-0" for="clinic_description" style="color:#1a3c5e;">
            Describe Your Clinic's Goals
          </label>
        </div>
        <p class="field-hint mb-2">
          Write a short description of your clinic type, the condition you want to screen for,
          how often you'd like to review results, how many patients to shortlist each session,
          and any constraints (e.g., no repeated patients within 30 days, decision support only).
          Plain language is fine — MedFlow will interpret it.
        </p>

        <!-- Example pills -->
        <div class="mb-2">
          <span class="section-label">Quick examples — click to use:</span><br/>
          <span class="example-pill" onclick="setExample('pancreatic')">🔬 Pancreatic cancer risk</span>
          <span class="example-pill" onclick="setExample('colorectal')">🔬 Colorectal cancer risk</span>
          <span class="example-pill" onclick="setExample('followup')">📋 Follow-up gaps &amp; safety-netting</span>
          <span class="example-pill" onclick="setExample('referral')">📨 Referral quality &amp; triage</span>
        </div>

        <textarea
          class="form-control"
          id="clinic_description"
          name="clinic_description"
          rows="6"
          placeholder="e.g. Family medicine clinic in rural Ontario. Weekly review of top 5 patients at risk of pancreatic cancer based on EMR notes. Do not repeat patients within 6 months. Generate referral-ready summaries and patient instructions. Decision support only — no diagnoses."
          style="font-size:0.93rem; border-radius:8px;"
        >{{ default_desc }}</textarea>
        <div class="field-hint">
          <i class="bi bi-lightbulb"></i>
          Tip: Include your review cadence (e.g. "weekly"), how many patients (e.g. "top 5"),
          deduplication window (e.g. "no repeats within 30 days"), and any guardrails
          (e.g. "decision support only, no diagnosis").
        </div>
      </div>

      <!-- Step 2: Optional overrides -->
      <div class="mb-4">
        <div class="d-flex align-items-center mb-1">
          <span class="step-badge">2</span>
          <label class="fw-semibold fs-6 mb-0" style="color:#1a3c5e;">
            Optional: Screening Parameters
          </label>
        </div>
        <p class="field-hint mb-3">
          These override defaults extracted from your description above.
          Leave blank to let MedFlow infer them automatically.
        </p>
        <div class="row g-3">
          <div class="col-md-6">
            <label class="form-label section-label" for="default_target_condition">
              Target Condition
            </label>
            <input
              type="text"
              class="form-control"
              id="default_target_condition"
              name="default_target_condition"
              placeholder="e.g. pancreatic_cancer, colorectal_cancer"
              style="border-radius:8px;"
            />
            <div class="field-hint">
              The specific cancer or condition to screen for. Use underscores, no spaces.
            </div>
          </div>
          <div class="col-md-6">
            <label class="form-label section-label" for="default_horizon_months">
              Risk Horizon (months)
            </label>
            <input
              type="number"
              class="form-control"
              id="default_horizon_months"
              name="default_horizon_months"
              placeholder="e.g. 36"
              min="1"
              max="120"
              style="border-radius:8px;"
            />
            <div class="field-hint">
              Timeframe over which risk is assessed (1–120 months). Default: 36 months.
            </div>
          </div>
        </div>
      </div>

      <!-- Submit -->
      <div class="d-grid mt-4">
        <button type="submit" class="btn btn-generate btn-lg">
          <i class="bi bi-cpu me-2"></i>Generate Screening Plan
        </button>
      </div>

      <p class="text-center text-muted mt-3" style="font-size:0.82rem;">
        MedFlow will analyse your description and build a structured plan.
        You will be able to review and adjust it before any patients are screened.
      </p>

    </form>
  </div>
</div>

<script>
const examples = {
  pancreatic: "Family medicine clinic in rural Ontario. Weekly clinician review, shortlist top 5 patients. Goal: identify patients at elevated risk of pancreatic cancer based on longitudinal EMR notes. Generate risk assessment, clinician chart summary, admin referral payload, and patient instructions. Do not repeat patients within 6 months. No diagnosis — decision support only.",
  colorectal: "Community family practice. Monthly review of top 5 patients overdue for colorectal cancer screening or showing relevant risk factors in their notes. Generate clinician summaries and referral letters for patients at moderate or high risk. Decision support only.",
  followup: "Family medicine clinic. Weekly review of patients with potential follow-up gaps: missed appointments, abnormal lab results not acted on, pending imaging. Generate follow-up gap reports and suggested care plan actions. No repeated patients within 30 days.",
  referral: "Rural family practice. Weekly session to prioritize and triage referral queue. For each top-5 patient, generate a structured referral intake checklist and referral letter. Highlight semi-urgent and urgent cases. Decision support only."
};

function setExample(key) {
  document.getElementById('clinic_description').value = examples[key];
}
</script>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
    return render_template_string(tpl, default_desc=default_desc, msg=msg)


@app.route("/plan", methods=["GET", "POST"])
def plan_page():
    global current_plan
    with _lock:
        plan = current_plan
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

        try:
            updated = ClinicPlanSchema.model_validate(data)
            with _lock:
                current_plan = updated
            plan = updated
            if not msg:
                msg = "Plan updated."
        except Exception as e:
            msg = f"Validation error while saving plan: {e}"

    plan_json = _plan_to_json_text(plan)
    task_rows = []
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
  <title>MedFlow — Review &amp; Configure Plan</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet"/>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet"/>
  <style>
    body { background-color: #f0f4f8; font-family: 'Segoe UI', sans-serif; }
    .brand-bar {
      background: linear-gradient(90deg, #1a3c5e 0%, #2563a8 100%);
      padding: 1.1rem 2rem; color: white;
    }
    .brand-bar .brand-title { font-size: 1.6rem; font-weight: 700; letter-spacing: 0.03em; }
    .brand-bar .brand-subtitle { font-size: 0.88rem; opacity: 0.82; margin-top: 0.1rem; }
    .card-panel {
      border: none; border-radius: 12px;
      box-shadow: 0 4px 24px rgba(30,60,100,0.10);
      background: white;
    }
    .section-label {
      font-size: 0.78rem; font-weight: 600;
      text-transform: uppercase; letter-spacing: 0.07em;
      color: #6b7a8d; margin-bottom: 0.3rem;
    }
    .field-hint { font-size: 0.82rem; color: #7a8898; margin-top: 0.25rem; }
    .step-badge {
      display: inline-flex; align-items: center; justify-content: center;
      background: #2563a8; color: white; border-radius: 50%;
      width: 24px; height: 24px; font-size: 0.78rem; font-weight: 700;
      margin-right: 0.5rem; flex-shrink: 0;
    }
    .btn-primary-medflow {
      background: linear-gradient(90deg, #1a3c5e, #2563a8);
      color: white; border: none; border-radius: 8px;
      padding: 0.6rem 1.8rem; font-weight: 600;
    }
    .btn-primary-medflow:hover { opacity: 0.9; color: white; }
    .btn-run {
      background: linear-gradient(90deg, #145a32, #1e8449);
      color: white; border: none; border-radius: 8px;
      padding: 0.6rem 1.8rem; font-weight: 600;
    }
    .btn-run:hover { opacity: 0.9; color: white; }
    .btn-run:disabled {
      background: #adb5bd; cursor: not-allowed; opacity: 0.7;
    }

    /* ── Task cards ── */
    .task-zone {
      min-height: 80px; border-radius: 10px;
      border: 2px dashed #c5d8f5; padding: 0.75rem;
      background: #f7faff; transition: background 0.2s;
    }
    .task-zone.drag-over { background: #dceeff; border-color: #2563a8; }
    .task-card {
      background: white; border-radius: 8px;
      border: 1px solid #dde6f0;
      box-shadow: 0 2px 6px rgba(30,60,100,0.07);
      padding: 0.65rem 0.9rem;
      margin-bottom: 0.5rem;
      cursor: grab; display: flex; align-items: flex-start; gap: 0.6rem;
      transition: box-shadow 0.15s;
      user-select: none;
    }
    .task-card:active { cursor: grabbing; box-shadow: 0 6px 18px rgba(30,60,100,0.18); }
    .task-card.dragging { opacity: 0.45; }
    .task-icon { font-size: 1.2rem; margin-top: 0.05rem; flex-shrink: 0; }
    .task-name { font-weight: 600; font-size: 0.9rem; color: #1a3c5e; }
    .task-desc { font-size: 0.78rem; color: #6b7a8d; margin-top: 0.1rem; }
    .task-badge-dep {
      display: inline-block; font-size: 0.72rem;
      background: #e8f0fb; color: #1a4080;
      border-radius: 12px; padding: 0.1rem 0.5rem;
      margin-top: 0.25rem; margin-right: 0.2rem;
    }
    .params-toggle { font-size: 0.75rem; color: #2563a8; cursor: pointer; margin-top: 0.3rem; text-decoration: underline dotted; }
    .params-box { display: none; margin-top: 0.4rem; }
    .params-box textarea { font-size: 0.78rem; font-family: monospace; border-radius: 6px; }
    .dep-error {
      font-size: 0.75rem; color: #c0392b; background: #fdf0ef;
      border: 1px solid #f5c6c6; border-radius: 6px;
      padding: 0.25rem 0.6rem; margin-top: 0.3rem; display: none;
    }
    .unsaved-banner {
      display: none; background: #fff3cd; border: 1px solid #ffc107;
      border-radius: 8px; padding: 0.5rem 1rem;
      font-size: 0.86rem; color: #7d5a00;
      align-items: center; gap: 0.5rem;
    }
  </style>
</head>
<body>

<!-- Brand bar -->
<div class="brand-bar d-flex align-items-center gap-3">
  <i class="bi bi-heart-pulse-fill fs-3"></i>
  <div>
    <div class="brand-title">MedFlow</div>
    <div class="brand-subtitle">AI-Assisted Clinical Risk Screening &amp; Workflow Orchestration</div>
  </div>
  <div class="ms-auto">
    <a href="/setup" class="btn btn-sm btn-light text-primary fw-semibold">
      <i class="bi bi-arrow-left-circle me-1"></i>Back to Setup
    </a>
  </div>
</div>

<div class="container py-4" style="max-width:860px;">

  <!-- Unsaved changes banner -->
  <div class="unsaved-banner d-flex mb-3" id="unsavedBanner">
    <i class="bi bi-exclamation-triangle-fill"></i>
    <span>You have unsaved changes. Please click <strong>Save Plan</strong> before running.</span>
  </div>

  {% if msg %}
  <div class="alert {% if 'error' in msg.lower() or 'invalid' in msg.lower() or 'failed' in msg.lower() %}alert-danger{% else %}alert-success{% endif %} d-flex align-items-center gap-2 mb-3">
    <i class="bi {% if 'error' in msg.lower() or 'invalid' in msg.lower() or 'failed' in msg.lower() %}bi-exclamation-triangle-fill{% else %}bi-check-circle-fill{% endif %}"></i>
    <div>{{ msg }}</div>
  </div>
  {% endif %}

  <form method="POST" action="/plan" id="planForm">

    <!-- ── Section 1: Screening parameters ── -->
    <div class="card-panel p-4 mb-4">
      <div class="d-flex align-items-center mb-3">
        <span class="step-badge">1</span>
        <h5 class="fw-bold mb-0" style="color:#1a3c5e;">Screening Parameters</h5>
      </div>
      <p class="text-muted mb-3" style="font-size:0.88rem;">
        These parameters control what condition is screened, over what time horizon, and how patients are scheduled for review.
        Fields marked <span class="text-danger">*</span> are required.
      </p>

      <div class="row g-3">
        <div class="col-md-6">
          <label class="form-label section-label">Target Condition <span class="text-danger">*</span></label>
          <input type="text" class="form-control tracked" name="target_condition"
            value="{{ plan.target_condition or '' }}"
            placeholder="e.g. pancreatic_cancer"
            style="border-radius:8px;"/>
          <div class="field-hint">The cancer or clinical condition to screen for. Use underscores (e.g. <code>colorectal_cancer</code>).</div>
        </div>
        <div class="col-md-3">
          <label class="form-label section-label">Risk Horizon (months) <span class="text-danger">*</span></label>
          <input type="number" class="form-control tracked" name="horizon_months"
            value="{{ plan.horizon_months or '' }}"
            min="1" max="120" placeholder="e.g. 36"
            style="border-radius:8px;"/>
          <div class="field-hint">Timeframe for risk assessment (1–120 months).</div>
        </div>
        <div class="col-md-3">
          <label class="form-label section-label">Review Cadence <span class="text-danger">*</span></label>
          <select class="form-select tracked" name="cadence" style="border-radius:8px;">
            {% for opt in ['daily','weekly','monthly','per_visit','yearly'] %}
            <option value="{{ opt }}" {% if plan.constraints.cadence == opt %}selected{% endif %}>
              {{ opt|capitalize }}
            </option>
            {% endfor %}
          </select>
          <div class="field-hint">How often the screening workflow runs.</div>
        </div>
        <div class="col-md-3">
          <label class="form-label section-label">Patients per Review <span class="text-danger">*</span></label>
          <input type="number" class="form-control tracked" name="review_limit"
            value="{{ plan.constraints.review_limit }}"
            min="1" max="200"
            style="border-radius:8px;"/>
          <div class="field-hint">Max patients shortlisted each session (e.g. 5).</div>
        </div>
        <div class="col-md-3">
          <label class="form-label section-label">Deduplication Window (days)</label>
          <input type="number" class="form-control tracked" name="dedup_days"
            value="{{ plan.constraints.dedup_days }}"
            min="0"
            style="border-radius:8px;"/>
          <div class="field-hint">A patient reviewed within this many days will not be shortlisted again (0 = no limit).</div>
        </div>
        <div class="col-md-3">
          <label class="form-label section-label">Selection Method</label>
          <select class="form-select tracked" name="selection_method" style="border-radius:8px;">
            {% for opt in ['top_k','threshold','threshold_then_top_k','first_k'] %}
            <option value="{{ opt }}" {% if plan.constraints.selection.method == opt %}selected{% endif %}>
              {{ opt.replace('_',' ')|title }}
            </option>
            {% endfor %}
          </select>
          <div class="field-hint">How patients are ranked and selected from the scored pool.</div>
        </div>
        <div class="col-md-3">
          <label class="form-label section-label">Risk Threshold — Low (&lt;)</label>
          <input type="number" class="form-control tracked" name="risk_low_lt" step="0.01" min="0" max="1"
            value="{{ plan.constraints.risk_level_policy.low_lt }}"
            style="border-radius:8px;"/>
          <div class="field-hint">Patients below this probability (0–1) are classified as <strong>low risk</strong>.</div>
        </div>
        <div class="col-md-3">
          <label class="form-label section-label">Risk Threshold — Moderate (&lt;)</label>
          <input type="number" class="form-control tracked" name="risk_moderate_lt" step="0.01" min="0" max="1"
            value="{{ plan.constraints.risk_level_policy.moderate_lt }}"
            style="border-radius:8px;"/>
          <div class="field-hint">Patients between the low and this value are <strong>moderate risk</strong>; above is <strong>high risk</strong>.</div>
        </div>
      </div>
    </div>

    <!-- Hidden inputs for task state (populated by JS) -->
    <div id="taskHiddenInputs"></div>

    <!-- ── Section 2: Task configuration ── -->
    <div class="card-panel p-4 mb-4">
      <div class="d-flex align-items-center mb-1">
        <span class="step-badge">2</span>
        <h5 class="fw-bold mb-0" style="color:#1a3c5e;">Workflow Tasks</h5>
      </div>
      <p class="text-muted mb-3" style="font-size:0.88rem;">
        The AI has suggested the tasks below based on your clinic description.
        <strong>Drag</strong> a task to move it between Active and Available.
        Tasks with unmet prerequisites cannot be activated — hover to see why.
      </p>

      <div class="row g-3">
        <!-- Active tasks -->
        <div class="col-md-6">
          <div class="d-flex align-items-center mb-2 gap-2">
            <i class="bi bi-check-circle-fill text-success"></i>
            <span class="fw-semibold" style="color:#145a32;">Active Tasks</span>
            <span class="badge bg-success ms-1" id="activeCount">0</span>
          </div>
          <div class="field-hint mb-2">These tasks will run for each selected patient.</div>
          <div class="task-zone" id="activeZone" ondragover="onDragOver(event)" ondrop="onDrop(event, 'active')" ondragleave="onDragLeave(event)">
            <div id="activeEmpty" class="text-center text-muted py-3" style="font-size:0.82rem; display:none;">
              <i class="bi bi-inbox fs-4 d-block mb-1"></i>No active tasks. Drag tasks here.
            </div>
          </div>
        </div>

        <!-- Available tasks -->
        <div class="col-md-6">
          <div class="d-flex align-items-center mb-2 gap-2">
            <i class="bi bi-circle text-secondary"></i>
            <span class="fw-semibold" style="color:#6b7a8d;">Available (Inactive) Tasks</span>
            <span class="badge bg-secondary ms-1" id="inactiveCount">0</span>
          </div>
          <div class="field-hint mb-2">Drag a task to Active to include it in the workflow.</div>
          <div class="task-zone" id="inactiveZone" ondragover="onDragOver(event)" ondrop="onDrop(event, 'inactive')" ondragleave="onDragLeave(event)">
            <div id="inactiveEmpty" class="text-center text-muted py-3" style="font-size:0.82rem; display:none;">
              <i class="bi bi-check-all fs-4 d-block mb-1"></i>All tasks are active.
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ── Save & Run ── -->
    <div class="card-panel p-4 d-flex flex-wrap align-items-center justify-content-between gap-3">
      <div>
        <div class="fw-semibold" style="color:#1a3c5e;">Ready to proceed?</div>
        <div class="field-hint">Save your plan first, then run the screening workflow.</div>
      </div>
      <div class="d-flex gap-2 flex-wrap">
        <button type="submit" class="btn btn-primary-medflow" id="saveBtn">
          <i class="bi bi-floppy me-1"></i>Save Plan
        </button>
        <a href="/run" class="btn btn-run" id="runBtn">
          <i class="bi bi-play-fill me-1"></i>Run Screening
        </a>
      </div>
    </div>

  </form>
</div>

<!-- Task metadata from Python -->
<script>
const TASK_META = {
  "intake_workflow":            { icon: "bi-hospital",          desc: "Generates a structured clinic workflow profile from your description." },
  "risk_assessment":            { icon: "bi-graph-up-arrow",    desc: "Scores each patient's risk probability for the target condition using their clinical notes." },
  "queue_prioritization":       { icon: "bi-sort-down",         desc: "Ranks patients by urgency to help prioritise the review queue." },
  "clinician_summary":          { icon: "bi-file-earmark-text", desc: "Produces a concise chart summary with suggested orders and referrals for the clinician." },
  "admin_referral":             { icon: "bi-send",              desc: "Generates a structured referral payload for administrative processing." },
  "patient_instructions":       { icon: "bi-person-lines-fill", desc: "Drafts plain-language instructions for the patient regarding next steps." },
  "results_summary":            { icon: "bi-clipboard2-data",   desc: "Summarises lab, imaging, and trending data from the patient's notes." },
  "transcription":              { icon: "bi-mic",               desc: "Placeholder for audio transcription; passes notes through as-is in this demo." },
  "referral_letter":            { icon: "bi-envelope-paper",    desc: "Drafts a formal referral letter addressed to the receiving clinician or service." },
  "differential_diagnosis":     { icon: "bi-search",            desc: "Lists possible diagnoses with supporting reasoning from the clinical notes." },
  "guideline_comparison":       { icon: "bi-journals",          desc: "Compares the patient's situation against relevant clinical guidelines." },
  "followup_gap_detection":     { icon: "bi-calendar-x",        desc: "Identifies missed follow-ups, pending results, and care gaps in the notes." },
  "referral_intake_checklist":  { icon: "bi-card-checklist",    desc: "Produces a structured checklist for the receiving service's intake process." },
  "lab_trend_summary":          { icon: "bi-activity",          desc: "Summarises lab value trends over time with clinician and patient-friendly narratives." },
  "care_plan_reconciliation":   { icon: "bi-arrow-repeat",      desc: "Compares prior care plan items against current notes to flag completed, changed, or unresolved items." }
};

const DEPENDS_ON = {
  "clinician_summary":         ["risk_assessment"],
  "admin_referral":            ["risk_assessment"],
  "patient_instructions":      ["risk_assessment"],
  "referral_letter":           ["risk_assessment"],
  "differential_diagnosis":    ["risk_assessment"],
  "guideline_comparison":      ["risk_assessment"],
  "referral_intake_checklist": ["admin_referral"],
  "queue_prioritization":      [],
  "lab_trend_summary":         [],
  "followup_gap_detection":    [],
  "care_plan_reconciliation":  [],
  "results_summary":           [],
  "intake_workflow":           [],
  "transcription":             [],
  "risk_assessment":           []
};

// Initial task state from Python
const INITIAL_TASKS = [
  {% for t in task_rows %}
  {
    name: {{ t.name | tojson }},
    enabled: {{ 'true' if t.enabled else 'false' }},
    params: {{ t.params_text | tojson }},
    order: {{ loop.index }}
  },
  {% endfor %}
];

// ── State ──
let taskState = {};  // name -> { enabled, params, order }
let dragSrc = null;
let planSaved = true;

function init() {
  INITIAL_TASKS.forEach(t => {
    taskState[t.name] = { enabled: t.enabled, params: t.params, order: t.order };
  });
  renderAllTasks();
  updateRunButton();
}

function getActiveNames() {
  return Object.entries(taskState)
    .filter(([,v]) => v.enabled)
    .sort((a,b) => a[1].order - b[1].order)
    .map(([k]) => k);
}

function getInactiveNames() {
  return Object.entries(taskState)
    .filter(([,v]) => !v.enabled)
    .sort((a,b) => a[1].order - b[1].order)
    .map(([k]) => k);
}

function renderAllTasks() {
  const az = document.getElementById('activeZone');
  const iz = document.getElementById('inactiveZone');
  // Clear existing cards (keep empty placeholders)
  az.querySelectorAll('.task-card').forEach(el => el.remove());
  iz.querySelectorAll('.task-card').forEach(el => el.remove());

  const active = getActiveNames();
  const inactive = getInactiveNames();

  active.forEach((name, idx) => az.appendChild(makeCard(name, true, idx)));
  inactive.forEach((name, idx) => iz.appendChild(makeCard(name, false, idx)));

  document.getElementById('activeCount').textContent = active.length;
  document.getElementById('inactiveCount').textContent = inactive.length;

  document.getElementById('activeEmpty').style.display = active.length === 0 ? 'block' : 'none';
  document.getElementById('inactiveEmpty').style.display = inactive.length === 0 ? 'block' : 'none';

  buildHiddenInputs();
}

function makeCard(name, enabled, idx) {
  const meta = TASK_META[name] || { icon: 'bi-gear', desc: 'Custom task.' };
  const deps = DEPENDS_ON[name] || [];
  const unmet = deps.filter(d => !taskState[d] || !taskState[d].enabled);

  const card = document.createElement('div');
  card.className = 'task-card' + (enabled && unmet.length > 0 ? ' border-warning' : '');
  card.draggable = true;
  card.dataset.name = name;

  const depsHtml = deps.length
    ? deps.map(d => `<span class="task-badge-dep"><i class="bi bi-arrow-return-right"></i> ${d}</span>`).join('')
    : '';

  const unmetHtml = (enabled && unmet.length > 0)
    ? `<div class="dep-error" style="display:block;"><i class="bi bi-exclamation-triangle-fill me-1"></i>Requires: ${unmet.join(', ')}</div>`
    : '';

  const paramsDisplay = taskState[name].params && taskState[name].params !== '{}' ? taskState[name].params : '{}';

  card.innerHTML = `
    <i class="bi ${meta.icon} task-icon" style="color:#2563a8;"></i>
    <div class="flex-grow-1">
      <div class="d-flex align-items-center gap-2">
        <span class="task-name">${name.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase())}</span>
        <span class="badge ${enabled ? 'bg-success' : 'bg-secondary'}" style="font-size:0.68rem;">${enabled ? 'Active' : 'Inactive'}</span>
      </div>
      <div class="task-desc">${meta.desc}</div>
      ${depsHtml ? `<div class="mt-1">${depsHtml}</div>` : ''}
      ${unmetHtml}
      <div class="params-toggle" onclick="toggleParams('${name}')">
        <i class="bi bi-sliders2 me-1"></i>Advanced parameters
      </div>
      <div class="params-box" id="params-${name}">
        <textarea class="form-control" rows="3" placeholder='{"key": "value"}'
          onchange="updateParams('${name}', this.value)"
          oninput="markUnsaved()">${paramsDisplay}</textarea>
        <div class="field-hint">Optional JSON key-value pairs passed to this task. Leave as <code>{}</code> unless advised.</div>
      </div>
    </div>
    <i class="bi bi-grip-vertical text-secondary" style="font-size:1.1rem; margin-top:2px; cursor:grab;"></i>
  `;

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
}

function onDragOver(e) {
  e.preventDefault();
  e.currentTarget.classList.add('drag-over');
}

function onDragLeave(e) {
  e.currentTarget.classList.remove('drag-over');
}

function onDrop(e, zone) {
  e.preventDefault();
  e.currentTarget.classList.remove('drag-over');
  if (!dragSrc) return;

  const name = dragSrc;
  const toActive = zone === 'active';

  // Dependency check when activating
  if (toActive) {
    const deps = DEPENDS_ON[name] || [];
    const unmet = deps.filter(d => !taskState[d] || !taskState[d].enabled);
    if (unmet.length > 0) {
      showDepToast(name, unmet);
      dragSrc = null;
      return;
    }
  }

  // Deactivation check: if removing, warn if other active tasks depend on this
  if (!toActive) {
    const dependents = Object.entries(DEPENDS_ON)
      .filter(([k, deps]) => deps.includes(name) && taskState[k] && taskState[k].enabled)
      .map(([k]) => k);
    if (dependents.length > 0) {
      showDepToast(name, [], dependents);
      dragSrc = null;
      return;
    }
  }

  taskState[name].enabled = toActive;
  // Re-assign order within zone
  const siblings = toActive ? getActiveNames().filter(n => n !== name) : getInactiveNames().filter(n => n !== name);
  taskState[name].order = siblings.length + 1;
  dragSrc = null;
  renderAllTasks();
  markUnsaved();
}

function showDepToast(name, unmet, dependents) {
  const existing = document.getElementById('depToast');
  if (existing) existing.remove();
  const label = name.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase());
  let msg;
  if (unmet && unmet.length)
    msg = `<strong>${label}</strong> requires these tasks to be active first: <strong>${unmet.join(', ')}</strong>.`;
  else
    msg = `Cannot deactivate <strong>${label}</strong> — other active tasks depend on it: <strong>${dependents.join(', ')}</strong>.`;
  const toast = document.createElement('div');
  toast.id = 'depToast';
  toast.className = 'alert alert-warning d-flex align-items-center gap-2 position-fixed';
  toast.style.cssText = 'bottom:1.5rem;left:50%;transform:translateX(-50%);z-index:9999;min-width:340px;max-width:580px;box-shadow:0 4px 18px rgba(0,0,0,0.15);';
  toast.innerHTML = `<i class="bi bi-exclamation-triangle-fill"></i><div>${msg}</div>`;
  document.body.appendChild(toast);
  setTimeout(() => { if (toast.parentNode) toast.remove(); }, 4000);
}

function buildHiddenInputs() {
  const container = document.getElementById('taskHiddenInputs');
  container.innerHTML = '';
  Object.entries(taskState).forEach(([name, state], idx) => {
    const add = (n, v) => {
      const inp = document.createElement('input');
      inp.type = 'hidden'; inp.name = n; inp.value = v;
      container.appendChild(inp);
    };
    if (state.enabled) add(`enabled__${name}`, 'on');
    add(`params__${name}`, state.params || '{}');
    add(`order__${name}`, state.order || idx + 1);
  });
}

function markUnsaved() {
  planSaved = false;
  document.getElementById('unsavedBanner').style.display = 'flex';
  updateRunButton();
}

function updateRunButton() {
  const btn = document.getElementById('runBtn');
  if (!planSaved) {
    btn.classList.add('disabled');
    btn.setAttribute('aria-disabled', 'true');
    btn.onclick = e => { e.preventDefault(); document.getElementById('unsavedBanner').scrollIntoView({behavior:'smooth'}); };
  } else {
    btn.classList.remove('disabled');
    btn.removeAttribute('aria-disabled');
    btn.onclick = null;
  }
}

document.getElementById('planForm').addEventListener('change', markUnsaved);
document.getElementById('planForm').addEventListener('submit', () => {
  buildHiddenInputs();
  planSaved = true;
});

init();
</script>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
    return render_template_string(tpl, plan=plan, plan_json=plan_json, task_rows=task_rows, msg=msg)



def _background_run(plan: ClinicPlanSchema):
    global current_results, run_state
    try:
        with _lock:
            run_state = {"status": "running", "message": "Run in progress...", "started_at": time.time()}
        patients = load_sample_patients()
        runner = _make_runner()
        bundle = runner.run_weekly_review(patients=patients, plan=plan, run_date=date.today().isoformat())
        with _lock:
            current_results = bundle
            run_state = {"status": "done", "message": f"Run completed. Selected {len(bundle.selected)} patients."}
    except Exception as e:
        with _lock:
            run_state = {"status": "error", "message": str(e)}


@app.route("/run")
def run_plan():
    with _lock:
        plan = current_plan
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
    global current_results
    msg = ""
    if request.method == "POST":
        action = request.form.get("action", "")
        patient_id = request.form.get("patient_id", "")
        if action and patient_id:
            task_map = {
                "generate_clinician_summary": "clinician_summary",
                "generate_admin_referral": "admin_referral",
                "generate_referral_letter": "referral_letter",
                "generate_patient_instructions": "patient_instructions",
                "generate_lab_trend_summary": "lab_trend_summary",
                "generate_followup_gap": "followup_gap_detection",
            }
            task_name = task_map.get(action)
            if task_name:
                err = _run_single_task_action(patient_id, task_name)
                msg = err or f"Ran {task_name} for {patient_id}."
            else:
                msg = "Unknown action."

    with _lock:
        plan = current_plan
        results = current_results
        status = dict(run_state)

    auto_refresh = status.get("status") == "running"
    tpl = """
    <html>
    <head>
      {% if auto_refresh %}<meta http-equiv="refresh" content="3">{% endif %}
    </head>
    <body>
      <h2>Results</h2>
      <p><a href='{{ url_for("setup_plan") }}'>Setup</a> | <a href='{{ url_for("plan_page") }}'>Plan</a> {% if plan %}| <a href='{{ url_for("run_plan") }}'>Run now</a>{% endif %}</p>
      <p><b>Status:</b> {{ status.get('status') }} | {{ status.get('message') }}</p>
      {% if auto_refresh %}<p>Auto-refresh is enabled while the run is active.</p>{% endif %}
      {% if msg %}<p style='color:#060;'>{{ msg }}</p>{% endif %}

      {% if not plan %}
        <p>No plan yet.</p>
      {% else %}
        <h3>Plan Snapshot</h3>
        <p>
          Target: {{ plan.target_condition or 'N/A' }}
          {% if plan.horizon_months %}| Horizon: {{ plan.horizon_months }} months{% endif %}
          | Cadence: {{ plan.constraints.cadence }}
          | Review limit: {{ plan.constraints.review_limit }}
          | Selection: {{ plan.constraints.selection.method }}
          {% if plan.constraints.selection.source_task %} ({{ plan.constraints.selection.source_task }}){% endif %}
        </p>
      {% endif %}

      {% if results %}
        <h3>Latest Run</h3>
        <p>
          Date: {{ results.run_date }} | Clinic: {{ results.clinic_name }} | Selected: {{ results.selected|length }} | Not selected: {{ results.not_selected_count }}
        </p>
        {% for b in results.selected %}
          <hr>
          <h4>{{ b.patient_id }}</h4>
          <p>{{ b.selection_reason }}</p>
          {% if b.risk %}
            <p><b>Risk:</b> {{ '%.3f'|format(b.risk.risk_probability) }} ({{ b.risk.risk_level }}) for {{ b.risk.target_condition }} / {{ b.risk.horizon_months }} months</p>
          {% endif %}
          {% if b.clinician_summary %}
            <p><b>Clinician summary:</b> {{ b.clinician_summary.summary_for_chart }}</p>
          {% endif %}
          {% if b.admin_referral %}
            <p><b>Admin referral:</b> {{ b.admin_referral.urgency }} to {{ b.admin_referral.destination_service }}<br>{{ b.admin_referral.reason_for_referral }}</p>
          {% endif %}
          {% if b.extra_outputs %}
            <details>
              <summary>Extra outputs ({{ b.extra_outputs|length }})</summary>
              <pre style="background:#f6f6f6;padding:10px;max-width:1200px;overflow:auto;">{{ b.extra_outputs | tojson(indent=2) }}</pre>
            </details>
          {% endif %}

          <form method="post" style="margin-top:8px;">
            <input type="hidden" name="patient_id" value="{{ b.patient_id }}">
            <button name="action" value="generate_clinician_summary" type="submit">Generate/Refresh Clinician Summary</button>
            <button name="action" value="generate_admin_referral" type="submit">Generate/Refresh Admin Referral</button>
            <button name="action" value="generate_referral_letter" type="submit">Generate Referral Letter</button>
            <button name="action" value="generate_patient_instructions" type="submit">Generate Patient Instructions</button>
            <button name="action" value="generate_lab_trend_summary" type="submit">Generate Lab Trend Summary</button>
            <button name="action" value="generate_followup_gap" type="submit">Run Follow-up Gap Detection</button>
          </form>
        {% endfor %}
      {% else %}
        <p>No results yet. Go to <a href='{{ url_for("plan_page") }}'>Plan</a> and click Run Plan.</p>
      {% endif %}
    </body></html>
    """
    resp = make_response(render_template_string(tpl, plan=plan, results=results, status=status, auto_refresh=auto_refresh, msg=msg))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp


# Backward-compatible alias used in some earlier patches
@app.route("/edit_plan", methods=["GET", "POST"])
def edit_plan_alias():
    return plan_page()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
