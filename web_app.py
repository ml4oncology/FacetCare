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

ENDPOINT = "http://192.168.0.2:881/v1/"
CLIENT = OpenAI(base_url=ENDPOINT, api_key="")
MODEL = "gpt-4.1-mini"  # Replace with model name if needed.

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
    <html><body>
      <h2>MedFlow Setup</h2>
      <p><a href='{{ url_for("plan_page") }}'>Plan</a> | <a href='{{ url_for("results_page") }}'>Results</a></p>
      {% if msg %}<p style='color:#b00;'>{{ msg }}</p>{% endif %}
      <form method="post">
        <p><b>Clinic description and goals</b></p>
        <textarea name="clinic_description" rows="10" cols="120">{{ default_desc }}</textarea>
        <p>
          Default target condition (optional): <input name="default_target_condition" size="40" value="pancreatic cancer">
          Horizon months (optional): <input name="default_horizon_months" size="6" value="12">
        </p>
        <button type="submit">Generate Plan</button>
      </form>
    </body></html>
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
    <html><body>
      <h2>Plan Editor</h2>
      <p><a href='{{ url_for("setup_plan") }}'>Setup</a> | <a href='{{ url_for("results_page") }}'>Results</a></p>
      {% if msg %}<p style='color:#060;'>{{ msg }}</p>{% endif %}
      <form method="post">
        <p>
          Target condition (optional): <input name="target_condition" size="40" value="{{ plan.target_condition or '' }}">
          Horizon months (optional): <input name="horizon_months" size="6" value="{{ plan.horizon_months or '' }}">
        </p>
        <p>
          Cadence: <input name="cadence" size="10" value="{{ plan.constraints.cadence }}">
          Review limit: <input name="review_limit" size="6" value="{{ plan.constraints.review_limit }}">
          Dedup days: <input name="dedup_days" size="6" value="{{ plan.constraints.dedup_days }}">
        </p>
        <p>
          Candidate pool strategy: <input name="candidate_pool_strategy" size="20" value="{{ plan.constraints.candidate_pool.strategy }}">
          Max candidates: <input name="max_candidates" size="8" value="{{ plan.constraints.candidate_pool.max_candidates or '' }}">
          Keywords CSV: <input name="candidate_pool_keywords" size="60" value="{{ (plan.constraints.candidate_pool.keywords or [])|join(', ') }}">
        </p>
        <p>
          Selection source task: <input name="selection_source_task" size="22" value="{{ plan.constraints.selection.source_task or '' }}">
          Method: <input name="selection_method" size="18" value="{{ plan.constraints.selection.method }}">
          K: <input name="selection_k" size="6" value="{{ plan.constraints.selection.k }}">
          Threshold: <input name="selection_threshold" size="8" value="{{ plan.constraints.selection.threshold if plan.constraints.selection.threshold is not none else '' }}">
        </p>
        <p>
          Risk low_lt: <input name="risk_low_lt" size="8" value="{{ plan.constraints.risk_level_policy.low_lt }}">
          Risk moderate_lt: <input name="risk_moderate_lt" size="8" value="{{ plan.constraints.risk_level_policy.moderate_lt }}">
        </p>
        <h3>Tasks</h3>
        <table border="1" cellpadding="6" cellspacing="0">
          <tr><th>Order</th><th>Enable</th><th>Task</th><th>Depends on</th><th>Params (JSON)</th></tr>
          {% for t in task_rows %}
          <tr>
            <td><input name="order__{{ t.name }}" size="4" value="{{ loop.index }}"></td>
            <td><input type="checkbox" name="enabled__{{ t.name }}" {% if t.enabled %}checked{% endif %}></td>
            <td><b>{{ t.name }}</b></td>
            <td>{{ t.depends_on }}</td>
            <td><textarea name="params__{{ t.name }}" rows="4" cols="55">{{ t.params_text }}</textarea></td>
          </tr>
          {% endfor %}
        </table>
        <p>
          <button type="submit">Save Plan</button>
          <button type="button" onclick="window.location='{{ url_for("run_plan") }}'">Run Plan</button>
        </p>
      </form>
      <h3>Raw Plan JSON</h3>
      <pre style="background:#f4f4f4;padding:12px;max-width:1200px;overflow:auto;">{{ plan_json }}</pre>
    </body></html>
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
