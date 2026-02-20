from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from .dedup import DedupStore, JSONFileDedupStore
from .schemas import ClinicPlanSchema, PatientRecord, QueuePrioritizationSchema, ReviewBundleSchema, SelectedPatientBundle
from .tasks import TaskBase, TaskContext


class TaskRunner:
    def __init__(
        self,
        *,
        ctx: TaskContext,
        task_registry: Dict[str, TaskBase],
        dedup_store: Optional[DedupStore] = None,
    ):
        self.ctx = ctx
        self.task_registry = task_registry
        self.dedup_store = dedup_store or JSONFileDedupStore("seen_patients.json")

    @staticmethod
    def _result_store_key(task_name: str) -> str:
        return f"{task_name}_by_patient"

    def _task_enabled(self, plan: ClinicPlanSchema, task_name: str) -> bool:
        for t in plan.tasks:
            if t.name == task_name:
                return bool(t.enabled)
        return False

    def _task_params(self, plan: ClinicPlanSchema, task_name: str) -> Dict[str, Any]:
        for t in plan.tasks:
            if t.name == task_name:
                return dict(t.params or {})
        return {}

    def _candidate_pool(self, patients: List[PatientRecord], plan: ClinicPlanSchema) -> List[PatientRecord]:
        pool = plan.constraints.candidate_pool
        out = list(patients)
        if pool.strategy == "recent_notes_only":
            rec = []
            for p in out:
                txt = p.longitudinal_notes.lower()
                if any(k in txt for k in ["202", "note", "lab", "follow", "new", "recent"]):
                    rec.append(p)
            out = rec or out
        elif pool.strategy == "keyword_prefilter":
            keywords = [k.lower() for k in pool.keywords or []]
            if keywords:
                filt = [p for p in out if any(k in p.longitudinal_notes.lower() for k in keywords)]
                out = filt or out
        if pool.max_candidates is not None:
            out = out[: int(pool.max_candidates)]
        return out

    def _dedup_filter(self, patients: List[PatientRecord], plan: ClinicPlanSchema, run_date: str) -> List[PatientRecord]:
        dd = int(plan.constraints.dedup_days)
        if dd <= 0:
            return list(patients)
        return [p for p in patients if not self.dedup_store.seen_within(p.patient_id, days=dd, as_of=run_date)]

    def _precompute_scored_tasks(self, *, plan: ClinicPlanSchema, candidates: List[PatientRecord], state: Dict[str, Any]) -> None:
        for task_name in ("risk_assessment", "queue_prioritization"):
            if not self._task_enabled(plan, task_name):
                continue
            task = self.task_registry.get(task_name)
            if task is None:
                continue
            params = self._task_params(plan, task_name)
            for p in candidates:
                out = task.run(ctx=self.ctx, plan=plan, patient=p, state=state, task_params=params)
                if out is not None:
                    state.setdefault(self._result_store_key(task_name), {})[p.patient_id] = out

    def _select_patients(self, *, plan: ClinicPlanSchema, candidates: List[PatientRecord], state: Dict[str, Any]) -> List[PatientRecord]:
        sel = plan.constraints.selection
        source = sel.source_task

        if source == "risk_assessment" and (state.get("risk_by_patient") or {}):
            ranked = []
            for p in candidates:
                r = state["risk_by_patient"].get(p.patient_id)
                if r is None:
                    continue
                ranked.append((p, float(r.risk_probability)))
            ranked.sort(key=lambda x: x[1], reverse=True)
            if sel.method in {"threshold", "threshold_then_top_k"} and sel.threshold is not None:
                ranked = [x for x in ranked if x[1] >= float(sel.threshold)]
            return [p for p, _ in (ranked if sel.method == "threshold" else ranked[: sel.k])]

        if source == "queue_prioritization" and (state.get("queue_priority_by_patient") or {}):
            ranked_q = []
            for p in candidates:
                q = state["queue_priority_by_patient"].get(p.patient_id)
                if isinstance(q, QueuePrioritizationSchema):
                    ranked_q.append((p, float(q.priority_score)))
            ranked_q.sort(key=lambda x: x[1], reverse=True)
            if sel.method in {"threshold", "threshold_then_top_k"} and sel.threshold is not None:
                ranked_q = [x for x in ranked_q if x[1] >= float(sel.threshold)]
            return [p for p, _ in (ranked_q if sel.method == "threshold" else ranked_q[: sel.k])]

        return candidates[: sel.k]

    def _selection_reason(self, *, patient: PatientRecord, plan: ClinicPlanSchema, state: Dict[str, Any], rank: int) -> str:
        source = plan.constraints.selection.source_task
        if source == "risk_assessment":
            r = (state.get("risk_by_patient") or {}).get(patient.patient_id)
            if r is not None:
                return f"Selected rank {rank} by risk_assessment score={float(r.risk_probability):.4f}."
        if source == "queue_prioritization":
            q = (state.get("queue_priority_by_patient") or {}).get(patient.patient_id)
            if q is not None:
                return f"Selected rank {rank} by queue_prioritization score={float(q.priority_score):.4f}."
        return f"Selected rank {rank} by clinic plan ordering ({plan.constraints.selection.method})."

    def _run_task_for_patient(self, task_name: str, plan: ClinicPlanSchema, patient: PatientRecord, state: Dict[str, Any]) -> Any:
        task = self.task_registry.get(task_name)
        if task is None:
            return None
        return task.run(ctx=self.ctx, plan=plan, patient=patient, state=state, task_params=self._task_params(plan, task_name))

    def run_weekly_review(self, *, patients: List[PatientRecord], plan: ClinicPlanSchema, run_date: Optional[str] = None) -> ReviewBundleSchema:
        run_date = run_date or dt.date.today().isoformat()
        state: Dict[str, Any] = {}

        if self._task_enabled(plan, "intake_workflow") and self.task_registry.get("intake_workflow") and patients:
            out = self._run_task_for_patient("intake_workflow", plan, patients[0], state)
            if out is not None:
                state.setdefault(self._result_store_key("intake_workflow"), {})[patients[0].patient_id] = out

        candidates = self._candidate_pool(patients, plan)
        candidates = self._dedup_filter(candidates, plan, run_date)
        self._precompute_scored_tasks(plan=plan, candidates=candidates, state=state)
        selected_patients = self._select_patients(plan=plan, candidates=candidates, state=state)

        selected_bundles: List[SelectedPatientBundle] = []
        precomputed = {"risk_assessment", "queue_prioritization", "intake_workflow"}

        for rank, p in enumerate(selected_patients, start=1):
            for ts in plan.tasks:
                if not ts.enabled or ts.name in precomputed or ts.name not in self.task_registry:
                    continue
                out = self._run_task_for_patient(ts.name, plan, p, state)
                if out is not None:
                    state.setdefault(self._result_store_key(ts.name), {})[p.patient_id] = out

            risk_obj = (state.get("risk_by_patient") or {}).get(p.patient_id)
            summary_obj = (state.get("clinician_summary_by_patient") or {}).get(p.patient_id)
            referral_obj = (state.get("admin_referral_by_patient") or {}).get(p.patient_id)

            extra_outputs: Dict[str, Any] = {}
            for task_name in [t.name for t in plan.tasks if t.enabled and t.name not in {"intake_workflow", "risk_assessment", "clinician_summary", "admin_referral"}]:
                mapping = state.get(self._result_store_key(task_name)) or {}
                if p.patient_id in mapping:
                    val = mapping[p.patient_id]
                    extra_outputs[task_name] = val.model_dump() if hasattr(val, "model_dump") else val

            selected_bundles.append(
                SelectedPatientBundle(
                    patient_id=p.patient_id,
                    selection_reason=self._selection_reason(patient=p, plan=plan, state=state, rank=rank),
                    risk=risk_obj,
                    clinician_summary=summary_obj,
                    admin_referral=referral_obj,
                    extra_outputs=extra_outputs,
                )
            )

            risk_prob = float(risk_obj.risk_probability) if risk_obj is not None else 0.0
            self.dedup_store.mark_seen(p.patient_id, as_of=run_date, risk_probability=risk_prob)

        workflow = state.get("workflow")
        clinic_name = getattr(workflow, "clinic_name", None) or "Unknown clinic"
        target = plan.target_condition
        horizon = plan.horizon_months
        if not target and plan.constraints.selection.source_task == "risk_assessment" and selected_bundles and selected_bundles[0].risk:
            target = selected_bundles[0].risk.target_condition
            horizon = selected_bundles[0].risk.horizon_months

        return ReviewBundleSchema(
            run_date=run_date,
            clinic_name=clinic_name,
            target_condition=target,
            horizon_months=horizon,
            selected=selected_bundles,
            not_selected_count=max(0, len(candidates) - len(selected_patients)),
        )


# Backward compatibility for older demo scripts
class PlanRunner(TaskRunner):
    def run(self, *, plan: ClinicPlanSchema, patients: List[PatientRecord], run_date: Optional[str] = None) -> ReviewBundleSchema:
        return self.run_weekly_review(plan=plan, patients=patients, run_date=run_date)
