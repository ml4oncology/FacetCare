from __future__ import annotations

import datetime as dt
import hashlib
import json
from typing import Any, Dict, List, Optional, Type

from . import prompts
from .dedup import DedupStore, JSONFileDedupStore
from .output_cache import JSONTaskOutputCache
from .schemas import (
    AdminReferralSchema,
    CarePlanReconciliationSchema,
    ClinicPlanSchema,
    ClinicWorkflowSchema,
    ClinicianSummarySchema,
    DifferentialDiagnosisSchema,
    FollowupGapSchema,
    GuidelineComparisonSchema,
    LabTrendSummarySchema,
    PatientInstructionsSchema,
    PatientRecord,
    QueuePrioritizationSchema,
    ReferralIntakeChecklistSchema,
    ReferralLetterSchema,
    ResultsSummarySchema,
    RiskAssessmentSchema,
    ReviewBundleSchema,
    SelectedPatientBundle,
)
from .tasks import TaskBase, TaskContext


class TaskRunner:
    def __init__(
        self,
        *,
        ctx: TaskContext,
        task_registry: Dict[str, TaskBase],
        dedup_store: Optional[DedupStore] = None,
        output_cache: Optional[JSONTaskOutputCache] = None,
    ):
        self.ctx = ctx
        self.task_registry = task_registry
        self.dedup_store = dedup_store or JSONFileDedupStore("facetcare_seen_patients.json")
        self.output_cache = output_cache

    @staticmethod
    def _result_store_key(task_name: str) -> str:
        return f"{task_name}_by_patient"

    @staticmethod
    def _alias_store_keys(task_name: str) -> List[str]:
        aliases = {
            "risk_assessment": ["risk_by_patient"],
            "queue_prioritization": ["queue_priority_by_patient"],
            "clinician_summary": ["clinician_summary_by_patient"],
            "admin_referral": ["admin_referral_by_patient"],
            "followup_gap_detection": ["followup_gap_by_patient"],
            "referral_intake_checklist": ["referral_intake_by_patient"],
            "lab_trend_summary": ["lab_trend_by_patient"],
            "care_plan_reconciliation": ["care_plan_recon_by_patient"],
        }
        out = [TaskRunner._result_store_key(task_name)]
        out.extend(aliases.get(task_name, []))
        return list(dict.fromkeys(out))

    @staticmethod
    def _schema_for_task(task_name: str) -> Optional[Type[Any]]:
        mapping: Dict[str, Type[Any]] = {
            "intake_workflow": ClinicWorkflowSchema,
            "risk_assessment": RiskAssessmentSchema,
            "queue_prioritization": QueuePrioritizationSchema,
            "clinician_summary": ClinicianSummarySchema,
            "admin_referral": AdminReferralSchema,
            "patient_instructions": PatientInstructionsSchema,
            "results_summary": ResultsSummarySchema,
            "referral_letter": ReferralLetterSchema,
            "differential_diagnosis": DifferentialDiagnosisSchema,
            "guideline_comparison": GuidelineComparisonSchema,
            "followup_gap_detection": FollowupGapSchema,
            "referral_intake_checklist": ReferralIntakeChecklistSchema,
            "lab_trend_summary": LabTrendSummarySchema,
            "care_plan_reconciliation": CarePlanReconciliationSchema,
        }
        return mapping.get(task_name)

    @staticmethod
    def _sha256_text(text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


    def _mark_task_source(self, *, state: Dict[str, Any], patient_id: str, task_name: str, source: str) -> None:
        by_patient = state.setdefault("_task_sources_by_patient", {})
        patient_map = by_patient.setdefault(patient_id, {})
        patient_map[task_name] = source

    def _get_task_sources_for_patient(self, *, state: Dict[str, Any], patient_id: str) -> Dict[str, str]:
        return dict(((state.get("_task_sources_by_patient") or {}).get(patient_id)) or {})

    def _cache_meta(self, *, task_name: str, plan: ClinicPlanSchema, patient: PatientRecord, task_params: Dict[str, Any]) -> Dict[str, Any]:
        llm_model = getattr(getattr(self, "ctx", None), "llm", None)
        model_name = getattr(llm_model, "model", None)
        clinic_level = task_name == "intake_workflow"
        patient_key = "__clinic__" if clinic_level else patient.patient_id
        note_hash = "clinic-level" if clinic_level else self._sha256_text(patient.longitudinal_notes)
        prompt_version = (prompts.PROMPT_TEMPLATE_VERSIONS or {}).get(task_name, "unknown")
        plan_fingerprint = self._sha256_text(json.dumps(plan.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, default=str))
        params_fingerprint = self._sha256_text(json.dumps(task_params or {}, sort_keys=True, ensure_ascii=False, default=str))
        return {
            "cache_schema": 1,
            "task": task_name,
            "patient": patient_key,
            "note_hash": note_hash,
            "plan_hash": plan_fingerprint,
            "task_params_hash": params_fingerprint,
            "task_params": task_params or {},
            "model": model_name or "unknown",
            "prompt_version": prompt_version,
        }

    def _deserialize_cached(self, task_name: str, payload: Any) -> Any:
        schema = self._schema_for_task(task_name)
        if schema is None or payload is None:
            return payload
        if isinstance(payload, schema):
            return payload
        if isinstance(payload, dict):
            return schema.model_validate(payload)
        return payload

    def _serialize_for_cache(self, obj: Any) -> Any:
        if hasattr(obj, "model_dump"):
            try:
                return obj.model_dump(mode="json")
            except Exception:
                return obj.model_dump()
        return obj

    def _write_output_to_state(self, *, task_name: str, patient: PatientRecord, out: Any, state: Dict[str, Any]) -> None:
        if out is None:
            return
        if task_name == "intake_workflow":
            state["workflow"] = out
            # also populate canonical store for visibility
            state.setdefault(self._result_store_key(task_name), {})[patient.patient_id] = out
            self._mark_task_source(state=state, patient_id=patient.patient_id, task_name=task_name, source="fresh")
            return
        for key in self._alias_store_keys(task_name):
            state.setdefault(key, {})[patient.patient_id] = out

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

        include_ids = [str(x).strip() for x in (pool.include_patient_ids or []) if str(x).strip()]
        if include_ids:
            include_set = {x for x in include_ids}
            out = [p for p in out if p.patient_id in include_set]

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
            for p in candidates:
                out = self._run_task_for_patient(task_name, plan, p, state, use_cache=True)
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

        if sel.method == "random_k":
            import random
            picked = list(candidates)
            random.Random(f"{plan.clinic_name}|{len(candidates)}|{sel.k}").shuffle(picked)
            return picked[: sel.k]

        if source == "queue_prioritization" and (state.get("queue_priority_by_patient") or {}):
            ranked_q = []
            for p in candidates:
                q = (state.get("queue_priority_by_patient") or {}).get(p.patient_id)
                if isinstance(q, QueuePrioritizationSchema):
                    ranked_q.append((p, float(q.priority_score)))
            ranked_q.sort(key=lambda x: x[1], reverse=True)
            if sel.method in {"threshold", "threshold_then_top_k"} and sel.threshold is not None:
                ranked_q = [x for x in ranked_q if x[1] >= float(sel.threshold)]
            return [p for p, _ in (ranked_q if sel.method == "threshold" else ranked_q[: sel.k])]

        if sel.method == "first_k":
            return candidates[: sel.k]

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

    def _run_task_for_patient(
        self,
        task_name: str,
        plan: ClinicPlanSchema,
        patient: PatientRecord,
        state: Dict[str, Any],
        *,
        use_cache: bool = True,
    ) -> Any:
        task = self.task_registry.get(task_name)
        if task is None:
            return None
        task_params = self._task_params(plan, task_name)

        cache_meta: Optional[Dict[str, Any]] = None
        if self.output_cache is not None and use_cache:
            try:
                cache_meta = self._cache_meta(task_name=task_name, plan=plan, patient=patient, task_params=task_params)
                cached_payload = self.output_cache.get(cache_meta)
                if cached_payload is not None:
                    out = self._deserialize_cached(task_name, cached_payload)
                    self._write_output_to_state(task_name=task_name, patient=patient, out=out, state=state)
                    if out is not None:
                        self._mark_task_source(state=state, patient_id=patient.patient_id, task_name=task_name, source="cache")
                    return out
            except Exception:
                cache_meta = None

        out = task.run(ctx=self.ctx, plan=plan, patient=patient, state=state, task_params=task_params)
        self._write_output_to_state(task_name=task_name, patient=patient, out=out, state=state)
        if out is not None:
            self._mark_task_source(state=state, patient_id=patient.patient_id, task_name=task_name, source="fresh")

        if self.output_cache is not None and out is not None:
            try:
                if cache_meta is None:
                    cache_meta = self._cache_meta(task_name=task_name, plan=plan, patient=patient, task_params=task_params)
                self.output_cache.set(cache_meta, self._serialize_for_cache(out))
            except Exception:
                pass
        return out

    def run_weekly_review(self, *, patients: List[PatientRecord], plan: ClinicPlanSchema, run_date: Optional[str] = None) -> ReviewBundleSchema:
        run_date = run_date or dt.date.today().isoformat()
        state: Dict[str, Any] = {}

        if self._task_enabled(plan, "intake_workflow") and self.task_registry.get("intake_workflow") and patients:
            out = self._run_task_for_patient("intake_workflow", plan, patients[0], state, use_cache=True)
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
                out = self._run_task_for_patient(ts.name, plan, p, state, use_cache=True)
                if out is not None:
                    state.setdefault(self._result_store_key(ts.name), {})[p.patient_id] = out

            risk_obj = (state.get("risk_by_patient") or {}).get(p.patient_id)
            summary_obj = (state.get("clinician_summary_by_patient") or {}).get(p.patient_id)
            referral_obj = (state.get("admin_referral_by_patient") or {}).get(p.patient_id)

            extra_outputs: Dict[str, Any] = {}
            for task_name in [
                t.name
                for t in plan.tasks
                if t.enabled and t.name not in {"intake_workflow", "risk_assessment", "clinician_summary", "admin_referral"}
            ]:
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
                    artifact_sources=self._get_task_sources_for_patient(state=state, patient_id=p.patient_id),
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
