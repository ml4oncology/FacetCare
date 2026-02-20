from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .llm_client import LLMJsonClient
from .normalizers import coerce_int, coerce_probability, ensure_list_str, first_non_empty, normalize_clinician_summary_payload, normalize_risk_payload
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
)


@dataclass
class TaskContext:
    llm: LLMJsonClient


class TaskBase:
    name: str = "base"

    def run(
        self,
        *,
        ctx: TaskContext,
        plan: ClinicPlanSchema,
        patient: PatientRecord,
        state: Dict[str, Any],
        task_params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        raise NotImplementedError


def _plan_target_and_horizon(plan: ClinicPlanSchema, task_params: Optional[Dict[str, Any]], fallback_target: str = "general_clinical_review") -> tuple[str, int]:
    params = task_params or {}
    target = first_non_empty(params.get("target_condition"), plan.target_condition, default=fallback_target)
    horizon = coerce_int(params.get("horizon_months", plan.horizon_months or 12), 12, lo=1, hi=120)
    return target, horizon


def _get_risk(state: Dict[str, Any], patient_id: str) -> Optional[RiskAssessmentSchema]:
    rbp = state.get("risk_by_patient") or {}
    r = rbp.get(patient_id)
    return r if isinstance(r, RiskAssessmentSchema) else None


def _get_summary(state: Dict[str, Any], patient_id: str) -> Optional[ClinicianSummarySchema]:
    sbp = state.get("clinician_summary_by_patient") or {}
    s = sbp.get(patient_id)
    return s if isinstance(s, ClinicianSummarySchema) else None


class IntakeWorkflowTask(TaskBase):
    name = "intake_workflow"

    def run(self, *, ctx: TaskContext, plan: ClinicPlanSchema, patient: PatientRecord, state: Dict[str, Any], task_params: Optional[Dict[str, Any]] = None) -> Any:
        # Workflow is clinic-level, so only generate once.
        if state.get("workflow") is not None:
            return state["workflow"]

        system = (
            "You design a clinic workflow profile. Return one JSON object matching ClinicWorkflowSchema. "
            "No markdown, no commentary."
        )
        user = f"Clinic description:\n{plan.clinic_description}"
        obj = ctx.llm.json_object_no_tools(system=system, user=user, temperature=0.0)
        from .normalizers import normalize_workflow_payload

        workflow = ClinicWorkflowSchema.model_validate(normalize_workflow_payload(obj))
        state["workflow"] = workflow
        return workflow


class RiskAssessmentTask(TaskBase):
    name = "risk_assessment"

    def run(self, *, ctx: TaskContext, plan: ClinicPlanSchema, patient: PatientRecord, state: Dict[str, Any], task_params: Optional[Dict[str, Any]] = None) -> RiskAssessmentSchema:
        target, horizon = _plan_target_and_horizon(plan, task_params, fallback_target="risk_target")
        policy = plan.constraints.risk_level_policy
        tp = task_params or {}
        if isinstance(tp.get("risk_level_policy"), dict):
            rlp = tp["risk_level_policy"]
            low = coerce_probability(rlp.get("low_lt", policy.low_lt), default=policy.low_lt)
            mod = coerce_probability(rlp.get("moderate_lt", policy.moderate_lt), default=policy.moderate_lt)
            # local policy override without mutating plan
            class _Tmp:
                def __init__(self, low_lt: float, moderate_lt: float):
                    self.low_lt = low_lt
                    self.moderate_lt = moderate_lt
                def to_level(self, p: float) -> str:
                    if p < self.low_lt:
                        return "low"
                    if p < self.moderate_lt:
                        return "moderate"
                    return "high"
            policy_obj: Any = _Tmp(low, mod)
        else:
            policy_obj = policy

        system = (
            "Return ONE JSON object for RiskAssessmentSchema. "
            "Estimate risk probability from longitudinal notes only. No diagnosis."
        )
        user = (
            f"Patient ID: {patient.patient_id}\n"
            f"Target condition: {target}\n"
            f"Horizon months: {horizon}\n"
            f"Notes:\n{patient.longitudinal_notes}"
        )
        obj = ctx.llm.json_object_no_tools(system=system, user=user, temperature=0.0)
        rp = coerce_probability(obj.get("risk_probability"), default=0.01)
        risk_level = policy_obj.to_level(rp)
        payload = normalize_risk_payload(obj, patient_id=patient.patient_id, target_condition=target, horizon_months=horizon, risk_level=risk_level)
        out = RiskAssessmentSchema.model_validate(payload)
        state.setdefault("risk_by_patient", {})[patient.patient_id] = out
        return out


class QueuePrioritizationTask(TaskBase):
    name = "queue_prioritization"

    def run(self, *, ctx: TaskContext, plan: ClinicPlanSchema, patient: PatientRecord, state: Dict[str, Any], task_params: Optional[Dict[str, Any]] = None) -> QueuePrioritizationSchema:
        workflow = state.get("workflow")
        target, horizon = _plan_target_and_horizon(plan, task_params)
        system = "Return one JSON object for QueuePrioritizationSchema. No markdown."
        user = (
            f"Clinic goals:\n{plan.clinic_description}\n\n"
            f"Workflow context:\n{workflow.model_dump_json(indent=2) if workflow else '{}'}\n\n"
            f"Patient {patient.patient_id} notes:\n{patient.longitudinal_notes}\n\n"
            f"Optional target: {target}; horizon={horizon} months"
        )
        obj = ctx.llm.json_object_no_tools(system=system, user=user, temperature=0.0)
        score = coerce_probability(obj.get("priority_score"), default=0.25)
        level = first_non_empty(obj.get("priority_level"), default=("high" if score >= 0.6 else "moderate" if score >= 0.3 else "low"))
        if level not in {"low", "moderate", "high"}:
            level = "moderate"
        out = QueuePrioritizationSchema(
            patient_id=patient.patient_id,
            priority_score=score,
            priority_level=level,
            queue_reason=first_non_empty(obj.get("queue_reason"), default="Prioritized based on note acuity and follow-up needs."),
            recommended_window=first_non_empty(obj.get("recommended_window"), default="next routine review"),
        )
        state.setdefault("queue_priority_by_patient", {})[patient.patient_id] = out
        return out


class ClinicianSummaryTask(TaskBase):
    name = "clinician_summary"

    def run(self, *, ctx: TaskContext, plan: ClinicPlanSchema, patient: PatientRecord, state: Dict[str, Any], task_params: Optional[Dict[str, Any]] = None) -> ClinicianSummarySchema:
        risk = _get_risk(state, patient.patient_id)
        target, horizon = _plan_target_and_horizon(plan, task_params)
        if risk is not None:
            target, horizon = risk.target_condition, risk.horizon_months
        system = "Return ONE JSON object for ClinicianSummarySchema. No markdown."
        user = (
            f"Patient ID: {patient.patient_id}\n"
            f"Target condition: {target}\nHorizon months: {horizon}\n"
            f"Risk context (if any): {risk.model_dump_json(indent=2) if risk else 'not provided'}\n\n"
            f"Notes:\n{patient.longitudinal_notes}"
        )
        obj = ctx.llm.json_object_no_tools(system=system, user=user, temperature=0.0)
        payload = normalize_clinician_summary_payload(obj, patient_id=patient.patient_id, target_condition=target, horizon_months=horizon)
        out = ClinicianSummarySchema.model_validate(payload)
        state.setdefault("clinician_summary_by_patient", {})[patient.patient_id] = out
        return out


class AdminReferralTask(TaskBase):
    name = "admin_referral"

    def run(self, *, ctx: TaskContext, plan: ClinicPlanSchema, patient: PatientRecord, state: Dict[str, Any], task_params: Optional[Dict[str, Any]] = None) -> AdminReferralSchema:
        risk = _get_risk(state, patient.patient_id)
        summ = _get_summary(state, patient.patient_id)
        target, _ = _plan_target_and_horizon(plan, task_params)
        if risk is not None:
            target = risk.target_condition
        workflow = state.get("workflow")
        default_dest = "Specialty clinic"
        if workflow is not None:
            try:
                default_dest = workflow.referral_pathway.external
            except Exception:
                pass
        tp = task_params or {}
        dest = first_non_empty(tp.get("destination_service"), default=default_dest)

        urgency = "routine"
        if risk is not None:
            urgency = "urgent" if risk.risk_level == "high" else ("semi-urgent" if risk.risk_level == "moderate" else "routine")
        system = "Return one JSON object for AdminReferralSchema. No markdown."
        user = (
            f"Patient ID: {patient.patient_id}\n"
            f"Target condition/context: {target}\n"
            f"Destination service: {dest}\n"
            f"Risk context: {risk.model_dump_json(indent=2) if risk else 'none'}\n"
            f"Clinician summary: {summ.model_dump_json(indent=2) if summ else 'none'}\n"
            f"Notes:\n{patient.longitudinal_notes}"
        )
        obj = ctx.llm.json_object_no_tools(system=system, user=user, temperature=0.0)
        out = AdminReferralSchema(
            patient_id=patient.patient_id,
            target_condition=first_non_empty(obj.get("target_condition"), target),
            urgency=first_non_empty(obj.get("urgency"), default=urgency) if first_non_empty(obj.get("urgency"), default=urgency) in {"routine", "semi-urgent", "urgent"} else urgency,
            destination_service=first_non_empty(obj.get("destination_service"), default=dest),
            reason_for_referral=first_non_empty(obj.get("reason_for_referral"), default="Clinical review identified need for specialty assessment."),
            attach_documents=ensure_list_str(obj.get("attach_documents") or ["Recent clinic notes", "Medication list", "Relevant labs/imaging"]),
            admin_notes=ensure_list_str(obj.get("admin_notes") or ["Verify referral completeness before sending"]),
        )
        state.setdefault("admin_referral_by_patient", {})[patient.patient_id] = out
        return out


class PatientInstructionsTask(TaskBase):
    name = "patient_instructions"

    def run(self, *, ctx: TaskContext, plan: ClinicPlanSchema, patient: PatientRecord, state: Dict[str, Any], task_params: Optional[Dict[str, Any]] = None) -> PatientInstructionsSchema:
        risk = _get_risk(state, patient.patient_id)
        summ = _get_summary(state, patient.patient_id)
        target, horizon = _plan_target_and_horizon(plan, task_params)
        if risk:
            target, horizon = risk.target_condition, risk.horizon_months
        system = "Return one JSON object for PatientInstructionsSchema with patient-friendly language. No markdown."
        user = (
            f"Patient {patient.patient_id}; context target={target}; horizon={horizon}\n"
            f"Clinician summary: {summ.model_dump_json(indent=2) if summ else 'none'}\n"
            f"Risk context: {risk.model_dump_json(indent=2) if risk else 'none'}\n"
            f"Notes:\n{patient.longitudinal_notes}"
        )
        obj = ctx.llm.json_object_no_tools(system=system, user=user, temperature=0.0)
        instructions = ensure_list_str(obj.get("instructions")) or [
            "Follow the clinic's recommended follow-up timeline.",
            "Seek urgent care if red-flag symptoms occur or worsen.",
        ]
        return PatientInstructionsSchema(patient_id=patient.patient_id, target_condition=target, horizon_months=horizon, instructions=instructions)


class ResultsSummaryTask(TaskBase):
    name = "results_summary"

    def run(self, *, ctx: TaskContext, plan: ClinicPlanSchema, patient: PatientRecord, state: Dict[str, Any], task_params: Optional[Dict[str, Any]] = None) -> ResultsSummarySchema:
        target, horizon = _plan_target_and_horizon(plan, task_params)
        system = "Return one JSON object for ResultsSummarySchema. If no explicit labs/imaging, summarize that limitation."
        user = f"Patient {patient.patient_id}; target={target}; horizon={horizon}\nNotes:\n{patient.longitudinal_notes}"
        obj = ctx.llm.json_object_no_tools(system=system, user=user, temperature=0.0)
        return ResultsSummarySchema(
            patient_id=patient.patient_id,
            target_condition=target,
            horizon_months=horizon,
            labs_summary=first_non_empty(obj.get("labs_summary"), default="No structured lab values were provided in the notes."),
            imaging_summary=first_non_empty(obj.get("imaging_summary"), default="No imaging details found in the provided notes."),
            trending_summary=first_non_empty(obj.get("trending_summary"), default="Trend assessment limited to free-text note patterns."),
        )


class ReferralLetterTask(TaskBase):
    name = "referral_letter"

    def run(self, *, ctx: TaskContext, plan: ClinicPlanSchema, patient: PatientRecord, state: Dict[str, Any], task_params: Optional[Dict[str, Any]] = None) -> ReferralLetterSchema:
        referral = state.get("admin_referral_by_patient", {}).get(patient.patient_id)
        risk = _get_risk(state, patient.patient_id)
        target, _ = _plan_target_and_horizon(plan, task_params)
        urgency = referral.urgency if isinstance(referral, AdminReferralSchema) else ("urgent" if risk and risk.risk_level == "high" else "routine")
        recipient = referral.destination_service if isinstance(referral, AdminReferralSchema) else "Consult service"

        system = "Return one JSON object for ReferralLetterSchema."
        user = (
            f"Patient {patient.patient_id}; target={target}; urgency={urgency}; recipient={recipient}\n"
            f"Referral context: {referral.model_dump_json(indent=2) if referral else 'none'}\n"
            f"Notes:\n{patient.longitudinal_notes}"
        )
        obj = ctx.llm.json_object_no_tools(system=system, user=user, temperature=0.0)
        body = first_non_empty(obj.get("letter_body"), default=f"Referral request for {patient.patient_id} for further assessment.")
        return ReferralLetterSchema(
            patient_id=patient.patient_id,
            target_condition=first_non_empty(obj.get("target_condition"), default=target),
            urgency=(first_non_empty(obj.get("urgency"), default=urgency) if first_non_empty(obj.get("urgency"), default=urgency) in {"routine", "semi-urgent", "urgent"} else urgency),
            recipient=first_non_empty(obj.get("recipient"), default=recipient),
            letter_body=body,
            attachments=ensure_list_str(obj.get("attachments") or ["Recent notes", "Medication list", "Pertinent results"]),
        )


class DifferentialDiagnosisTask(TaskBase):
    name = "differential_diagnosis"

    def run(self, *, ctx: TaskContext, plan: ClinicPlanSchema, patient: PatientRecord, state: Dict[str, Any], task_params: Optional[Dict[str, Any]] = None) -> DifferentialDiagnosisSchema:
        target, horizon = _plan_target_and_horizon(plan, task_params)
        summ = _get_summary(state, patient.patient_id)
        system = "Return one JSON object for DifferentialDiagnosisSchema. No diagnosis certainty statements."
        user = (
            f"Patient {patient.patient_id}; target={target}; horizon={horizon}\n"
            f"Summary: {summ.model_dump_json(indent=2) if summ else 'none'}\nNotes:\n{patient.longitudinal_notes}"
        )
        obj = ctx.llm.json_object_no_tools(system=system, user=user, temperature=0.0)
        return DifferentialDiagnosisSchema(
            patient_id=patient.patient_id,
            target_condition=target,
            horizon_months=horizon,
            possible_diagnoses=ensure_list_str(obj.get("possible_diagnoses") or ["Insufficient data to propose a useful differential"]) ,
            reasoning=first_non_empty(obj.get("reasoning"), default="Use as a brainstorming aid only; confirm clinically."),
        )


class GuidelineComparisonTask(TaskBase):
    name = "guideline_comparison"

    def run(self, *, ctx: TaskContext, plan: ClinicPlanSchema, patient: PatientRecord, state: Dict[str, Any], task_params: Optional[Dict[str, Any]] = None) -> GuidelineComparisonSchema:
        target, horizon = _plan_target_and_horizon(plan, task_params)
        system = "Return one JSON object for GuidelineComparisonSchema."
        user = f"Patient {patient.patient_id}; target={target}; horizon={horizon}\nNotes:\n{patient.longitudinal_notes}"
        obj = ctx.llm.json_object_no_tools(system=system, user=user, temperature=0.0)
        return GuidelineComparisonSchema(
            patient_id=patient.patient_id,
            target_condition=target,
            horizon_months=horizon,
            recommended_guidelines=ensure_list_str(obj.get("recommended_guidelines") or ["Use local clinic and specialty pathway guidelines"]),
            evidence_summary=first_non_empty(obj.get("evidence_summary"), default="Guideline mapping requires clinician confirmation."),
        )


class FollowupGapDetectionTask(TaskBase):
    name = "followup_gap_detection"

    def run(self, *, ctx: TaskContext, plan: ClinicPlanSchema, patient: PatientRecord, state: Dict[str, Any], task_params: Optional[Dict[str, Any]] = None) -> FollowupGapSchema:
        system = "Return one JSON object for FollowupGapSchema."
        user = (
            f"Clinic goals:\n{plan.clinic_description}\n\n"
            f"Patient {patient.patient_id} notes:\n{patient.longitudinal_notes}\n\n"
            "Identify possible follow-up gaps, missed tests, or repeat symptoms needing review."
        )
        obj = ctx.llm.json_object_no_tools(system=system, user=user, temperature=0.0)
        sev = first_non_empty(obj.get("gap_severity"), default="moderate")
        if sev not in {"low", "moderate", "high"}:
            sev = "moderate"
        out = FollowupGapSchema(
            patient_id=patient.patient_id,
            pending_items=ensure_list_str(obj.get("pending_items") or ["Review chart for pending follow-up tasks"]),
            missed_followup_signals=ensure_list_str(obj.get("missed_followup_signals") or ["Recurrent or unresolved issue mentioned in notes"]),
            suggested_actions=ensure_list_str(obj.get("suggested_actions") or ["Schedule clinician review", "Confirm whether recommended tests were completed"]),
            gap_severity=sev,
        )
        state.setdefault("followup_gap_by_patient", {})[patient.patient_id] = out
        return out


class ReferralIntakeChecklistTask(TaskBase):
    name = "referral_intake_checklist"

    def run(self, *, ctx: TaskContext, plan: ClinicPlanSchema, patient: PatientRecord, state: Dict[str, Any], task_params: Optional[Dict[str, Any]] = None) -> ReferralIntakeChecklistSchema:
        workflow = state.get("workflow")
        default_dest = workflow.referral_pathway.external if isinstance(workflow, ClinicWorkflowSchema) else "Specialty clinic"
        dest = first_non_empty((task_params or {}).get("destination_service"), default=default_dest)
        system = "Return one JSON object for ReferralIntakeChecklistSchema."
        user = f"Destination: {dest}\nPatient {patient.patient_id} notes:\n{patient.longitudinal_notes}"
        obj = ctx.llm.json_object_no_tools(system=system, user=user, temperature=0.0)
        triage = first_non_empty(obj.get("triage_bucket"), default="routine")
        if triage not in {"routine", "semi-urgent", "urgent"}:
            triage = "routine"
        out = ReferralIntakeChecklistSchema(
            patient_id=patient.patient_id,
            destination_service=first_non_empty(obj.get("destination_service"), default=dest),
            triage_bucket=triage,
            available_info=ensure_list_str(obj.get("available_info") or ["Recent clinic notes"]),
            missing_info=ensure_list_str(obj.get("missing_info") or ["Referral-specific required fields should be verified"]),
            checklist_items=ensure_list_str(obj.get("checklist_items") or ["Confirm indication", "Attach key documents", "Verify contact details"]),
        )
        state.setdefault("referral_intake_by_patient", {})[patient.patient_id] = out
        return out


class LabTrendSummaryTask(TaskBase):
    name = "lab_trend_summary"

    def run(self, *, ctx: TaskContext, plan: ClinicPlanSchema, patient: PatientRecord, state: Dict[str, Any], task_params: Optional[Dict[str, Any]] = None) -> LabTrendSummarySchema:
        timeframe = first_non_empty((task_params or {}).get("timeframe_label"), plan.constraints.cadence, default="recent period")
        system = "Return one JSON object for LabTrendSummarySchema. If exact numeric labs are absent, summarize qualitative trends."
        user = f"Timeframe: {timeframe}\nPatient {patient.patient_id} notes:\n{patient.longitudinal_notes}"
        obj = ctx.llm.json_object_no_tools(system=system, user=user, temperature=0.0)
        out = LabTrendSummarySchema(
            patient_id=patient.patient_id,
            timeframe_label=first_non_empty(obj.get("timeframe_label"), default=timeframe),
            clinician_summary=first_non_empty(obj.get("clinician_summary"), default="Qualitative trend summary based on available note text."),
            patient_friendly_summary=first_non_empty(obj.get("patient_friendly_summary"), default="Your chart notes were reviewed for important result trends."),
            concerning_trends=ensure_list_str(obj.get("concerning_trends") or ["No explicit numeric trend available from free-text notes"]),
            suggested_next_steps=ensure_list_str(obj.get("suggested_next_steps") or ["Review actual lab reports for confirmation"]),
        )
        state.setdefault("lab_trend_by_patient", {})[patient.patient_id] = out
        return out


class CarePlanReconciliationTask(TaskBase):
    name = "care_plan_reconciliation"

    def run(self, *, ctx: TaskContext, plan: ClinicPlanSchema, patient: PatientRecord, state: Dict[str, Any], task_params: Optional[Dict[str, Any]] = None) -> CarePlanReconciliationSchema:
        system = "Return one JSON object for CarePlanReconciliationSchema."
        user = (
            f"Clinic goals:\n{plan.clinic_description}\n\n"
            f"Patient {patient.patient_id} notes:\n{patient.longitudinal_notes}\n\n"
            "Reconcile prior plans, unresolved items, and next actions from the notes."
        )
        obj = ctx.llm.json_object_no_tools(system=system, user=user, temperature=0.0)
        out = CarePlanReconciliationSchema(
            patient_id=patient.patient_id,
            prior_plan_items=ensure_list_str(obj.get("prior_plan_items") or ["Prior care plan items not explicitly structured in notes"]),
            completed_items=ensure_list_str(obj.get("completed_items")),
            unresolved_items=ensure_list_str(obj.get("unresolved_items") or ["Open items require clinician review"]),
            changed_items=ensure_list_str(obj.get("changed_items")),
            suggested_next_steps=ensure_list_str(obj.get("suggested_next_steps") or ["Confirm active plan with patient at next review"]),
        )
        state.setdefault("care_plan_recon_by_patient", {})[patient.patient_id] = out
        return out


class TranscriptionTask(TaskBase):
    name = "transcription"

    def run(self, *, ctx: TaskContext, plan: ClinicPlanSchema, patient: PatientRecord, state: Dict[str, Any], task_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Placeholder since demo repo does not process audio.
        return {"patient_id": patient.patient_id, "transcript": patient.longitudinal_notes}


def default_task_registry() -> Dict[str, TaskBase]:
    return {
        "intake_workflow": IntakeWorkflowTask(),
        "risk_assessment": RiskAssessmentTask(),
        "queue_prioritization": QueuePrioritizationTask(),
        "clinician_summary": ClinicianSummaryTask(),
        "admin_referral": AdminReferralTask(),
        "patient_instructions": PatientInstructionsTask(),
        "results_summary": ResultsSummaryTask(),
        "transcription": TranscriptionTask(),
        "referral_letter": ReferralLetterTask(),
        "differential_diagnosis": DifferentialDiagnosisTask(),
        "guideline_comparison": GuidelineComparisonTask(),
        "followup_gap_detection": FollowupGapDetectionTask(),
        "referral_intake_checklist": ReferralIntakeChecklistTask(),
        "lab_trend_summary": LabTrendSummaryTask(),
        "care_plan_reconciliation": CarePlanReconciliationTask(),
    }
