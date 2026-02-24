from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import re
from typing import Any, Dict, List, Optional

from .llm_client import LLMJsonClient
from . import prompts
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


def _get_followup_gap(state: Dict[str, Any], patient_id: str) -> Optional[FollowupGapSchema]:
    fbp = state.get("followup_gap_by_patient") or {}
    f = fbp.get(patient_id)
    return f if isinstance(f, FollowupGapSchema) else None


def _get_guideline(state: Dict[str, Any], patient_id: str) -> Optional[GuidelineComparisonSchema]:
    gbp = state.get("guideline_comparison_by_patient") or {}
    g = gbp.get(patient_id)
    return g if isinstance(g, GuidelineComparisonSchema) else None


def _get_admin_referral(state: Dict[str, Any], patient_id: str) -> Optional[AdminReferralSchema]:
    abp = state.get("admin_referral_by_patient") or {}
    a = abp.get(patient_id)
    return a if isinstance(a, AdminReferralSchema) else None


def _patient_demographics(patient: PatientRecord) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key in ["patient_name", "date_of_birth", "sex", "ohip_number", "address", "phone"]:
        val = getattr(patient, key, None)
        if isinstance(val, str) and val.strip():
            out[key] = val.strip()
    return out


def _patient_notes_for_prompt(patient: PatientRecord) -> str:
    base_notes = str(getattr(patient, "longitudinal_notes", "") or "").strip()
    demo = _patient_demographics(patient)
    if not demo:
        return base_notes
    lines: List[str] = ["PATIENT HEADER (chart metadata):"]
    label_map = {
        "patient_name": "Patient name",
        "date_of_birth": "DOB",
        "sex": "Sex",
        "ohip_number": "OHIP",
        "address": "Address",
        "phone": "Phone",
    }
    for key in ["patient_name", "date_of_birth", "sex", "ohip_number", "address", "phone"]:
        if key in demo:
            lines.append(f"- {label_map[key]}: {demo[key]}")
    if getattr(patient, "longitudinal_note_entries", None):
        lines.append(f"- Longitudinal note count: {len(patient.longitudinal_note_entries)}")
    if base_notes:
        lines.extend(["", "LONGITUDINAL CLINIC NOTES:", base_notes])
    return "\n".join(lines).strip()


def _estimate_age_from_dob(dob_text: Optional[str]) -> Optional[int]:
    if not dob_text or not isinstance(dob_text, str):
        return None
    s = dob_text.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            dob = dt.datetime.strptime(s, fmt).date()
            today = dt.date.today()
            return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        except Exception:
            continue
    return None


def _clean_line(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _format_referral_letter_text(
    *,
    patient: PatientRecord,
    plan: ClinicPlanSchema,
    target: str,
    recipient: str,
    urgency: str,
    llm_body: str,
    referral: Optional[AdminReferralSchema],
    risk: Optional[RiskAssessmentSchema],
    summ: Optional[ClinicianSummarySchema],
) -> str:
    demo = _patient_demographics(patient)
    patient_name = demo.get("patient_name") or patient.patient_id
    dob = demo.get("date_of_birth") or "Not available in chart"
    ohip = demo.get("ohip_number") or "Not available in chart"
    addr = demo.get("address") or "Not available in chart"
    phone = demo.get("phone") or "Not available in chart"
    age = _estimate_age_from_dob(demo.get("date_of_birth"))
    sex = demo.get("sex") or "Not available in chart"

    clinic_name = ""
    try:
        wf = getattr(plan, "workflow", None)
        clinic_name = _clean_line(getattr(wf, "clinic_name", "")) if wf is not None else ""
    except Exception:
        clinic_name = ""
    if not clinic_name:
        clinic_name = "FacetCare Demo Clinic"

    referral_reason = ""
    request_line = ""
    if isinstance(referral, AdminReferralSchema):
        referral_reason = _clean_line(getattr(referral, "reason", "") or "")
        request_line = _clean_line(getattr(referral, "request_to_specialist", "") or "")
    if not referral_reason:
        referral_reason = f"Assessment and management of concerns related to {target.replace('_', ' ')}"
    if not request_line:
        request_line = "I would appreciate your assessment and recommendations for next steps."

    key_bullets: List[str] = []
    if risk is not None:
        try:
            key_bullets.append(f"Risk score {float(risk.risk_score):.2f} ({risk.risk_level})")
        except Exception:
            pass
        for x in list(getattr(risk, "reasons", []) or [])[:3]:
            x = _clean_line(x)
            if x:
                key_bullets.append(x)
    if summ is not None:
        for x in list(getattr(summ, "key_points", []) or [])[:4]:
            x = _clean_line(x)
            if x:
                key_bullets.append(x)
    if isinstance(referral, AdminReferralSchema):
        for x in list(getattr(referral, "required_pre_referral_steps", []) or [])[:3]:
            x = _clean_line(x)
            if x:
                key_bullets.append(x)
    deduped: List[str] = []
    seen: set[str] = set()
    for x in key_bullets:
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        deduped.append(x)
    key_bullets = deduped[:6] if deduped else ["Please see attached longitudinal clinic notes for the most relevant findings."]

    investigations_lines: List[str] = []
    for ln in str(getattr(patient, "longitudinal_notes", "") or "").splitlines():
        s = _clean_line(ln)
        low = s.lower()
        if any(tok in low for tok in ["ct", "mri", "ultrasound", "lab", "cbc", "lft", "a1c", "ferritin", "lipase", "amylase", "cmp"]):
            investigations_lines.append(s)
        if len(investigations_lines) >= 4:
            break
    if not investigations_lines:
        investigations_lines = ["Relevant investigations are described in the chart notes; exact values may be incomplete in free-text documentation."]

    history_line = "Relevant history and medications should be confirmed in the EMR medication/problem list."
    meds = list(getattr(summ, "medications_to_review", []) or []) if summ is not None else []
    meds = [_clean_line(m) for m in meds if _clean_line(m)]
    if meds:
        history_line = "Medications to review: " + "; ".join(meds[:5])

    llm_body_clean = str(llm_body or "").strip()
    has_structured_header = any(tag in llm_body_clean.upper() for tag in ["DATE:", "RE:", "REASON FOR REFERRAL:"])

    if has_structured_header:
        return llm_body_clean

    today_str = dt.date.today().isoformat()
    lines: List[str] = [
        clinic_name,
        "[Address / Phone / Fax]",
        "",
        f"DATE: {today_str}",
        f"TO: Dr. {recipient}",
        "FAX: Not available in chart",
        f"RE: {patient_name}",
        f"DOB: {dob}",
        f"OHIP No: {ohip}",
        f"Address: {addr}",
        f"Phone: {phone}",
        "",
        f"Dear Dr. {recipient},",
        "",
        f"Reason for Referral: {referral_reason}.",
    ]
    if age is not None:
        if sex != "Not available in chart":
            lines.append(f"Background: {patient_name} is a {age}-year-old patient ({sex}) followed in family medicine.")
        else:
            lines.append(f"Background: {patient_name} is a {age}-year-old patient followed in family medicine.")
    else:
        lines.append(f"Background: {patient_name} is a patient followed in family medicine.")
    lines.extend(["", "Key Clinical Findings:"])
    for b in key_bullets:
        lines.append(f"- {b}")
    lines.extend(["", "Investigations Completed:"])
    for inv in investigations_lines:
        lines.append(f"- {inv}")
    lines.extend(["", "Relevant History & Medications:", f"- {history_line}", "- Allergies: Not available in chart", "", f"Request: {request_line}"])
    if llm_body_clean:
        lines.extend(["", "Additional Clinical Context:", llm_body_clean])
    lines.extend(["", "Sincerely,", "[Referring Clinician Name]", "[Billing Number]"])
    return "\n".join(lines).strip()


class IntakeWorkflowTask(TaskBase):
    name = "intake_workflow"

    def run(self, *, ctx: TaskContext, plan: ClinicPlanSchema, patient: PatientRecord, state: Dict[str, Any], task_params: Optional[Dict[str, Any]] = None) -> Any:
        # Workflow is clinic-level, so only generate once.
        if state.get("workflow") is not None:
            return state["workflow"]

        system, user = prompts.intake_workflow_prompt(clinic_description=plan.clinic_description)
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

        system, user = prompts.risk_assessment_prompt(
            patient_id=patient.patient_id,
            target=target,
            horizon=horizon,
            notes=_patient_notes_for_prompt(patient),
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
        risk = _get_risk(state, patient.patient_id)
        summ = _get_summary(state, patient.patient_id)
        gap = _get_followup_gap(state, patient.patient_id)
        target, horizon = _plan_target_and_horizon(plan, task_params)
        system, user = prompts.queue_prioritization_prompt(
            clinic_goals=plan.clinic_description,
            workflow_json=workflow.model_dump_json(indent=2) if workflow else "{}",
            patient_id=patient.patient_id,
            notes=_patient_notes_for_prompt(patient),
            target=target,
            horizon=horizon,
            risk_json=risk.model_dump_json(indent=2) if risk else "none",
            summary_json=summ.model_dump_json(indent=2) if summ else "none",
            followup_gap_json=gap.model_dump_json(indent=2) if gap else "none",
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
        system, user = prompts.clinician_summary_prompt(
            patient_id=patient.patient_id,
            target=target,
            horizon=horizon,
            risk_json=risk.model_dump_json(indent=2) if risk else "not provided",
            notes=_patient_notes_for_prompt(patient),
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
        guideline = _get_guideline(state, patient.patient_id)
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
        system, user = prompts.admin_referral_prompt(
            patient_id=patient.patient_id,
            target=target,
            destination=dest,
            risk_json=risk.model_dump_json(indent=2) if risk else "none",
            summary_json=summ.model_dump_json(indent=2) if summ else "none",
            guideline_json=guideline.model_dump_json(indent=2) if guideline else "none",
            notes=_patient_notes_for_prompt(patient),
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
        gap = _get_followup_gap(state, patient.patient_id)
        admin_ref = _get_admin_referral(state, patient.patient_id)
        target, horizon = _plan_target_and_horizon(plan, task_params)
        if risk:
            target, horizon = risk.target_condition, risk.horizon_months
        system, user = prompts.patient_instructions_prompt(
            patient_id=patient.patient_id,
            target=target,
            horizon=horizon,
            summary_json=summ.model_dump_json(indent=2) if summ else "none",
            risk_json=risk.model_dump_json(indent=2) if risk else "none",
            followup_gap_json=gap.model_dump_json(indent=2) if gap else "none",
            admin_referral_json=admin_ref.model_dump_json(indent=2) if admin_ref else "none",
            notes=_patient_notes_for_prompt(patient),
        )
        obj = ctx.llm.json_object_no_tools(system=system, user=user, temperature=0.0)
        instructions = ensure_list_str(obj.get("instructions")) or [
            "Follow the clinic's recommended follow-up timeline.",
            "Seek urgent care if red-flag symptoms occur or worsen.",
        ]
        out = PatientInstructionsSchema(patient_id=patient.patient_id, target_condition=target, horizon_months=horizon, instructions=instructions)
        state.setdefault("patient_instructions_by_patient", {})[patient.patient_id] = out
        return out


class ResultsSummaryTask(TaskBase):
    name = "results_summary"

    def run(self, *, ctx: TaskContext, plan: ClinicPlanSchema, patient: PatientRecord, state: Dict[str, Any], task_params: Optional[Dict[str, Any]] = None) -> ResultsSummarySchema:
        target, horizon = _plan_target_and_horizon(plan, task_params)
        system, user = prompts.results_summary_prompt(
            patient_id=patient.patient_id,
            target=target,
            horizon=horizon,
            notes=_patient_notes_for_prompt(patient),
        )
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
        summ = _get_summary(state, patient.patient_id)
        target, _ = _plan_target_and_horizon(plan, task_params)
        urgency = referral.urgency if isinstance(referral, AdminReferralSchema) else ("urgent" if risk and risk.risk_level == "high" else "routine")
        recipient = referral.destination_service if isinstance(referral, AdminReferralSchema) else "Consult service"

        system, user = prompts.referral_letter_prompt(
            patient_id=patient.patient_id,
            target=target,
            urgency=urgency,
            recipient=recipient,
            referral_json=referral.model_dump_json(indent=2) if isinstance(referral, AdminReferralSchema) else "none",
            risk_json=risk.model_dump_json(indent=2) if risk else "none",
            summary_json=summ.model_dump_json(indent=2) if summ else "none",
            notes=_patient_notes_for_prompt(patient),
        )
        obj = ctx.llm.json_object_no_tools(system=system, user=user, temperature=0.0)
        raw_body = first_non_empty(obj.get("letter_body"), default=f"Referral request for {patient.patient_id} for further assessment.")
        body = _format_referral_letter_text(
            patient=patient,
            plan=plan,
            target=target,
            recipient=recipient,
            urgency=urgency,
            llm_body=raw_body,
            referral=referral if isinstance(referral, AdminReferralSchema) else None,
            risk=risk,
            summ=summ,
        )
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
        system, user = prompts.differential_prompt(
            patient_id=patient.patient_id,
            target=target,
            horizon=horizon,
            summary_json=summ.model_dump_json(indent=2) if summ else "none",
            notes=_patient_notes_for_prompt(patient),
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
        risk = _get_risk(state, patient.patient_id)
        summ = _get_summary(state, patient.patient_id)
        system, user = prompts.guideline_comparison_prompt(
            patient_id=patient.patient_id,
            target=target,
            horizon=horizon,
            notes=_patient_notes_for_prompt(patient),
            risk_json=risk.model_dump_json(indent=2) if risk else "none",
            summary_json=summ.model_dump_json(indent=2) if summ else "none",
        )
        obj = ctx.llm.json_object_no_tools(system=system, user=user, temperature=0.0)
        out = GuidelineComparisonSchema(
            patient_id=patient.patient_id,
            target_condition=target,
            horizon_months=horizon,
            recommended_guidelines=ensure_list_str(obj.get("recommended_guidelines") or ["Use local clinic and specialty pathway guidelines"]),
            evidence_summary=first_non_empty(obj.get("evidence_summary"), default="Guideline mapping requires clinician confirmation."),
        )
        state.setdefault("guideline_comparison_by_patient", {})[patient.patient_id] = out
        return out


class FollowupGapDetectionTask(TaskBase):
    name = "followup_gap_detection"

    def run(self, *, ctx: TaskContext, plan: ClinicPlanSchema, patient: PatientRecord, state: Dict[str, Any], task_params: Optional[Dict[str, Any]] = None) -> FollowupGapSchema:
        summ = _get_summary(state, patient.patient_id)
        risk = _get_risk(state, patient.patient_id)
        system, user = prompts.followup_gap_prompt(
            clinic_goals=plan.clinic_description,
            patient_id=patient.patient_id,
            notes=_patient_notes_for_prompt(patient),
            summary_json=summ.model_dump_json(indent=2) if summ else "none",
            risk_json=risk.model_dump_json(indent=2) if risk else "none",
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
        admin_ref = state.get("admin_referral_by_patient", {}).get(patient.patient_id)
        system, user = prompts.referral_intake_checklist_prompt(
            destination=dest,
            patient_id=patient.patient_id,
            notes=_patient_notes_for_prompt(patient),
            admin_referral_json=admin_ref.model_dump_json(indent=2) if isinstance(admin_ref, AdminReferralSchema) else "none",
            workflow_json=workflow.model_dump_json(indent=2) if isinstance(workflow, ClinicWorkflowSchema) else "none",
        )
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
        system, user = prompts.lab_trend_prompt(
            timeframe=timeframe,
            patient_id=patient.patient_id,
            notes=_patient_notes_for_prompt(patient),
        )
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
        system, user = prompts.care_plan_reconciliation_prompt(
            clinic_goals=plan.clinic_description,
            patient_id=patient.patient_id,
            notes=_patient_notes_for_prompt(patient),
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
