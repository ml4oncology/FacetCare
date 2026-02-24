from __future__ import annotations

from typing import Iterable, Optional, Tuple


JSON_ONLY_RULE = (
    "Return exactly one JSON object that matches the requested schema. "
    "No markdown, no code fences, no commentary, no extra keys."
)


PROMPT_TEMPLATE_VERSIONS = {
    "clinic_plan": "2026-02-22-v1",
    "intake_workflow": "2026-02-22-v1",
    "risk_assessment": "2026-02-22-v1",
    "queue_prioritization": "2026-02-22-v1",
    "clinician_summary": "2026-02-22-v1",
    "admin_referral": "2026-02-22-v1",
    "patient_instructions": "2026-02-22-v1",
    "results_summary": "2026-02-22-v1",
    "referral_letter": "2026-02-22-v1",
    "differential_diagnosis": "2026-02-22-v1",
    "guideline_comparison": "2026-02-22-v1",
    "followup_gap_detection": "2026-02-22-v1",
    "referral_intake_checklist": "2026-02-22-v1",
    "lab_trend_summary": "2026-02-22-v1",
    "care_plan_reconciliation": "2026-02-22-v1",
    "transcription": "2026-02-22-v1",
}


def _join_context(lines: Iterable[str]) -> str:
    return "\n".join([ln for ln in lines if ln is not None and str(ln).strip()])


def _notes_block(notes: str) -> str:
    txt = (notes or "").strip()
    return f"Notes:\n{txt}" if txt else "Notes:\n[No notes provided]"


def clinic_plan_prompt(
    *,
    clinic_description: str,
    default_target_condition: Optional[str],
    default_horizon_months: Optional[int],
    valid_task_names: Iterable[str],
) -> Tuple[str, str]:
    system = (
        "You are a clinic program designer for a schema-driven orchestration app.\n"
        f"{JSON_ONLY_RULE}\n\n"
        "Goal: convert a natural-language clinic workflow description into a practical plan.\n"
        "Prioritize real workflow fit, not generic AI task lists.\n"
        "Do not force disease risk scoring if the clinic asks for another workflow.\n"
        "Only enable tasks that support the stated clinic goals.\n\n"
        "Valid task names are:\n"
        f"{', '.join(valid_task_names)}\n\n"
        "constraints.candidate_pool.strategy must be one of: all, recent_notes_only, keyword_prefilter\n"
        "constraints.selection.method must be one of: top_k, threshold, threshold_then_top_k, first_k\n"
        "If no scoring task is requested, use selection.method=first_k and selection.source_task=null.\n"
        "If risk_assessment is enabled, target_condition and horizon_months are usually helpful.\n"
        "If not a disease-risk workflow, target_condition and horizon_months can be null.\n"
        "Use conservative defaults for review volume and deduplication when the clinic is vague."
    )
    user = (
        "Clinic description and goals:\n"
        f"{clinic_description}\n\n"
        f"Default target_condition (optional): {default_target_condition}\n"
        f"Default horizon_months (optional): {default_horizon_months}\n"
    )
    return system, user


def intake_workflow_prompt(*, clinic_description: str) -> Tuple[str, str]:
    system = (
        "You design clinic workflow profiles for downstream agent orchestration. "
        "Return one JSON object matching ClinicWorkflowSchema. "
        "Infer a plausible workflow from the description, but keep assumptions explicit in goals or staffing_constraints. "
        "Prefer concise, operational phrasing. "
        f"{JSON_ONLY_RULE}"
    )
    user = f"Clinic description:\n{clinic_description}"
    return system, user


def risk_assessment_prompt(*, patient_id: str, target: str, horizon: int, notes: str) -> Tuple[str, str]:
    system = (
        "You are a clinical risk-triage assistant for longitudinal primary-care notes. "
        "Return one JSON object for RiskAssessmentSchema. "
        "Estimate a near-term risk probability from the note text only. "
        "This is NOT a diagnosis. Do not claim certainty.\n"
        "Use the full note history, including weak longitudinal signals (repeated symptoms, weight change, metabolic changes, unresolved workup, family history, care gaps).\n"
        "Calibrate conservatively when evidence is sparse.\n"
        "Key risk and protective factors should be specific note-grounded phrases, not generic textbook statements.\n"
        "Recommended next steps must be triage-oriented and realistic for primary care.\n"
        f"{JSON_ONLY_RULE}"
    )
    user = _join_context(
        [
            f"Patient ID: {patient_id}",
            f"Target condition: {target}",
            f"Horizon months: {horizon}",
            _notes_block(notes),
        ]
    )
    return system, user


def queue_prioritization_prompt(*, clinic_goals: str, workflow_json: str, patient_id: str, notes: str, target: str, horizon: int) -> Tuple[str, str]:
    system = (
        "Return one JSON object for QueuePrioritizationSchema. "
        "Prioritize how urgently this chart should be reviewed within the clinic workflow. "
        "Use note acuity, unresolved follow-up, and workflow constraints. "
        "priority_score must be 0 to 1. queue_reason should be specific and practical. "
        f"{JSON_ONLY_RULE}"
    )
    user = _join_context(
        [
            "Clinic goals:",
            clinic_goals,
            "",
            "Workflow context:",
            workflow_json,
            "",
            f"Patient {patient_id}",
            _notes_block(notes),
            "",
            f"Optional target: {target}; horizon={horizon} months",
        ]
    )
    return system, user


def clinician_summary_prompt(*, patient_id: str, target: str, horizon: int, risk_json: str, notes: str) -> Tuple[str, str]:
    system = (
        "Return one JSON object for ClinicianSummarySchema. "
        "Produce a high-value chart review summary for a busy clinician. "
        "summary_for_chart should be a complete review-ready narrative (usually 4 to 7 sentences), not a one-liner. "
        "Include the main concern, important longitudinal signals, uncertainty/limitations, and why follow-up matters now. "
        "suggested_orders and suggested_referrals must be actionable and realistic. "
        "safety_netting should contain explicit red-flag or follow-up precautions. "
        f"{JSON_ONLY_RULE}"
    )
    user = _join_context(
        [
            f"Patient ID: {patient_id}",
            f"Target condition: {target}",
            f"Horizon months: {horizon}",
            f"Risk context (if any): {risk_json}",
            "",
            _notes_block(notes),
        ]
    )
    return system, user


def admin_referral_prompt(*, patient_id: str, target: str, destination: str, risk_json: str, summary_json: str, notes: str) -> Tuple[str, str]:
    system = (
        "Return one JSON object for AdminReferralSchema. "
        "Create a referral-support package for clinic staff. "
        "reason_for_referral should be a detailed, chart-grounded explanation (2 to 5 sentences) with the clinical concern and urgency rationale. "
        "attach_documents should list practical documents to include. "
        "admin_notes should mention missing information or logistics checks when relevant. "
        f"{JSON_ONLY_RULE}"
    )
    user = _join_context(
        [
            f"Patient ID: {patient_id}",
            f"Target condition/context: {target}",
            f"Destination service: {destination}",
            f"Risk context: {risk_json}",
            f"Clinician summary: {summary_json}",
            _notes_block(notes),
        ]
    )
    return system, user


def patient_instructions_prompt(*, patient_id: str, target: str, horizon: int, summary_json: str, risk_json: str, notes: str) -> Tuple[str, str]:
    system = (
        "Return one JSON object for PatientInstructionsSchema with patient-friendly language. "
        "Instructions should be clear, specific, and non-alarming. "
        "Do not diagnose. Focus on follow-up, monitoring, and safety-net advice. "
        f"{JSON_ONLY_RULE}"
    )
    user = _join_context(
        [
            f"Patient {patient_id}; context target={target}; horizon={horizon}",
            f"Clinician summary: {summary_json}",
            f"Risk context: {risk_json}",
            _notes_block(notes),
        ]
    )
    return system, user


def results_summary_prompt(*, patient_id: str, target: str, horizon: int, notes: str) -> Tuple[str, str]:
    system = (
        "Return one JSON object for ResultsSummarySchema. "
        "Summarize labs, imaging, and trends only from what is present in the notes. "
        "If exact values are absent, explicitly state that and summarize qualitative trends or uncertainty. "
        f"{JSON_ONLY_RULE}"
    )
    user = _join_context([f"Patient {patient_id}; target={target}; horizon={horizon}", _notes_block(notes)])
    return system, user


def referral_letter_prompt(*, patient_id: str, target: str, urgency: str, recipient: str, referral_json: str, risk_json: str, summary_json: str, notes: str) -> Tuple[str, str]:
    system = (
        "Return one JSON object for ReferralLetterSchema. "
        "Write a complete referral letter body that can be pasted into a chart or fax template. "
        "letter_body should be a full note (typically 1 to 3 short paragraphs), not a sentence fragment. "
        "Include reason for referral, key longitudinal findings, relevant context, and what is being requested from the consultant. "
        "Stay factual and note-based. Avoid unsupported claims. "
        f"{JSON_ONLY_RULE}"
    )
    user = _join_context(
        [
            f"Patient {patient_id}; target={target}; urgency={urgency}; recipient={recipient}",
            f"Referral context: {referral_json}",
            f"Risk context: {risk_json}",
            f"Clinician summary: {summary_json}",
            _notes_block(notes),
        ]
    )
    return system, user


def differential_prompt(*, patient_id: str, target: str, horizon: int, summary_json: str, notes: str) -> Tuple[str, str]:
    system = (
        "Return one JSON object for DifferentialDiagnosisSchema. "
        "This is a differential brainstorming aid only. No certainty claims. "
        "possible_diagnoses should be ordered by plausibility from the available notes and include non-malignant alternatives when appropriate. "
        "reasoning should summarize the note-grounded rationale and uncertainty. "
        f"{JSON_ONLY_RULE}"
    )
    user = _join_context([f"Patient {patient_id}; target={target}; horizon={horizon}", f"Summary: {summary_json}", _notes_block(notes)])
    return system, user


def guideline_comparison_prompt(*, patient_id: str, target: str, horizon: int, notes: str) -> Tuple[str, str]:
    system = (
        "Return one JSON object for GuidelineComparisonSchema. "
        "Map the patient scenario to likely guideline families or pathways (primary care, GI, oncology, urgent evaluation) and summarize how the notes relate. "
        "Do not cite exact guideline text if not provided. Keep it high-level and clinically practical. "
        f"{JSON_ONLY_RULE}"
    )
    user = _join_context([f"Patient {patient_id}; target={target}; horizon={horizon}", _notes_block(notes)])
    return system, user


def followup_gap_prompt(*, clinic_goals: str, patient_id: str, notes: str) -> Tuple[str, str]:
    system = (
        "Return one JSON object for FollowupGapSchema. "
        "Identify possible follow-up gaps, missed tests, unresolved symptoms, or ambiguous completions from note text. "
        "Distinguish observed signals from assumptions. suggested_actions should be practical next chart actions. "
        f"{JSON_ONLY_RULE}"
    )
    user = _join_context(["Clinic goals:", clinic_goals, "", f"Patient {patient_id}", _notes_block(notes)])
    return system, user


def referral_intake_checklist_prompt(*, destination: str, patient_id: str, notes: str) -> Tuple[str, str]:
    system = (
        "Return one JSON object for ReferralIntakeChecklistSchema. "
        "Build a referral intake readiness checklist for the destination service. "
        "Separate what is available in the notes from what is still missing. "
        f"{JSON_ONLY_RULE}"
    )
    user = _join_context([f"Destination: {destination}", f"Patient {patient_id}", _notes_block(notes)])
    return system, user


def lab_trend_prompt(*, timeframe: str, patient_id: str, notes: str) -> Tuple[str, str]:
    system = (
        "Return one JSON object for LabTrendSummarySchema. "
        "If exact numeric labs are absent, summarize qualitative trends and limitations transparently. "
        "Provide both clinician-facing and patient-friendly summaries. "
        f"{JSON_ONLY_RULE}"
    )
    user = _join_context([f"Timeframe: {timeframe}", f"Patient {patient_id}", _notes_block(notes)])
    return system, user


def care_plan_reconciliation_prompt(*, clinic_goals: str, patient_id: str, notes: str) -> Tuple[str, str]:
    system = (
        "Return one JSON object for CarePlanReconciliationSchema. "
        "Reconcile prior plans, completed items, unresolved items, and changed plans from free-text notes. "
        "Prefer concrete chart actions over vague statements. "
        f"{JSON_ONLY_RULE}"
    )
    user = _join_context(["Clinic goals:", clinic_goals, "", f"Patient {patient_id}", _notes_block(notes)])
    return system, user
