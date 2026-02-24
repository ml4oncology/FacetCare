from __future__ import annotations

from typing import Iterable, Optional, Tuple


JSON_ONLY_RULE = (
    "Return exactly one JSON object that matches the requested schema. "
    "No markdown, no code fences, no commentary, no extra keys."
)


PROMPT_TEMPLATE_VERSIONS = {
    "clinic_plan": "2026-02-23-v3",
    "intake_workflow": "2026-02-23-v3",
    "risk_assessment": "2026-02-23-v3",
    "queue_prioritization": "2026-02-23-v3",
    "clinician_summary": "2026-02-23-v3",
    "admin_referral": "2026-02-23-v3",
    "patient_instructions": "2026-02-23-v3",
    "results_summary": "2026-02-23-v3",
    "referral_letter": "2026-02-23-v3",
    "differential_diagnosis": "2026-02-23-v3",
    "guideline_comparison": "2026-02-23-v3",
    "followup_gap_detection": "2026-02-23-v3",
    "referral_intake_checklist": "2026-02-23-v3",
    "lab_trend_summary": "2026-02-23-v3",
    "care_plan_reconciliation": "2026-02-23-v3",
    "transcription": "2026-02-23-v3",
}


def _join_context(lines: Iterable[str]) -> str:
    return "\n".join([ln for ln in lines if ln is not None and str(ln).strip()])


def _notes_block(notes: str) -> str:
    txt = (notes or "").strip()
    return f"Notes:\n{txt}" if txt else "Notes:\n[No notes provided]"


def _schema_keys_block(name: str, keys: Iterable[str], extra: Optional[str] = None) -> str:
    parts = [
        f"Output schema: {name}",
        "Required keys (exact): " + ", ".join(keys),
        "Keep values JSON-native (strings, numbers, arrays).",
    ]
    if extra:
        parts.append(extra)
    return "\n".join(parts)


def _evidence_grounding_block() -> str:
    return (
        "Use note-grounded language. Prefer concrete chart phrases over generic disease facts. "
        "If evidence is missing, say so in the relevant text/list item instead of inventing details."
    )


def _output_discipline_block() -> str:
    return (
        "Work internally in this order before writing JSON: "
        "(1) extract explicit evidence from notes, "
        "(2) infer likely clinical signals cautiously, "
        "(3) map to the requested schema, "
        "(4) verify every required key is present, "
        "(5) remove unsupported claims. "
        "Do not reveal this reasoning. Return only the final JSON object."
    )


def _missing_data_rules_block() -> str:
    return (
        "Missing-data rules: keep all required keys even when evidence is incomplete. "
        "Use empty arrays for missing list fields, use brief uncertainty statements for missing text fields, "
        "and preserve requested labels/levels conservatively. Never invent dates, test values, or diagnoses."
    )


_FEW_SHOT_SNIPPETS = {
    "risk_assessment": (
        "Compact example style (not real patient): "
        '{"patient_id":"P-EX","target_condition":"pancreatic cancer","horizon_months":12,'
        '"risk_probability":0.18,"risk_level":"moderate",'
        '"key_risk_factors":["Progressive weight loss documented across visits","New diabetes/worsening glycemia noted"],'
        '"key_protective_factors":["No explicit jaundice documented"],'
        '"recommended_next_steps":["Review timing of prior abdominal imaging","Arrange clinician assessment for persistent alarm symptoms"],'
        '"safety_notes":["Risk estimate is note-based and not diagnostic"]}'
    ),
    "clinician_summary": (
        "Compact example style: "
        '{"patient_id":"P-EX","target_condition":"target","horizon_months":12,'
        '"summary_for_chart":"Longitudinal notes suggest a clinically relevant change pattern with incomplete workup and persistent symptoms. '
        'Risk context is uncertain but concern is above routine background. Review is warranted to confirm what has already been completed and what remains outstanding.",'
        '"suggested_orders":["Confirm whether previously suggested tests were completed"],'
        '"suggested_referrals":["Consider specialty referral if concerning pattern persists"],'
        '"safety_netting":["Document red-flag symptoms and clear return precautions"]}'
    ),
    "queue_prioritization": (
        "Compact example style: "
        '{"patient_id":"P-EX","priority_score":0.74,"priority_level":"high",'
        '"queue_reason":"Repeated unresolved symptoms plus documented missed follow-up make this chart time-sensitive.",'
        '"recommended_window":"within 1 week"}'
    ),
    "followup_gap_detection": (
        "Compact example style: "
        '{"patient_id":"P-EX","pending_items":["Verify whether imaging was completed"],'
        '"missed_followup_signals":["Abnormal symptom trend mentioned without clear closure"],'
        '"suggested_actions":["Contact patient to confirm completion of recommended workup"],'
        '"gap_severity":"moderate"}'
    ),
    "guideline_comparison": (
        "Compact example style: "
        '{"patient_id":"P-EX","target_condition":"target","horizon_months":12,'
        '"recommended_guidelines":["Primary care urgent symptom pathway","Local specialty referral triage pathway"],'
        '"evidence_summary":"Use the note pattern and documented risk context to map the patient to a local escalation or surveillance pathway. '
        'If hereditary or strong family history details are incomplete, note uncertainty and confirm before labeling eligibility."}'
    ),
    "admin_referral": (
        "Compact example style: "
        '{"patient_id":"P-EX","target_condition":"target","urgency":"semi-urgent","destination_service":"Specialty clinic",'
        '"reason_for_referral":"Referral requested for persistent concerning symptoms with unresolved follow-up and risk signals documented in longitudinal notes.",'
        '"attach_documents":["Recent clinic notes","Relevant labs/imaging"],'
        '"admin_notes":["Confirm referral contact details","Check missing prerequisite tests"]}'
    ),
}


def _few_shot_block(task_name: str) -> str:
    ex = _FEW_SHOT_SNIPPETS.get(task_name)
    return f"Formatting example:\\n{ex}" if ex else ""



def clinic_plan_prompt(*, clinic_description: str, default_target_condition: Optional[str], default_horizon_months: Optional[int], valid_task_names: Iterable[str]) -> Tuple[str, str]:
    system = (
        "You are a clinic program designer for a schema-driven orchestration app.\n"
        f"{JSON_ONLY_RULE}\n\n"
        "Goal: convert a natural-language clinic workflow description into a practical plan.\n"
        "Prioritize real workflow fit, not generic AI task lists.\n"
        "Do not force disease risk scoring if the clinic asks for another workflow.\n"
        "Only enable tasks that support the stated clinic goals.\n\n"
        + _schema_keys_block(
            "ClinicPlanSchema",
            ["plan_version", "clinic_description", "target_condition", "horizon_months", "constraints", "tasks", "workflow"],
            extra=(
                "constraints.candidate_pool.strategy must be one of: all, recent_notes_only, keyword_prefilter\n"
                "constraints.selection.method must be one of: top_k, threshold, threshold_then_top_k, first_k, random_k\n"
                "If no scoring task is requested, use selection.method=first_k and selection.source_task=null.\n"
                "If risk_assessment is enabled, target_condition and horizon_months are usually helpful.\n"
                "If not a disease-risk workflow, target_condition and horizon_months can be null."
            ),
        )
        + "\n\nValid task names are:\n"
        + ", ".join(valid_task_names)
        + "\n\nUse conservative defaults for review volume and deduplication when the clinic is vague."
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
        f"{JSON_ONLY_RULE} "
        + _schema_keys_block(
            "ClinicWorkflowSchema",
            ["clinic_type", "clinic_name", "note_capture", "review_cadence", "referral_pathway", "staffing_constraints", "goals", "guardrails"],
            extra=(
                "Infer a plausible workflow from the description, but keep assumptions explicit in goals or staffing_constraints. "
                "Prefer concise, operational phrasing."
            ),
        )
    )
    user = f"Clinic description:\n{clinic_description}"
    return system, user


def risk_assessment_prompt(*, patient_id: str, target: str, horizon: int, notes: str) -> Tuple[str, str]:
    system = (
        "You are a clinical risk-triage assistant for longitudinal primary-care notes. "
        f"{JSON_ONLY_RULE} "
        + _schema_keys_block(
            "RiskAssessmentSchema",
            ["patient_id", "target_condition", "horizon_months", "risk_probability", "risk_level", "key_risk_factors", "key_protective_factors", "recommended_next_steps", "safety_notes"],
            extra=(
                "risk_probability must be a number between 0 and 1. risk_level must be low/moderate/high. "
                "This is NOT a diagnosis. Do not claim certainty.\n"
                "Use the full note history, including weak longitudinal signals (repeated symptoms, weight change, metabolic changes, unresolved workup, family history, care gaps).\n"
                "Calibrate conservatively when evidence is sparse.\n"
                + _evidence_grounding_block()
            ),
        )
        + "\n\n"
        + _output_discipline_block()
        + "\n"
        + _missing_data_rules_block()
        + "\n"
        + _few_shot_block("risk_assessment")
    )
    user = _join_context([
        f"Patient ID: {patient_id}",
        f"Target condition: {target}",
        f"Horizon months: {horizon}",
        _notes_block(notes),
    ])
    return system, user


def queue_prioritization_prompt(*, clinic_goals: str, workflow_json: str, patient_id: str, notes: str, target: str, horizon: int, risk_json: str = "none", summary_json: str = "none", followup_gap_json: str = "none") -> Tuple[str, str]:
    system = (
        "You assign chart review priority inside a clinic workflow. "
        f"{JSON_ONLY_RULE} "
        + _schema_keys_block(
            "QueuePrioritizationSchema",
            ["patient_id", "priority_score", "priority_level", "queue_reason", "recommended_window"],
            extra=(
                "priority_score must be a number from 0 to 1. priority_level must be low/moderate/high.\n"
                "Use note acuity, unresolved follow-up, workflow constraints, and any supplied risk/gap context.\n"
                "queue_reason should be specific and practical. recommended_window should be actionable (for example: 'within 48 hours' or 'next routine review').\n"
                + _evidence_grounding_block()
            ),
        )
        + "\n\n"
        + _output_discipline_block()
        + "\n"
        + _missing_data_rules_block()
        + "\n"
        + _few_shot_block("queue_prioritization")
    )
    user = _join_context([
        "Clinic goals:", clinic_goals, "",
        "Workflow context JSON:", workflow_json, "",
        f"Patient ID: {patient_id}",
        f"Optional target: {target}; horizon={horizon} months",
        "Risk context JSON (optional):", risk_json,
        "Clinician summary JSON (optional):", summary_json,
        "Follow-up gap JSON (optional):", followup_gap_json,
        "",
        _notes_block(notes),
    ])
    return system, user


def clinician_summary_prompt(*, patient_id: str, target: str, horizon: int, risk_json: str, notes: str) -> Tuple[str, str]:
    system = (
        "You produce a high-value chart review summary for a busy clinician. "
        f"{JSON_ONLY_RULE} "
        + _schema_keys_block(
            "ClinicianSummarySchema",
            ["patient_id", "target_condition", "horizon_months", "summary_for_chart", "suggested_orders", "suggested_referrals", "safety_netting"],
            extra=(
                "summary_for_chart should be a complete review-ready narrative (usually 4 to 7 sentences), not a one-liner.\n"
                "Include the main concern, important longitudinal signals, uncertainty/limitations, and why follow-up matters now.\n"
                "suggested_orders and suggested_referrals must be actionable and realistic. safety_netting must include explicit red-flag or follow-up precautions.\n"
                + _evidence_grounding_block()
            ),
        )
        + "\n\n"
        + _output_discipline_block()
        + "\n"
        + _missing_data_rules_block()
        + "\n"
        + _few_shot_block("clinician_summary")
    )
    user = _join_context([
        f"Patient ID: {patient_id}",
        f"Target condition: {target}",
        f"Horizon months: {horizon}",
        "Risk context JSON (optional):",
        risk_json,
        "",
        _notes_block(notes),
    ])
    return system, user


def admin_referral_prompt(*, patient_id: str, target: str, destination: str, risk_json: str, summary_json: str, guideline_json: str = "none", notes: str = "") -> Tuple[str, str]:
    system = (
        "You create a referral-support package for clinic staff. "
        f"{JSON_ONLY_RULE} "
        + _schema_keys_block(
            "AdminReferralSchema",
            ["patient_id", "target_condition", "urgency", "destination_service", "reason_for_referral", "attach_documents", "admin_notes"],
            extra=(
                "urgency must be one of: routine, semi-urgent, urgent.\n"
                "reason_for_referral should be a detailed, chart-grounded explanation (2 to 5 sentences) with the clinical concern and urgency rationale.\n"
                "attach_documents should list practical documents to include. admin_notes should mention missing information or logistics checks when relevant.\n"
                + _evidence_grounding_block()
            ),
        )
        + "\n\n"
        + _output_discipline_block()
        + "\n"
        + _missing_data_rules_block()
        + "\n"
        + _few_shot_block("admin_referral")
    )
    user = _join_context([
        f"Patient ID: {patient_id}",
        f"Target condition/context: {target}",
        f"Destination service: {destination}",
        "Risk context JSON:", risk_json,
        "Clinician summary JSON:", summary_json,
        "Guideline comparison JSON (optional):", guideline_json,
        _notes_block(notes),
    ])
    return system, user


def patient_instructions_prompt(*, patient_id: str, target: str, horizon: int, summary_json: str, risk_json: str, followup_gap_json: str = "none", admin_referral_json: str = "none", notes: str = "") -> Tuple[str, str]:
    system = (
        "You write patient-facing instructions based on the chart, for after-visit or outreach use. "
        f"{JSON_ONLY_RULE} "
        + _schema_keys_block(
            "PatientInstructionsSchema",
            ["patient_id", "target_condition", "horizon_months", "instructions"],
            extra=(
                "instructions must be a list of short patient-friendly action statements (usually 4 to 8 items).\n"
                "Use plain language, non-alarming wording, and no unsupported diagnosis claims.\n"
                "Include follow-up timing and return precautions when supported by the note context."
            ),
        )
        + "\n\n"
        + _output_discipline_block()
        + "\n"
        + _missing_data_rules_block()
    )
    user = _join_context([
        f"Patient ID: {patient_id}",
        f"Context target: {target}",
        f"Horizon months: {horizon}",
        "Clinician summary JSON (optional):", summary_json,
        "Risk context JSON (optional):", risk_json,
        "Follow-up gap JSON (optional):", followup_gap_json,
        "Admin referral JSON (optional):", admin_referral_json,
        _notes_block(notes),
    ])
    return system, user


def results_summary_prompt(*, patient_id: str, target: str, horizon: int, notes: str) -> Tuple[str, str]:
    system = (
        "You summarize tests and trends from free-text notes only. "
        f"{JSON_ONLY_RULE} "
        + _schema_keys_block(
            "ResultsSummarySchema",
            ["patient_id", "target_condition", "horizon_months", "labs_summary", "imaging_summary", "trending_summary"],
            extra=(
                "Summarize labs, imaging, and trends only from what is present in the notes. "
                "If exact values are absent, explicitly state that and summarize qualitative trends or uncertainty."
            ),
        )
    )
    user = _join_context([f"Patient ID: {patient_id}", f"Target condition: {target}", f"Horizon months: {horizon}", _notes_block(notes)])
    return system, user


def referral_letter_prompt(*, patient_id: str, target: str, urgency: str, recipient: str, referral_json: str, risk_json: str, summary_json: str, notes: str) -> Tuple[str, str]:
    system = (
        "You draft a specialist referral letter body for a clinician. "
        f"{JSON_ONLY_RULE} "
        + _schema_keys_block(
            "ReferralLetterSchema",
            ["patient_id", "target_condition", "urgency", "recipient", "letter_body", "attachments"],
            extra=(
                "urgency must be one of: routine, semi-urgent, urgent.\n"
                "letter_body should be a full note (typically 1 to 3 short paragraphs), not a sentence fragment.\n"
                "Include reason for referral, key longitudinal findings, relevant context, and what is being requested from the consultant.\n"
                "Stay factual and note-based. Avoid unsupported claims or fabricated demographics/results."
            ),
        )
    )
    user = _join_context([
        f"Patient ID: {patient_id}",
        f"Target condition: {target}",
        f"Urgency: {urgency}",
        f"Recipient: {recipient}",
        "Referral context JSON:", referral_json,
        "Risk context JSON:", risk_json,
        "Clinician summary JSON:", summary_json,
        _notes_block(notes),
    ])
    return system, user


def differential_prompt(*, patient_id: str, target: str, horizon: int, summary_json: str, notes: str) -> Tuple[str, str]:
    system = (
        "You produce a differential brainstorming aid for clinician review only. "
        f"{JSON_ONLY_RULE} "
        + _schema_keys_block(
            "DifferentialDiagnosisSchema",
            ["patient_id", "target_condition", "horizon_months", "possible_diagnoses", "reasoning"],
            extra=(
                "No certainty claims. possible_diagnoses should be ordered by plausibility and include non-malignant alternatives when appropriate. "
                + _evidence_grounding_block()
            ),
        )
    )
    user = _join_context([f"Patient ID: {patient_id}", f"Target condition: {target}", f"Horizon months: {horizon}", "Clinician summary JSON (optional):", summary_json, _notes_block(notes)])
    return system, user


def guideline_comparison_prompt(*, patient_id: str, target: str, horizon: int, notes: str, risk_json: str = "none", summary_json: str = "none") -> Tuple[str, str]:
    system = (
        "You map the note scenario to likely guideline families/pathways. "
        f"{JSON_ONLY_RULE} "
        + _schema_keys_block(
            "GuidelineComparisonSchema",
            ["patient_id", "target_condition", "horizon_months", "recommended_guidelines", "evidence_summary"],
            extra=(
                "Do not cite exact guideline text if not provided. Keep it high-level and clinically practical. "
                + _evidence_grounding_block()
            ),
        )
        + "\n\n"
        + _output_discipline_block()
        + "\n"
        + _missing_data_rules_block()
        + "\n"
        + _few_shot_block("guideline_comparison")
    )
    user = _join_context([
        f"Patient ID: {patient_id}",
        f"Target condition: {target}",
        f"Horizon months: {horizon}",
        "Risk context JSON (optional):", risk_json,
        "Clinician summary JSON (optional):", summary_json,
        _notes_block(notes),
    ])
    return system, user


def followup_gap_prompt(*, clinic_goals: str, patient_id: str, notes: str, summary_json: str = "none", risk_json: str = "none") -> Tuple[str, str]:
    system = (
        "You identify possible follow-up gaps from longitudinal notes. "
        f"{JSON_ONLY_RULE} "
        + _schema_keys_block(
            "FollowupGapSchema",
            ["patient_id", "pending_items", "missed_followup_signals", "suggested_actions", "gap_severity"],
            extra=(
                "gap_severity must be one of: low, moderate, high.\n"
                "Distinguish observed signals from assumptions. suggested_actions should be practical next chart actions.\n"
                + _evidence_grounding_block()
            ),
        )
        + "\n\n"
        + _output_discipline_block()
        + "\n"
        + _missing_data_rules_block()
        + "\n"
        + _few_shot_block("followup_gap_detection")
    )
    user = _join_context([
        "Clinic goals:", clinic_goals, "",
        f"Patient ID: {patient_id}",
        "Clinician summary JSON (optional):", summary_json,
        "Risk context JSON (optional):", risk_json,
        _notes_block(notes),
    ])
    return system, user


def referral_intake_checklist_prompt(*, destination: str, patient_id: str, notes: str, admin_referral_json: str = "none", workflow_json: str = "none") -> Tuple[str, str]:
    system = (
        "You build a referral intake readiness checklist for the destination service. "
        f"{JSON_ONLY_RULE} "
        + _schema_keys_block(
            "ReferralIntakeChecklistSchema",
            ["patient_id", "destination_service", "triage_bucket", "available_info", "missing_info", "checklist_items"],
            extra=(
                "triage_bucket must be one of: routine, semi-urgent, urgent.\n"
                "Separate what is available in the notes from what is still missing.\n"
                "Use the supplied referral/workflow context if provided."
            ),
        )
    )
    user = _join_context([
        f"Destination service: {destination}",
        f"Patient ID: {patient_id}",
        "Admin referral JSON (optional):", admin_referral_json,
        "Workflow JSON (optional):", workflow_json,
        _notes_block(notes),
    ])
    return system, user


def lab_trend_prompt(*, timeframe: str, patient_id: str, notes: str) -> Tuple[str, str]:
    system = (
        "You summarize lab/result trends from notes for both clinicians and patients. "
        f"{JSON_ONLY_RULE} "
        + _schema_keys_block(
            "LabTrendSummarySchema",
            ["patient_id", "timeframe_label", "clinician_summary", "patient_friendly_summary", "concerning_trends", "suggested_next_steps"],
            extra=(
                "If exact numeric labs are absent, summarize qualitative trends and limitations transparently. "
                "Keep the patient_friendly_summary plain-language."
            ),
        )
    )
    user = _join_context([f"Timeframe: {timeframe}", f"Patient ID: {patient_id}", _notes_block(notes)])
    return system, user


def care_plan_reconciliation_prompt(*, clinic_goals: str, patient_id: str, notes: str) -> Tuple[str, str]:
    system = (
        "You reconcile prior plans, completed items, unresolved items, and changed plans from free-text notes. "
        f"{JSON_ONLY_RULE} "
        + _schema_keys_block(
            "CarePlanReconciliationSchema",
            ["patient_id", "prior_plan_items", "completed_items", "unresolved_items", "changed_items", "suggested_next_steps"],
            extra=("Prefer concrete chart actions over vague statements. " + _evidence_grounding_block()),
        )
    )
    user = _join_context(["Clinic goals:", clinic_goals, "", f"Patient ID: {patient_id}", _notes_block(notes)])
    return system, user
