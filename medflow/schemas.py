from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class NoteCaptureSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str
    format: str
    cadence: str


class ReviewCadenceSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    who_reviews: str
    frequency: str
    trigger: str


class ReferralPathwaySchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    internal: str
    external: str
    urgent_flags: List[str]


class GuardrailsSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    privacy: str
    safety: str
    logging: str


class ClinicWorkflowSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clinic_type: str
    clinic_name: str
    note_capture: NoteCaptureSchema
    review_cadence: ReviewCadenceSchema
    referral_pathway: ReferralPathwaySchema
    staffing_constraints: List[str]
    goals: List[str]
    guardrails: GuardrailsSchema


class RiskAssessmentSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_id: str
    target_condition: str
    horizon_months: int
    risk_probability: float = Field(ge=0.0, le=1.0)
    risk_level: str
    key_risk_factors: List[str]
    key_protective_factors: List[str]
    recommended_next_steps: List[str]
    safety_notes: List[str]


class ClinicianSummarySchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_id: str
    target_condition: str
    horizon_months: int
    summary_for_chart: str
    suggested_orders: List[str]
    suggested_referrals: List[str]
    safety_netting: List[str]


class AdminReferralSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_id: str
    target_condition: str
    urgency: Literal["routine", "semi-urgent", "urgent"]
    destination_service: str
    reason_for_referral: str
    attach_documents: List[str]
    admin_notes: List[str]


class RiskLevelPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    low_lt: float = Field(default=0.01, ge=0.0, le=1.0)
    moderate_lt: float = Field(default=0.05, ge=0.0, le=1.0)

    def to_level(self, p: float) -> str:
        if p < self.low_lt:
            return "low"
        if p < self.moderate_lt:
            return "moderate"
        return "high"


class CandidatePoolConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy: Literal["all", "recent_notes_only", "keyword_prefilter"] = "all"
    max_candidates: Optional[int] = Field(default=None, ge=1)
    keywords: List[str] = Field(default_factory=list)


class SelectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # source_task is optional. If omitted, runner will auto-pick a reasonable source.
    source_task: Optional[str] = None
    # method controls selection behavior. "first_k" supports non-scored workflows.
    method: Literal["top_k", "threshold", "threshold_then_top_k", "first_k"] = "top_k"
    k: int = Field(default=5, ge=1)
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ProgramConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cadence: Literal["daily", "weekly", "per_visit", "monthly", "yearly"] = "weekly"
    review_limit: int = Field(default=5, ge=1)
    dedup_days: int = Field(default=180, ge=0)
    candidate_pool: CandidatePoolConfig = Field(default_factory=CandidatePoolConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    risk_level_policy: RiskLevelPolicy = Field(default_factory=RiskLevelPolicy)


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    enabled: bool = True
    params: Dict[str, Any] = Field(default_factory=dict)
    depends_on: List[str] = Field(default_factory=list)


class ClinicPlanSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plan_version: str = "2.0"
    clinic_description: str
    # Optional so the plan can be clinic-agnostic (not always disease-risk workflows).
    target_condition: Optional[str] = None
    horizon_months: Optional[int] = Field(default=None, ge=1, le=120)
    constraints: ProgramConstraints = Field(default_factory=ProgramConstraints)
    tasks: List[TaskSpec] = Field(default_factory=list)
    workflow: Optional[ClinicWorkflowSchema] = None


class PatientRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_id: str
    longitudinal_notes: str


class SelectedPatientBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_id: str
    selection_reason: str
    risk: Optional[RiskAssessmentSchema] = None
    clinician_summary: Optional[ClinicianSummarySchema] = None
    admin_referral: Optional[AdminReferralSchema] = None
    extra_outputs: Dict[str, Any] = Field(default_factory=dict)


class PatientInstructionsSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_id: str
    target_condition: str
    horizon_months: int
    instructions: List[str]


class ResultsSummarySchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_id: str
    target_condition: str
    horizon_months: int
    labs_summary: str
    imaging_summary: str
    trending_summary: str


class ReferralLetterSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_id: str
    target_condition: str
    urgency: Literal["routine", "semi-urgent", "urgent"]
    recipient: str
    letter_body: str
    attachments: List[str]


class DifferentialDiagnosisSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_id: str
    target_condition: str
    horizon_months: int
    possible_diagnoses: List[str]
    reasoning: str


class GuidelineComparisonSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_id: str
    target_condition: str
    horizon_months: int
    recommended_guidelines: List[str]
    evidence_summary: str


# New clinic-agnostic task outputs
class FollowupGapSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_id: str
    pending_items: List[str]
    missed_followup_signals: List[str]
    suggested_actions: List[str]
    gap_severity: Literal["low", "moderate", "high"]


class ReferralIntakeChecklistSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_id: str
    destination_service: str
    triage_bucket: Literal["routine", "semi-urgent", "urgent"]
    available_info: List[str]
    missing_info: List[str]
    checklist_items: List[str]


class LabTrendSummarySchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_id: str
    timeframe_label: str
    clinician_summary: str
    patient_friendly_summary: str
    concerning_trends: List[str]
    suggested_next_steps: List[str]


class QueuePrioritizationSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_id: str
    priority_score: float = Field(ge=0.0, le=1.0)
    priority_level: Literal["low", "moderate", "high"]
    queue_reason: str
    recommended_window: str


class CarePlanReconciliationSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_id: str
    prior_plan_items: List[str]
    completed_items: List[str]
    unresolved_items: List[str]
    changed_items: List[str]
    suggested_next_steps: List[str]


class ReviewBundleSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_date: str
    clinic_name: str
    target_condition: Optional[str] = None
    horizon_months: Optional[int] = None
    selected: List[SelectedPatientBundle]
    not_selected_count: int
