from __future__ import annotations

from typing import List

from .schemas import RiskAssessmentSchema, SelectionConfig


def select_top_k(risks: List[RiskAssessmentSchema], *, selection: SelectionConfig) -> List[RiskAssessmentSchema]:
    ranked = sorted(risks, key=lambda r: float(r.risk_probability), reverse=True)

    if selection.method == "threshold_then_top_k" and selection.threshold is not None:
        ranked = [r for r in ranked if float(r.risk_probability) >= float(selection.threshold)]

    return ranked[: selection.k]


def make_selection_reason(risk: RiskAssessmentSchema, rank: int) -> str:
    return f"Selected rank {rank} by risk_probability={risk.risk_probability:.4f}."
