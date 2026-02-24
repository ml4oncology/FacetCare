from __future__ import annotations

import argparse
import os

from openai import OpenAI

from medflow.llm_client import LLMJsonClient
from medflow.plan_builder import build_clinic_plan_from_description
from medflow.schemas import PatientRecord
from medflow.tasks import TaskContext, default_task_registry
from medflow.runner import PlanRunner
from medflow.dedup import JSONFileDedupStore


def main() -> None:
    parser = argparse.ArgumentParser(description="MedFlow CLI runner")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("MEDFLOW_ENDPOINT", "http://192.168.0.1:888/v1/"),
        help="LLM API endpoint URL",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MEDFLOW_MODEL", "medgema"),
        help="Model name served at the endpoint",
    )
    args = parser.parse_args()

    client = OpenAI(base_url=args.endpoint, api_key="")
    model = args.model

    llm = LLMJsonClient(client=client, model=model)
    ctx = TaskContext(llm=llm)

    clinic_description = (
        "Family Practice Clinic in Ontario.  Mostly EMR free-text visit notes.  "
        "Clinicians can review a shortlist weekly, top 5.  Do not repeat patients within 6 months.  "
        "Generate clinician chart summary and an admin referral payload if moderate or high risk.  "
        "Also generate patient instructions.  Guardrails: no diagnosis, no PHI stored in logs."
    )

    plan = build_clinic_plan_from_description(
        llm=llm,
        clinic_description=clinic_description,
        default_target_condition="pancreatic_cancer",
        default_horizon_months=36,
    )

    patients = [
        PatientRecord(
            patient_id="P009",
            longitudinal_notes="""
PATIENT P009
DOB: 1968-09-14
<note_1102_2023-10-02> DM2, on metformin.  A1c 7.1.  Some dyspepsia off/on.
<note_1388_2024-03-10> Epigastric discomfort.  No jaundice.  Appetite ok.
<note_1710_2024-08-23> Missed follow-up.  Weight stable.  No alarm sx.
<note_1905_2025-01-18> Weight down 4 kg.  A1c 7.8.  Missed FU.  Some early satiety.
""".strip(),
        ),
        PatientRecord(
            patient_id="P010",
            longitudinal_notes="""
PATIENT P010
DOB: 1975-01-20
<note_2001_2024-11-01> Routine FU.  HTN controlled.  No GI complaints.
<note_2088_2025-01-10> Viral URI.  Appetite ok.  Weight stable.
""".strip(),
        ),
    ]

    dedup = JSONFileDedupStore("dedup_store.json")
    runner = PlanRunner(ctx=ctx, task_registry=default_task_registry(), dedup_store=dedup)

    bundle = runner.run(plan=plan, patients=patients, run_date="2025-02-01")
    print("\n=== Clinic Plan ===")
    print(plan.model_dump_json(indent=2))
    print("\n=== Review Bundle ===")
    print(bundle.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
