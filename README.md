# FacetCare

Plan-driven clinic workflow orchestration with LLM tasks (llama.cpp/vLLM compatible).

## What is this
- A Plan Builder converts a clinic's natural-language workflow description into a validated Clinic Plan (JSON).
- A deterministic Runner executes that plan repeatedly:
  - scores patients (LLM task)
  - selects top-K or thresholded top-K
  - enforces a dedup window
  - generates clinic-ready artifacts for selected patients only
  - packages everything into a Review Bundle for human review

## Install
pip install pydantic openai

## Run
python demo.py or python web_app.py

Edit demo.py or web_app.py to match your base_url and served model name.
