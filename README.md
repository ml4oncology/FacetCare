# FacetCare: Flexible Agentic Clinical Engine for Triage

FacetCare is a clinician-centered, agentic triage platform that adapts to the clinic, instead of forcing the clinic to adapt to the software. The core idea is simple: most clinical AI tools fail not because the model is weak, but because the workflow is a poor fit. In busy clinics, especially primary care, clinicians do not have time to manually review long, messy longitudinal notes or to use rigid tools that add clicks and interrupt their workflow. FacetCare is designed to solve that adoption problem. 

The physician starts with plain language, for example, "Review the top 5 patients at risk of colorectal cancer every week and prepare summaries and referral drafts." FacetCare converts that request into a validated clinic plan, runs the plan over longitudinal patient notes, and returns a concise review bundle for physician approval.

<p align="center">
  <img src="assets/logo.png" alt="FacetCare logo" width="1000"/>
</p>

<div style="text-align: center;">
  <img src="https://img.youtube.com/vi/lUyKTqrua48/hqdefault.jpg" 
       alt="FacetCare demo" 
       style="max-width: 60%; height: auto;">
  <br>
  <a href="https://www.youtube.com/watch?v=lUyKTqrua48">▶️ Watch demo</a>
</div>


## What this project is about
FacetCare is a **clinician-in-the-loop orchestration system** built around a medical LLM (MedGemma) and a deterministic workflow engine.  It is a prototype of how medical AI can be integrated into actual clinic workflows while preserving usability, transparency, and clinician control.

### The problem FacetCare addresses

Clinical decision support systems often have low real-world adoption because they are deployed in ways that do not match how clinicians work. Even strong models are ignored if they increase friction.

FacetCare targets that gap by focusing on:

* **workflow fit** (the tool adapts to the clinic)
* **usable outputs** (summaries, gaps, referrals, patient instructions)
* **human review** (the clinician remains the decision-maker)
* **structured orchestration** (repeatable and auditable workflows)

The project is especially relevant for settings where clinicians are time-constrained and much of the useful information is buried in unstructured chart notes.

## How FacetCare works (high level)

FacetCare has three main layers:

### 1) Plan Builder

The clinician describes the goal in plain language, such as:

> “Review the top 5 patients at high pancreatic cancer risk every week, avoid repeat reviews for 30 days, and generate summaries and referral drafts.”

FacetCare translates this into a **structured clinic plan** using MedGemma with explicit parameters, such as:

* review cadence
* patient count
* deduplication window
* target condition and risk horizon (when relevant)
* enabled tasks
* task dependencies

This makes the workflow reusable and inspectable.

### 2) Agentic Runner

The Agentic Runner executes the clinic plan over patient records.

It handles:

* candidate selection
* task ordering and dependencies
* deduplication logic (to avoid repeatedly surfacing the same patient too often)
* caching of task outputs
* generation of a per-patient review bundle

### 3) Task Registry (MedGemma-backed tasks)

FacetCare uses MedGemma for **bounded clinical tasks** with explicit input and output expectations.

Examples of tasks include:

* **Risk assessment** (risk-oriented triage from longitudinal notes)
* **Clinician summary** (concise chart synthesis)
* **Follow-up gap detection** (possible missing follow-up or unresolved issues)
* **Guideline comparison** (optional workflow support)
* **Admin referral support** (structured referral-ready content)
* **Patient instructions** (patient-facing handout language)

## What the clinician sees

FacetCare includes a lightweight web application that supports the end-to-end workflow:

### Setup and Planning

* The clinician enters a plain-language clinic goal
* Optional advanced parameters can be expanded when needed
* FacetCare generates a structured plan that can be modified manually

### Run and Results

* The plan is executed on a patient set
* The clinician receives a prioritized patient list
* The results page shows key status fields and generated artifacts

### Patient Review

Each patient has a dynamic review page that changes based on the plan. Depending on the workflow, the page can show:

* risk output
* clinician summary
* follow-up gaps
* referral content
* patient instructions
* source note preview
* raw structured output (for auditability)

The clinician can also refresh individual tasks and mark patients as reviewed.

## Why MedGemma is used in this project

FacetCare uses **MedGemma** as the medical reasoning engine because this project requires strong performance on clinical language understanding, especially over noisy and longitudinal chart notes.

MedGemma is used here in a controlled way:

* for task-level medical reasoning
* inside a orchestration pipeline
* with structured output expectations
* under clinician review

## Design principles

FacetCare is built around a few core principles:

### Workflow-native

The system should adapt to the clinic’s workflow, not force the clinic to change.

### Physician-in-the-loop

FacetCare prepares drafts and triage outputs, but clinicians review and decide what action to take.

### Structured and auditable

The orchestration and task outputs are designed to be inspectable and testable.

### Practical outputs

The system prioritizes outputs clinicians can actually use, such as summaries, referrals, and patient handouts.

## Current prototype scope

This repository is a prototype focused on demonstrating:

* plain-language workflow specification
* plan generation with MedGemma and execution
* MedGemma-backed clinical task orchestration
* a clinician-facing review experience
* exportable/printable workflow artifacts

It is **not** a production clinical system and does not autonomously place orders, send referrals, or modify the medical record. All outputs are intended for clinician review.

## Example use cases

FacetCare is designed to support multiple workflows using the same engine, including:

* **Cancer risk triage** from family doctor progress notes
* **Follow-up gap review** for high-risk or under-screened patients
* **Referral preparation** workflows in primary care
* Other custom triage workflows defined by the clinic in plain language

The same orchestration approach can be adapted to different conditions and clinic needs without rebuilding the product for each use case.

## Install

Create and activate a dedicated environment, then install the two required packages:

```bash
# Create the environment
conda create -n FacetCare python=3.11 -y
conda activate FacetCare

# Install dependencies
pip install pydantic openai
```

If you prefer plain `venv`:

```bash
python -m venv FacetCare
source FacetCare/bin/activate        # Windows: FacetCare\Scripts\activate
pip install pydantic openai
```

## Run

FacetCare requires an **OpenAI-compatible LLM inference server** (e.g. llama.cpp, vLLM, or Ollama with OpenAI compatibility) to be running and reachable before starting the app.

Run the web app with:

```bash
python web_app.py --endpoint <url> --model <model_name>
```

| Argument | Default | Description |
|---|---|---|
| `--endpoint` | `http://192.168.0.2:881/v1/` | Base URL of your OpenAI-compatible inference server |
| `--model` | `medgemma` | Model name as served by the inference server |
| `--host` | `127.0.0.1` | Host to bind the Flask app to |
| `--port` | `5000` | Port to bind the Flask app to |


**Example — local network server, accessible to a colleague:**

```bash
python web_app.py \
  --endpoint http://192.168.0.2:881/v1/ \
  --model medgemma \
  --host 0.0.0.0 \
  --port 5000
```

Then open `http://<your-machine-ip>:5000` in a browser.

**Example — localhost only (default):**

```bash
python web_app.py --endpoint http://127.0.0.1:8080/v1/ --model medgemma-1.5-4b-it
```

Then open `http://127.0.0.1:5000` in a browser.
