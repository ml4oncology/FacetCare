# FacetCare: Flexible Agentic Clinical Engine for Triage

FacetCare is a clinician-centered, agentic triage platform that adapts to the clinic, instead of forcing the clinic to adapt to the software. The physician starts with plain language, for example, "Review the top 5 patients at risk of colorectal cancer every week and prepare summaries and referral drafts." FacetCare converts that request into a validated clinic plan, runs the plan over longitudinal patient notes, and returns a concise review bundle for physician approval.

<p align="center">
  <img src="assets/logo.png" alt="FacetCare logo" width="200"/>
</p>

## What is this
- A Plan Builder converts a clinic's natural-language workflow description into a validated Clinic Plan (JSON).
- A deterministic Runner executes that plan repeatedly:
  - scores patients (LLM task)
  - selects top-K or thresholded top-K
  - enforces a dedup window
  - generates clinic-ready artifacts for selected patients only
  - packages everything into a Review Bundle for human review

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

All arguments can also be set via environment variables: `FACETCARE_ENDPOINT`, `FACETCARE_MODEL`, `FACETCARE_HOST`, `FACETCARE_PORT`.

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
python web_app.py --endpoint http://127.0.0.1:8080/v1/ --model llama-3-8b
```

Then open `http://127.0.0.1:5000` in a browser.
