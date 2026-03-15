# AGENTS.md

## Mission
Build and maintain a research-grade AI traffic signal optimization platform for GSDSEF (Greater San Diego Science & Engineering Fair) competition and publication-style evaluation. The platform benchmarks 10 signal controllers across 4 families using rigorous statistical methodology. No external AI API dependencies.

## Engineering Rules
- Python version: `3.11`.
- Use typed Python (`type hints`) for all public methods.
- Keep modules decoupled; each layer should expose clear interfaces.
- Avoid hard-coded file paths; use config-driven directories.
- All randomness must respect the global seed in `traffic_ai/config/settings.py`.
- Persist experiment outputs to `artifacts/` only.
- No external API dependencies (no Anthropic, OpenAI, or other AI service calls).

## Data and Experiment Rules
- Raw datasets go to `data/raw/`, processed data to `data/processed/`.
- Never overwrite raw source files after ingestion.
- Every experiment run must emit:
  - run config JSON
  - metrics CSV
  - model artifacts
  - plots (300 DPI, publication quality)
- Include baseline controllers in every comparative run.

## Quality Rules
- Unit tests live in `tests/`.
- New metrics/controllers require at least one test.
- Run quick validation before finalizing:
  - `python main.py --quick-run`
  - `pytest -q`

## Repo Layout
- Core package: `traffic_ai/`
- CLI entrypoint: `main.py`
- Dashboard: `traffic_ai/dashboard/streamlit_app.py`
- API: `traffic_ai/api/server.py`

