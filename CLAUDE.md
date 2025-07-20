# AIDE ML - Project Structure Guide

This document provides a guide to the project structure of AIDE ML, an LLM-driven agent that writes, evaluates, and improves machine learning code.

## Project Structure Overview

```
aideml/
├── aide/                   # All source code
│   ├── backend/            # LLM backend integrations
│   ├── example_tasks/      # Example tasks for the agent to solve
│   ├── utils/              # Configuration, data preview, metrics, and other utilities
│   ├── webui/              # Streamlit-based web UI
│   ├── agent.py            # Core agent logic for tree search and code generation
│   ├── interpreter.py      # Executes and evaluates code solutions
│   ├── journal.py          # Logging and recording experiment progress
│   ├── journal2report.py   # Convert journal logs to reports
│   ├── run.py              # Main module for running experiments
│   └── __init__.py         # Main package initialization
├── docs/                   # Documentation and planning
│   ├── memos/              # Status reports and memos
│   ├── papers/             # Research papers and references
│   ├── explain.md          # Project explanation
│   └── plan.md             # Claude Code integration plan
├── sample_results/         # Example outputs from the agent
├── tests/                  # Unit tests directory
├── environment.yml         # Conda environment specification
├── requirements.txt        # Python dependencies
├── run_aide.py             # Primary CLI entry point
├── run_webui.py            # Web UI entry point
├── setup_dev.sh            # Development setup script
├── test_claude_code_backend.py     # Claude Code backend tests
└── test_integration_aide_ml.py     # Integration tests
```

## Core Modules Quick Reference

**Core Components:**
- `aide/agent.py` - Implements the agentic tree search algorithm.
- `aide/interpreter.py` - Handles the execution of generated Python scripts and evaluates their performance.
- `aide/run.py` - The main script to run AIDE ML from the command line.
- `aide/journal.py` - Manages logging and reporting of the agent's activities.

**LLM Backends (`aide/backend/`):**
- `aide/backend/backend_openai.py` - Integration with OpenAI models.
- `aide/backend/backend_anthropic.py` - Integration with Anthropic models.
- `aide/backend/backend_gemini.py` - Integration with Gemini models.
- `aide/backend/backend_claude_code.py` - Integration with Claude Code SDK (partially implemented).
- `aide/backend/backend_openrouter.py` - Integration with OpenRouter API.
- `aide/backend/utils.py` - Shared utilities for backend implementations.

**Web UI:**
- `aide/webui/app.py` - A Streamlit application for interacting with the AIDE ML agent.

**Utilities (`aide/utils/`):**
- `aide/utils/config.py` - Manages configuration settings for the agent and experiments.
- `aide/utils/config.yaml` - Default configuration file for AIDE ML.
- `aide/utils/metric.py` - Defines and calculates evaluation metrics.
- `aide/utils/tree_export.py` - Exports the solution search tree for visualization.
- `aide/utils/data_preview.py` - Utilities for previewing and analyzing data.
- `aide/utils/response.py` - Response handling utilities.
- `aide/utils/serialize.py` - Serialization utilities for data structures.
- `aide/utils/viz_templates/` - HTML/JS templates for visualization.

## Environment Setup

Before running or developing AIDE ML, ensure you have the `aideml` conda environment activated. This will provide all the necessary dependencies.

```bash
conda activate aideml
```

## Development Guidelines

**Testing:**
- All new features and bug fixes **must** include corresponding unit tests.
- Create a `tests/` directory in the root of the project for all test files.
- Test files should follow the naming convention `test_*.py`.
- Run tests using `pytest`:
  ```bash
  pytest tests/
  ```

## Key Entry Points & Main Files

**Running an Experiment:**
- `python run_aide.py` - The primary entry point for running the AIDE ML agent from the command line.
- `python -m aide.run` - Alternative method to run experiments using the module directly.

**Web UI:**
- `python run_webui.py` - Launches the Streamlit-based web interface.
- `python -m aide.webui.app` - Alternative method to launch the web UI.

**Example Tasks:**
- `aide/example_tasks/bitcoin_price.md` - Bitcoin price prediction task.
- `aide/example_tasks/house_prices.md` - House price regression task.

## Technical Stack

**Core:** Python 3.10+
**Machine Learning:** The agent can generate code using various ML libraries, but the core project has minimal direct ML dependencies.
**LLM Integrations:** OpenAI, Anthropic, Gemini, OpenRouter, Claude Code (partial)
**Web Framework:** Streamlit (for the UI)

## Claude Code Integration Status

The project includes a partial implementation of Claude Code SDK integration:

**Implemented:**
- Backend module (`aide/backend/backend_claude_code.py`) with full query implementation
- Backend registration in `aide/backend/__init__.py`
- Unit tests (`test_claude_code_backend.py`)
- Integration tests stub (`test_integration_aide_ml.py`)

**Not Yet Implemented:**
- `claude-code-sdk` dependency not in `requirements.txt`
- Backend configuration support in `aide/utils/config.py`
- CLI arguments (`--backend`, `--backend-opt`) in `run_aide.py`
- Backend parameter passing in `aide/agent.py`

See `docs/plan.md` for the full integration plan and `docs/memos/status_20250720-061453.md` for the latest implementation status.
