# AIDE ML - Project Structure Guide

This document provides a guide to the project structure of AIDE ML, an LLM-driven agent that writes, evaluates, and improves machine learning code.

## Project Structure Overview

```
aideml/
├── aide/                   # All source code
│   ├── backend/            # LLM backend integrations (OpenAI, Anthropic, Gemini)
│   ├── utils/              # Configuration, data preview, metrics, and other utilities
│   ├── webui/              # Streamlit-based web UI
│   ├── agent.py            # Core agent logic for tree search and code generation
│   ├── interpreter.py      # Executes and evaluates code solutions
│   ├── journal.py          # Logging and recording experiment progress
│   ├── run.py              # Main entry point for running experiments
│   └── __init__.py         # Main package initialization
├── example_tasks/          # Example tasks for the agent to solve
└── scripts/                # Utility scripts (currently empty)
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

**Web UI:**
- `aide/webui/app.py` - A Streamlit application for interacting with the AIDE ML agent.

**Utilities (`aide/utils/`):**
- `aide/utils/config.py` - Manages configuration settings for the agent and experiments.
- `aide/utils/metric.py` - Defines and calculates evaluation metrics.
- `aide/utils/tree_export.py` - Exports the solution search tree for visualization.

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

**Web UI:**
- `python -m aide.webui.app` - Launches the Streamlit-based web interface.

## Technical Stack

**Core:** Python 3.10+
**Machine Learning:** The agent can generate code using various ML libraries, but the core project has minimal direct ML dependencies.
**LLM Integrations:** OpenAI, Anthropic, Gemini
**Web Framework:** Streamlit (for the UI)
