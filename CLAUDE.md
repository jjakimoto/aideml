# AIDE ML - Project Structure Guide

This document provides a guide to the project structure of AIDE ML, an LLM-driven agent that writes, evaluates, and improves machine learning code.

## Project Structure Overview

```
aideml/
├── .github/                # GitHub-specific configurations
│   └── workflows/          # GitHub Actions CI/CD workflows
│       ├── auto-merge.yml  # Automated PR merging workflow
│       ├── claude.yml      # Claude-specific workflow
│       ├── pr-fix.yml      # PR fixing automation
│       └── pr-review.yml   # PR review automation
├── aide/                   # All source code
│   ├── backend/            # LLM backend integrations
│   ├── example_tasks/      # Example tasks for the agent to solve
│   │   ├── bitcoin_price/  # Bitcoin price prediction task data
│   │   │   └── BTC-USD.csv # Bitcoin historical price data
│   │   ├── house_prices/   # House price prediction task data
│   │   │   ├── data_description.txt
│   │   │   ├── sample_submission.csv
│   │   │   ├── test.csv
│   │   │   └── train.csv
│   │   ├── bitcoin_price.md
│   │   └── house_prices.md
│   ├── utils/              # Configuration, data preview, metrics, and other utilities
│   ├── webui/              # Streamlit-based web UI
│   │   ├── app.py          # Main Streamlit application
│   │   └── style.css       # Custom CSS styling
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
├── sample_results/         # Example outputs from the agent (60+ ML solutions)
├── tests/                  # Unit tests directory
│   ├── __init__.py
│   ├── test_benchmark_backends.py
│   ├── test_dev_mode.py
│   ├── test_e2e_all_tasks.py
│   ├── test_hybrid_backend.py
│   ├── test_mcp_integration.py
│   ├── test_performance_monitor.py
│   └── test_specialized_prompts.py
├── LICENSE                 # Project license file
├── README.md               # Project README
├── environment.yml         # Conda environment specification
├── requirements.txt        # Python dependencies
├── run_aide.py             # Primary CLI entry point
├── run_webui.py            # Web UI entry point
├── setup_dev.sh            # Development setup script
├── test_claude_code_backend.py     # Claude Code backend tests (root level)
├── test_integration_aide_ml.py     # Integration tests (root level)
└── test_mcp_standalone.py          # Standalone MCP functionality tests (root level)
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
- `aide/backend/backend_claude_code.py` - Integration with Claude Code SDK with MCP support (fully implemented).
- `aide/backend/backend_hybrid.py` - Hybrid backend that intelligently routes queries to different providers based on task type.
- `aide/backend/backend_openrouter.py` - Integration with OpenRouter API.
- `aide/backend/utils.py` - Shared utilities for backend implementations.
- `aide/backend/mcp_server.py` - MCP (Model Context Protocol) server for AIDE ML function calls.

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
- `aide/utils/performance_monitor.py` - Performance monitoring for LLM backends with metrics tracking and analysis.
- `aide/utils/view_performance.py` - CLI tool for viewing and analyzing backend performance metrics.
- `aide/utils/benchmark_backends.py` - Systematic performance benchmarking across multiple backends.
- `aide/utils/specialized_prompts.py` - Task-specific prompt enhancements for different ML task types.
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
**LLM Integrations:** OpenAI, Anthropic, Gemini, OpenRouter, Claude Code (fully implemented)
**Web Framework:** Streamlit (for the UI)

<<<<<<< HEAD
=======
## Claude Code Integration Status

The project includes a **fully implemented** Claude Code SDK integration:

**Fully Implemented:**
- Backend module (`aide/backend/backend_claude_code.py`) with full query implementation
- Backend registration in `aide/backend/__init__.py`
- Unit tests (`test_claude_code_backend.py`)
- Integration tests (`test_integration_aide_ml.py`)
- `claude-code-sdk>=0.2.0` dependency in `requirements.txt`
- Backend configuration support in `aide/utils/config.py`
- CLI arguments (`--backend`, `--backend-opt`) in `run_aide.py`
- Backend parameter passing in `aide/agent.py`

**Recent Enhancements (Newly Implemented):**
- ✅ Hybrid Backend: Intelligent routing of queries to different providers based on task type
  - Code generation tasks automatically routed to Claude Code
  - Analysis and review tasks can use different models
  - Configurable via `--backend hybrid` with customizable routing rules
- ✅ Performance Monitoring: Comprehensive tracking and analysis of backend performance
  - Automatic metrics collection for all queries
  - Performance comparison across backends
  - CLI tool for viewing performance statistics
  - Persistent logging for historical analysis
- ✅ Specialized Prompts: ML task-specific prompt enhancements
  - Automatic detection of ML task types (classification, regression, time series, NLP, CV)
  - Task-specific best practices and pitfall warnings
  - Enhanced prompts with relevant guidance for each task type
  - Improved code review with task-aware hints

**Recent Enhancements (Newly Implemented):**
- ✅ Tool Extensions: Integration of Claude Code's MCP (Model Context Protocol) for enhanced capabilities
  - MCP configuration generation from FunctionSpec objects
  - MCP tool naming convention (mcp__aide__call_<function_name>)
  - Support for MCP-based function calling in Claude Code backend
  - MCP server implementation for handling function calls
  - Opt-in MCP support via `use_mcp=true` parameter
  - Automatic cleanup of temporary MCP configurations
- ✅ Systematic Performance Benchmarking: Comprehensive benchmarking across backends
  - Automated benchmark execution with configurable tasks
  - Performance metrics collection and comparison
  - Historical tracking and trend analysis
  - Human-readable reports and JSON output
- ✅ End-to-End Testing: Full test coverage across all example tasks
  - Automated testing of all example tasks
  - Multi-backend compatibility testing
  - Parallel execution for efficiency
  - Comprehensive test reporting with metrics

See `docs/plan.md` for the full integration plan and `docs/memos/status_20250720-115738.md` for the latest implementation status.

>>>>>>> e636a34 (docs: Update CLAUDE.md with latest status report reference)
## Using the New Features

**Hybrid Backend Usage:**
```bash
# Use hybrid backend with default settings (Claude Code for code, GPT-4o for analysis)
python run_aide.py --task aide/example_tasks/house_prices.md --backend hybrid

# Customize hybrid routing
python run_aide.py --task aide/example_tasks/bitcoin_price.md \
    --backend hybrid \
    --backend-opt agent.hybrid.code_backend=claude_code \
    --backend-opt agent.hybrid.code_model=claude-opus-4 \
    --backend-opt agent.hybrid.analysis_backend=openai \
    --backend-opt agent.hybrid.analysis_model=gpt-4o
```

**Performance Monitoring:**
```bash
# View performance summary for all backends
python -m aide.utils.view_performance summary

# Compare backend performance
python -m aide.utils.view_performance compare --backends claude_code openai anthropic

# View recent performance for a specific backend
python -m aide.utils.view_performance recent claude_code --hours 24

# Export performance data
python -m aide.utils.view_performance export performance_data.json
```

**Specialized Prompts:**
```bash
# Enable specialized prompts (enabled by default)
python run_aide.py --task aide/example_tasks/house_prices.md \
    --backend-opt use_specialized_prompts=true

# Disable specialized prompts if needed
python run_aide.py --task aide/example_tasks/bitcoin_price.md \
    --backend-opt use_specialized_prompts=false
```

**MCP (Model Context Protocol) Usage:**
```bash
# Enable MCP for function calling in Claude Code backend
python run_aide.py --task aide/example_tasks/house_prices.md \
    --backend claude_code \
    --backend-opt use_mcp=true

# Use MCP with custom configuration path
python run_aide.py --task aide/example_tasks/bitcoin_price.md \
    --backend claude_code \
    --backend-opt use_mcp=true \
    --backend-opt mcp_config_path=/path/to/mcp-config.json

# MCP with hybrid backend (applies to Claude Code backend)
python run_aide.py --task task.md \
    --backend hybrid \
    --backend-opt agent.hybrid.code_backend=claude_code \
    --backend-opt agent.claude_code.use_mcp=true

# Advanced MCP with HTTP mode
python run_aide.py --task aide/example_tasks/house_prices.md \
    --backend claude_code \
    --backend-opt use_mcp=true \
    --backend-opt use_advanced_mcp=true \
    --backend-opt mcp_http_mode=true \
    --backend-opt mcp_http_port=8080

# Advanced MCP with stdio mode (enhanced features)
python run_aide.py --task aide/example_tasks/bitcoin_price.md \
    --backend claude_code \
    --backend-opt use_mcp=true \
    --backend-opt use_advanced_mcp=true
```

**Async Support Usage:**
```python
# Using async support in custom code
import asyncio
from aide.backend.backend_claude_code import query_async

async def main():
    # Run multiple queries concurrently
    tasks = []
    for i in range(3):
        task = query_async(
            system_message="You are a helpful assistant",
            user_message=f"Task {i}: Generate a function",
            model="claude-opus-4"
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    for result in results:
        output, req_time, in_tokens, out_tokens, info = result
        print(f"Completed in {req_time:.2f}s with ~{in_tokens + out_tokens} tokens")

asyncio.run(main())
```

## MCP Integration Details

The MCP (Model Context Protocol) integration enhances Claude Code's function calling capabilities:

1. **Automatic MCP Configuration**: When `use_mcp=true` and a FunctionSpec is provided, the backend automatically generates an MCP configuration that exposes the function as an MCP tool using `python -m aide.backend.mcp_server`.

2. **MCP Tool Naming**: Functions are exposed with the naming convention `mcp__aide__call_<function_name>`, following MCP security best practices.

3. **Robust MCP Server**: Enhanced MCP server implementation with two versions:
   - **Basic Server** (`aide/backend/mcp_server.py`):
     - Proper JSON-RPC error handling with standard error codes
     - Robust stdio mode communication
     - Graceful error recovery and logging
     - Single function support
   - **Advanced Server** (`aide/backend/mcp_server_advanced.py`):
     - All basic server features plus:
     - HTTP mode support (in addition to stdio)
     - Multiple function registration and management
     - Function parameter validation with JSON schema
     - Async handler support
     - Server statistics and monitoring
     - Middleware support for request/response processing
     - Health check and stats endpoints (HTTP mode)

4. **Improved Function Extraction**: Refactored function call extraction with prioritized strategies:
   - Primary: MCP tool call extraction from messages
   - Fallback: JSON extraction from response text with multiple parsing strategies

5. **Graceful Fallback**: When MCP is not available or disabled, the backend falls back to text-based function specification in prompts.

6. **Comprehensive Testing**: MCP functionality is tested in `tests/test_mcp_integration.py` and `test_mcp_standalone.py` (root level) with coverage for all refactored components.

## Systematic Performance Benchmarking

The project now includes comprehensive benchmarking capabilities to compare performance across different LLM backends:

**Usage:**
```bash
# Run benchmark on all backends with default tasks
python -m aide.utils.benchmark_backends

# Benchmark specific backends
python -m aide.utils.benchmark_backends --backends claude_code openai anthropic

# Use custom tasks directory
python -m aide.utils.benchmark_backends --tasks-dir /path/to/tasks

# Run with parallel workers
python -m aide.utils.benchmark_backends --parallel 4
```

**Features:**
- Automated benchmark execution across multiple backends
- Performance metrics collection (duration, success rate, token usage)
- Comparative analysis and reporting
- Historical tracking and trend analysis
- Human-readable reports in Markdown format
- JSON output for programmatic analysis

**Output:**
- Benchmark results saved to `~/.aide_ml/benchmarks/`
- Latest results always available at `benchmark_latest.json`
- Detailed reports generated as `benchmark_report_*.md`

## End-to-End Testing

Comprehensive end-to-end testing ensures AIDE ML works correctly across all example tasks:

**Usage:**
```bash
# Run full end-to-end tests
python -m pytest tests/test_e2e_all_tasks.py -v

# Run quick smoke test
python -m pytest tests/test_e2e_all_tasks.py::TestE2ESmoke -v

# Test specific example tasks
python -m pytest tests/test_e2e_all_tasks.py::TestE2EAllTasks::test_specific_task_bitcoin_price -v
```

**Features:**
- Automated testing of all example tasks
- Parallel execution for efficiency
- Detailed metrics extraction from outputs
- Multi-backend compatibility testing
- Task data validation
- Comprehensive test reporting

**Test Results:**
- Results saved to `~/.aide_ml/e2e_test_results/`
- JSON format with detailed metrics per task
- Success rate tracking and failure analysis
