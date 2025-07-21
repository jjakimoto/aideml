# Claude Code Usage Examples for AIDE ML

This document provides practical examples of using Claude Code SDK with AIDE ML. For a complete overview of the integration, see the main documentation in [CLAUDE.md](../CLAUDE.md).

## Table of Contents
- [Basic Usage](#basic-usage)
- [Authentication Methods](#authentication-methods)
- [Advanced Features](#advanced-features)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

## Basic Usage

### Running AIDE ML with Claude Code

The simplest way to use Claude Code with AIDE ML:

```bash
# Using subscription authentication (default)
python run_aide.py --task aide/example_tasks/house_prices.md --backend claude_code

# With explicit model selection
python run_aide.py --task aide/example_tasks/bitcoin_price.md \
    --backend claude_code \
    --backend-opt model=claude-opus-4
```

### Python API Usage

```python
from aide.backend.backend_claude_code import query

# Basic query
output, req_time, in_tokens, out_tokens, info = query(
    system_message="You are an ML engineer solving a regression problem.",
    user_message="Create a function to preprocess housing price data.",
    model="claude-opus-4",
    temperature=0.2
)

print(f"Response generated in {req_time:.2f}s")
print(f"Output:\n{output}")
```

## Authentication Methods

### 1. Subscription Authentication (Default)

No API key required - uses your Claude Code subscription:

```bash
# Simply specify the backend
python run_aide.py --task task.md --backend claude_code
```

```python
# In Python
from aide.backend.backend_claude_code import query

output, _, _, _, _ = query(
    system_message="System prompt",
    user_message="User message",
    provider="subscription"  # This is the default
)
```

### 2. Anthropic API Key

```bash
# Set environment variable
export ANTHROPIC_API_KEY="your-api-key-here"

# Run with API key authentication
python run_aide.py --task task.md \
    --backend claude_code \
    --backend-opt provider=anthropic
```

### 3. AWS Bedrock

```bash
# Configure AWS credentials
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_REGION="us-east-1"

# Run with Bedrock
python run_aide.py --task task.md \
    --backend claude_code \
    --backend-opt provider=bedrock
```

### 4. Google Vertex AI

```bash
# Set up Google Cloud credentials
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"

# Run with Vertex AI
python run_aide.py --task task.md \
    --backend claude_code \
    --backend-opt provider=vertex
```

## Advanced Features

### 1. Hybrid Backend Usage

Automatically route different types of queries to optimal backends:

```bash
# Default: Claude Code for code generation, GPT-4o for analysis
python run_aide.py --task aide/example_tasks/house_prices.md --backend hybrid

# Custom routing configuration
python run_aide.py --task task.md \
    --backend hybrid \
    --backend-opt agent.hybrid.code_backend=claude_code \
    --backend-opt agent.hybrid.code_model=claude-opus-4 \
    --backend-opt agent.hybrid.analysis_backend=openai \
    --backend-opt agent.hybrid.analysis_model=gpt-4o-mini
```

### 2. MCP (Model Context Protocol) Integration

Enable enhanced function calling with MCP:

```bash
# Basic MCP usage
python run_aide.py --task task.md \
    --backend claude_code \
    --backend-opt use_mcp=true

# Advanced MCP with HTTP mode
python run_aide.py --task task.md \
    --backend claude_code \
    --backend-opt use_mcp=true \
    --backend-opt use_advanced_mcp=true \
    --backend-opt mcp_http_mode=true \
    --backend-opt mcp_http_port=8080
```

### 3. Function Calling

```python
from aide.backend.backend_claude_code import query
from aide.backend.utils import FunctionSpec

# Define a function specification
func_spec = FunctionSpec(
    name="evaluate_model",
    description="Evaluate ML model performance",
    json_schema={
        "type": "object",
        "properties": {
            "model_type": {"type": "string"},
            "metrics": {"type": "array", "items": {"type": "string"}},
            "cross_validation": {"type": "boolean"}
        },
        "required": ["model_type", "metrics"]
    }
)

# Query with function calling
output, _, _, _, _ = query(
    system_message="You are evaluating ML models.",
    user_message="Evaluate a random forest model for the housing price task.",
    func_spec=func_spec,
    use_mcp=True  # Enable MCP for better function calling
)
```

### 4. Async Support

```python
import asyncio
from aide.backend.backend_claude_code import query_async

async def process_multiple_tasks():
    # Create multiple concurrent queries
    tasks = []
    
    for task_name in ["preprocessing", "feature_engineering", "model_training"]:
        task = query_async(
            system_message="You are an ML engineer.",
            user_message=f"Generate code for {task_name}",
            model="claude-opus-4"
        )
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    for i, (output, req_time, _, _, _) in enumerate(results):
        print(f"Task {i+1} completed in {req_time:.2f}s")

# Run the async function
asyncio.run(process_multiple_tasks())
```

## Performance Optimization

### 1. Enable Performance Monitoring

Track and analyze backend performance:

```bash
# View performance summary
python -m aide.utils.view_performance summary

# Compare Claude Code with other backends
python -m aide.utils.view_performance compare --backends claude_code openai

# Export performance data
python -m aide.utils.view_performance export claude_performance.json
```

### 2. Optimize Query Parameters

```bash
# Reduce tokens for faster responses
python run_aide.py --task task.md \
    --backend claude_code \
    --backend-opt temperature=0.1 \
    --backend-opt max_tokens=1000

# Use specialized prompts for ML tasks
python run_aide.py --task task.md \
    --backend claude_code \
    --backend-opt use_specialized_prompts=true
```

### 3. Benchmark Different Configurations

```bash
# Run comprehensive benchmarks
python -m aide.utils.benchmark_backends \
    --backends claude_code openai anthropic \
    --parallel 4

# View benchmark results
cat ~/.aide_ml/benchmarks/benchmark_latest.json
```

## Troubleshooting

### Common Issues

1. **Authentication Error**
   ```bash
   # Check Claude Code is installed and authenticated
   claude-code --version
   
   # Re-authenticate if needed
   claude-code auth login
   ```

2. **Provider Not Available**
   ```bash
   # Verify provider configuration
   python -c "from aide.backend.backend_claude_code import CLAUDE_CODE_PROVIDERS; print(CLAUDE_CODE_PROVIDERS)"
   ```

3. **Token Limit Exceeded**
   ```bash
   # Use lower max_tokens or split the task
   python run_aide.py --task task.md \
       --backend claude_code \
       --backend-opt max_tokens=500
   ```

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Now run your queries - detailed logs will be shown
from aide.backend.backend_claude_code import query
# ... your code ...
```

### Testing Your Configuration

Run the test script to verify your setup:

```bash
# Test basic functionality
python test_claude_code_backend.py

# Run integration tests
python test_integration_aide_ml.py

# Test specific features
pytest tests/test_mcp_integration.py -v
```

## Example ML Tasks

### 1. Classification Task

```bash
# Binary classification with specialized prompts
python run_aide.py \
    --task "Classify customer churn using transaction data" \
    --backend claude_code \
    --backend-opt use_specialized_prompts=true \
    --backend-opt temperature=0.1
```

### 2. Time Series Forecasting

```bash
# Use hybrid backend for complex time series task
python run_aide.py \
    --task "Forecast bitcoin prices for next 7 days" \
    --backend hybrid \
    --backend-opt agent.hybrid.code_backend=claude_code \
    --backend-opt agent.hybrid.analysis_backend=openai
```

### 3. NLP Task

```bash
# Text classification with MCP
python run_aide.py \
    --task "Sentiment analysis on product reviews" \
    --backend claude_code \
    --backend-opt use_mcp=true \
    --backend-opt use_specialized_prompts=true
```

## Best Practices

1. **Start with Default Settings**: The default subscription authentication and model settings work well for most tasks.

2. **Use Hybrid Backend for Complex Projects**: Let the hybrid backend automatically route queries to the most appropriate model.

3. **Enable MCP for Function-Heavy Tasks**: When your task involves multiple function calls, MCP provides better structured outputs.

4. **Monitor Performance**: Use the performance monitoring tools to optimize your configuration over time.

5. **Leverage Specialized Prompts**: For ML-specific tasks, specialized prompts improve code quality and catch common pitfalls.

## Additional Resources

- [AIDE ML Documentation](../README.md)
- [Claude Code SDK Documentation](https://github.com/anthropics/claude-code-sdk)
- [Integration Plan](plan.md)
- [Latest Status Report](memos/)

For more examples and advanced usage, check the test files:
- `test_claude_code_backend.py` - Basic functionality tests
- `test_integration_aide_ml.py` - Integration examples
- `tests/test_mcp_integration.py` - MCP usage examples