# Claude Code Integration Plan for AIDE ML

## Overview
This document outlines the plan to integrate Claude Code SDK as a new backend option for AIDE ML, allowing users to leverage Claude's advanced coding capabilities through the Claude Code interface.

## What Has Been Implemented

### 1. Backend Module (`aide/backend/backend_claude_code.py`)
- ✅ Full implementation of Claude Code backend
- ✅ Subscription-based authentication (default)
- ✅ Optional API key authentication
- ✅ Support for Bedrock and Vertex AI
- ✅ Function calling compatibility
- ✅ Async-to-sync wrapper for AIDE ML compatibility

### 2. Backend Registration (`aide/backend/__init__.py`)
- ✅ Imported `backend_claude_code` module
- ✅ Added to `provider_to_query_func` mapping
- ✅ Extended `query()` function to support explicit backend selection
- ✅ Backend parameter allows bypassing model-based provider detection

### 3. Documentation
- ✅ Comprehensive integration plan (`docs/plan.md`)
- ✅ Usage examples (`docs/claude_code_usage_example.md`)
- ✅ Test scripts for verification

### 4. Test Scripts
- ✅ `test_claude_code_backend.py` - Unit tests for backend functionality
- ✅ `test_integration_aide_ml.py` - Integration tests with AIDE ML

## Quick Start

### 1. Install Claude Code SDK
```bash
pip install claude-code-sdk
```

### 2. Configure AIDE ML to use Claude Code

#### Option A: Using Claude Code Subscription (Recommended)
No API key needed - uses your Claude Code subscription:

```python
# In your AIDE ML config
config = {
    "backend": "claude_code",
    "backend_options": {
        "provider": "subscription",  # This is the default
        "model": "claude-opus-4",
        "max_turns": 3,
        "temperature": 0.2
    }
}
```

#### Option B: Using Anthropic API Key (Optional)
If you prefer to use an API key:

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

```python
config = {
    "backend": "claude_code",
    "backend_options": {
        "provider": "anthropic",
        "model": "claude-opus-4"
    }
}
```

### 3. Run AIDE ML with Claude Code

```bash
# Using subscription auth (default)
python run_aide.py --task example_tasks/house_prices.yaml --backend claude_code

# Or with explicit configuration
python run_aide.py \
    --task example_tasks/house_prices.yaml \
    --backend claude_code \
    --backend-opt provider=subscription \
    --backend-opt model=claude-opus-4
```

## Next Steps for Full Integration

### 1. Install Dependencies
```bash
pip install claude-code-sdk
# or add to requirements.txt
```

### 2. Update Configuration System
Add Claude Code configuration to `aide/utils/config.py`:
```python
# In Config class
self.backend = "claude_code"  # or from CLI args
self.backend_options = {
    "provider": "subscription",
    "model": "claude-opus-4",
    "max_turns": 3,
    "temperature": 0.2
}
```

### 3. Update CLI Interface
Modify `aide/run.py` to accept Claude Code backend:
```python
parser.add_argument("--backend", choices=["openai", "anthropic", "claude_code", ...])
parser.add_argument("--backend-opt", action="append", help="Backend options as key=value")
```

### 4. Test with Example Tasks
```bash
# Test with bitcoin price prediction
python run_aide.py \
    --task aide/example_tasks/bitcoin_price.md \
    --backend claude_code \
    --backend-opt provider=subscription

# Test with house prices
python run_aide.py \
    --task aide/example_tasks/house_prices.md \
    --backend claude_code \
    --backend-opt provider=subscription
```

### 5. Update Agent to Use Backend Parameter
In `aide/agent.py`, ensure the backend parameter is passed:
```python
output = query(
    backend=self.cfg.backend,  # Add this line
    system_message=system_message,
    user_message=user_message,
    model=self.cfg.model,
    # ... other parameters
)
```

## Testing Checklist

- [ ] Install claude-code-sdk
- [ ] Run unit tests: `python test_claude_code_backend.py`
- [ ] Run integration tests: `python test_integration_aide_ml.py`
- [ ] Test with bitcoin_price task
- [ ] Test with house_prices task
- [ ] Verify subscription authentication works
- [ ] Test function calling with review_func_spec
- [ ] Compare performance with other backends

## Configuration Examples

### Subscription Auth (Default)
```yaml
backend: claude_code
backend_options:
  provider: subscription
  model: claude-opus-4
  max_turns: 3
```

### API Key Auth
```yaml
backend: claude_code
backend_options:
  provider: anthropic
  model: claude-opus-4
```

### AWS Bedrock
```yaml
backend: claude_code
backend_options:
  provider: bedrock
  model: claude-opus-4
```

## Benefits of Claude Code Integration

1. **Subscription-based**: No API key management for most users
2. **Code-optimized**: Better understanding of code structure and patterns
3. **Multi-turn support**: Improved iterative refinement
4. **Fast iteration**: Claude Code's efficiency for coding tasks
5. **Tool use**: Native support for function calling

## Known Limitations

1. Token counting is estimated (Claude Code SDK may not provide exact counts)
2. Async API requires sync wrapper for AIDE ML compatibility
3. Some Claude Code features (like MCP) not yet integrated

## Support and Troubleshooting

For issues with:
- **Authentication**: Check Claude Code is installed and logged in
- **Performance**: Adjust max_turns and temperature parameters
- **Compatibility**: Ensure claude-code-sdk version is up to date

## Future Enhancements

1. **Hybrid Approach**: Use Claude Code for code generation and other models for analysis
2. **Specialized Prompts**: Leverage Claude Code's optimized prompting for different ML tasks
3. **Tool Extensions**: Integrate Claude Code's MCP for enhanced capabilities
4. **Performance Monitoring**: Track and compare performance across backends