# AIDE ML Examples and Tutorials

This directory contains comprehensive tutorials for running AIDE ML with different example tasks and backends. These tutorials demonstrate how to use AIDE ML with both the Claude Code backend and the original Anthropic backend.

## 📁 Directory Structure

```
examples/
├── README.md              # This file
├── bitcoin_price/         # Bitcoin price prediction tutorial
│   ├── README.md          # Detailed tutorial for bitcoin task
│   └── run_bitcoin.py     # Script to run bitcoin task
├── house_prices/          # House price prediction tutorial  
│   ├── README.md          # Detailed tutorial for house prices task
│   └── run_house_prices.py # Script to run house prices task
└── scripts/               # Utility scripts
    ├── setup_env.sh       # Environment setup script
    ├── setup_api_keys.py  # API key configuration helper
    └── compare_backends.py # Backend performance comparison
```

## 🚀 Quick Start

1. **Set up your environment**:
   ```bash
   cd examples/scripts
   bash setup_env.sh
   ```

2. **Configure API keys**:
   ```bash
   python setup_api_keys.py
   ```

3. **Run your first example**:
   ```bash
   # Using Claude Code backend
   cd ../bitcoin_price
   python run_bitcoin.py --backend claude_code
   
   # Using Anthropic backend
   python run_bitcoin.py --backend anthropic
   ```

## 📚 Available Tutorials

### 1. Bitcoin Price Prediction
A time series forecasting task where AIDE ML builds a model to predict Bitcoin closing prices.

- **Task Type**: Time Series Forecasting
- **Evaluation Metric**: RMSE on log-transformed prices
- **Data**: Historical BTC-USD price data
- [📖 Full Tutorial](bitcoin_price/README.md)

### 2. House Price Prediction
A regression task where AIDE ML predicts house sale prices using 79 different features.

- **Task Type**: Regression
- **Evaluation Metric**: RMSE on log-transformed prices
- **Data**: Ames Housing dataset with train/test split
- [📖 Full Tutorial](house_prices/README.md)

## 🔧 Backend Options

### Claude Code Backend
The Claude Code backend provides enhanced capabilities through the Claude Code SDK:
- Function calling support via MCP (Model Context Protocol)
- Improved code generation with task-specific optimizations
- Better error handling and debugging

### Anthropic Backend  
The original Anthropic backend uses the standard Anthropic API:
- Direct API access with function calling
- Proven reliability for ML tasks
- Standard Anthropic model features

### Backend Comparison
Use the comparison script to evaluate performance differences:
```bash
cd scripts
python compare_backends.py --task bitcoin_price
```

## 🔑 API Key Setup

Both backends require API keys:

### Claude Code Backend
- Requires Claude Code SDK authentication
- Set up automatically when using Claude Code CLI

### Anthropic Backend
- Requires `ANTHROPIC_API_KEY` environment variable
- The setup script handles conversion from `ANTHROPIC_API_KEY_TILDE`

## 📊 Performance Monitoring

AIDE ML includes built-in performance monitoring:
```bash
# View performance summary
python -m aide.utils.view_performance summary

# Compare backends
python -m aide.utils.view_performance compare --backends claude_code anthropic
```

## 🛠️ Advanced Features

### Hybrid Backend
Combine multiple backends for optimal performance:
```bash
python run_bitcoin.py --backend hybrid \
    --backend-opt agent.hybrid.code_backend=claude_code \
    --backend-opt agent.hybrid.analysis_backend=anthropic
```

### MCP (Model Context Protocol)
Enable advanced function calling:
```bash
python run_bitcoin.py --backend claude_code --backend-opt use_mcp=true
```

### Specialized Prompts
Task-specific prompt enhancements are enabled by default for better results.

## 📝 Tips for Best Results

1. **Start Simple**: Begin with the bitcoin price example as it's simpler
2. **Monitor Progress**: Use the journal files to track AIDE ML's reasoning
3. **Compare Backends**: Try both backends to see which works best for your use case
4. **Check Logs**: Review `~/.aide_ml/logs/` for detailed execution logs
5. **Iterate**: AIDE ML uses tree search, so let it explore multiple solutions

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure API keys are properly set up using the setup script
2. **Environment Issues**: Make sure the conda environment is activated
3. **Memory Issues**: For large tasks, consider using `--num_workers 1`
4. **Timeout Issues**: Increase timeout with `--backend-opt timeout=600`

### Getting Help

- Check the [main documentation](../README.md)
- Review the [troubleshooting guide](../docs/troubleshooting.md)
- Submit issues on GitHub

## 🎯 Next Steps

After running these examples:
1. Try modifying the task descriptions to see how AIDE ML adapts
2. Experiment with different model configurations
3. Create your own ML tasks following the same pattern
4. Explore the generated solutions in `sample_results/`

Happy experimenting with AIDE ML! 🚀