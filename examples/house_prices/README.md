# House Price Prediction Tutorial

This tutorial demonstrates how to use AIDE ML to build a regression model for predicting house sale prices using both Claude Code and Anthropic backends.

## 📋 Task Overview

**Goal**: Predict the sales price for each house in the test set  
**Evaluation Metric**: Root-Mean-Squared-Error (RMSE) between logarithm of predicted and observed prices  
**Data**: Ames Housing dataset with 79 explanatory variables  
**Files**:
- `train.csv`: Training data with sale prices
- `test.csv`: Test data without sale prices
- `data_description.txt`: Detailed description of all features
- `sample_submission.csv`: Example submission format

## 🚀 Quick Start

### 1. Using Claude Code Backend

```bash
# Basic run
python run_house_prices.py --backend claude_code

# With specific model and enhanced features
python run_house_prices.py --backend claude_code --model claude-opus-4 --use-mcp

# With more iterations for better results
python run_house_prices.py --backend claude_code --iterations 15 --workers 6
```

### 2. Using Anthropic Backend

```bash
# Basic run
python run_house_prices.py --backend anthropic

# With specific model
python run_house_prices.py --backend anthropic --model claude-3-5-sonnet-20241022

# With custom parameters
python run_house_prices.py --backend anthropic --timeout 400 --iterations 12
```

### 3. Compare Both Backends

```bash
# Run comparison
python run_house_prices.py --compare

# Compare with more iterations
python run_house_prices.py --compare --iterations 20
```

## 📊 Understanding the Task

The house price prediction task is more complex than Bitcoin prediction because:

1. **79 Features**: Mix of numerical and categorical variables
2. **Missing Data**: Many features have missing values requiring imputation
3. **Feature Engineering**: Opportunity to create interaction features
4. **Diverse Data Types**: Includes areas, counts, qualities, conditions, and dates
5. **Domain Knowledge**: Understanding real estate helps feature selection

### Key Feature Categories

1. **Physical Attributes**:
   - Living area (GrLivArea)
   - Number of rooms and bathrooms
   - Garage size and type
   - Basement information

2. **Quality and Condition**:
   - Overall quality (OverallQual)
   - Overall condition (OverallCond)
   - Kitchen quality
   - Exterior quality

3. **Location and Neighborhood**:
   - Neighborhood
   - Proximity to various conditions
   - Lot configuration

4. **Temporal Features**:
   - Year built
   - Year remodeled
   - Month and year sold

## 🔧 Script Options

The `run_house_prices.py` script supports:

```
Options:
  --backend           Backend to use (claude_code, anthropic)
  --model            Model name (e.g., claude-opus-4)
  --use-mcp          Enable MCP for Claude Code backend
  --timeout          Request timeout in seconds (default: 400)
  --workers          Number of parallel workers (default: 4)
  --iterations       Number of iterations (default: 15)
  --compare          Run with both backends and compare
  --output-dir       Directory for outputs (default: outputs/)
  --debug            Enable debug logging
  --feature-eng      Level of feature engineering (basic, advanced, auto)
```

## 📈 Expected Outputs

After running, you'll find:

1. **Solution Code**: `outputs/house_prices_<backend>_<timestamp>/solution.py`
   - Complete ML pipeline from data loading to predictions

2. **Submission File**: `outputs/house_prices_<backend>_<timestamp>/submission.csv`
   - Predictions in the required format

3. **Journal Log**: `outputs/house_prices_<backend>_<timestamp>/journal.json`
   - Detailed log of AIDE ML's reasoning process

4. **Performance Report**: `outputs/house_prices_<backend>_<timestamp>/report.md`
   - Summary of approaches tried and results

5. **Tree Visualization**: `outputs/house_prices_<backend>_<timestamp>/tree.html`
   - Interactive exploration of the solution space

## 🎯 Typical Solutions

AIDE ML explores various approaches:

### 1. **Preprocessing Strategies**
- Handling missing values (median/mode imputation, forward fill)
- Encoding categorical variables (one-hot, target encoding)
- Feature scaling (standardization, normalization)
- Outlier detection and treatment

### 2. **Feature Engineering**
- Creating polynomial features
- Interaction terms (e.g., TotalSF = GrLivArea + TotalBsmtSF)
- Age features (e.g., HouseAge = YrSold - YearBuilt)
- Quality aggregations

### 3. **Model Approaches**
- **Linear Models**: Ridge, Lasso, ElasticNet
- **Tree-based**: Random Forest, XGBoost, LightGBM
- **Ensemble Methods**: Voting, Stacking, Blending
- **Neural Networks**: For complex patterns

### 4. **Advanced Techniques**
- Cross-validation strategies (K-fold, stratified)
- Hyperparameter tuning (GridSearch, Bayesian optimization)
- Feature selection (importance-based, recursive)
- Ensemble of different preprocessing pipelines

## 📊 Performance Expectations

Typical performance ranges:
- **RMSE (log scale)**: 0.11 - 0.13 for good models
- **Kaggle Leaderboard**: Top 20% with score < 0.125
- **Execution time**: 3-8 minutes per iteration
- **Total time**: 45-120 minutes for full run

### Backend Comparison

| Aspect | Claude Code | Anthropic |
|--------|-------------|-----------|
| Feature Engineering | Excellent with domain insights | Good general approach |
| Model Selection | Task-aware recommendations | Comprehensive exploration |
| Code Organization | Clean, modular pipelines | Standard ML workflow |
| Error Recovery | Smart debugging and fixes | Basic error handling |

## 🔍 Advanced Usage

### Custom Feature Engineering

Guide AIDE ML with specific instructions:
```bash
python run_house_prices.py --backend claude_code \
    --feature-eng advanced \
    --custom-prompt "Focus on creating interaction features between area measurements and quality ratings"
```

### Ensemble Focus

Direct AIDE ML to use ensembles:
```bash
python run_house_prices.py --backend claude_code \
    --custom-prompt "Build an ensemble using XGBoost, LightGBM, and CatBoost with stacking"
```

## 🐛 Troubleshooting

### Common Issues

1. **Memory Errors**:
   ```bash
   # Reduce workers and batch size
   python run_house_prices.py --workers 1 --backend-opt batch_size=1000
   ```

2. **Feature Engineering Explosion**:
   - Too many polynomial features can cause memory issues
   - Guide AIDE ML to be selective with features

3. **Overfitting**:
   - Complex models may overfit
   - Ensure proper cross-validation is used

4. **Submission Format Errors**:
   - Check that Id column matches test.csv
   - Ensure no missing predictions

### Debug Mode

For detailed debugging:
```bash
python run_house_prices.py --backend claude_code --debug --iterations 1
```

## 📚 Learning Opportunities

This task teaches:
1. **Data Preprocessing**: Handling real-world messy data
2. **Feature Engineering**: Creating meaningful features
3. **Model Selection**: Choosing appropriate algorithms
4. **Ensemble Methods**: Combining models effectively
5. **Cross-validation**: Proper model evaluation

## 🚀 Challenge Yourself

1. **Beat the Baseline**: Aim for RMSE < 0.130
2. **Minimize Features**: Get good results with < 20 features
3. **Interpretability**: Build an interpretable model
4. **Speed**: Optimize for fast training and prediction
5. **Robustness**: Handle edge cases and outliers

## 📖 Additional Resources

- [Ames Housing Dataset Paper](http://jse.amstat.org/v19n3/decock.pdf)
- [Feature Engineering Guide](https://www.kaggle.com/learn/feature-engineering)
- [Kaggle House Prices Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

## 💡 Tips from Experience

1. **OverallQual is King**: This feature is highly predictive
2. **Living Area Matters**: Square footage features are crucial
3. **Handle Skewness**: Log transform skewed features
4. **Missing Indicators**: Sometimes missing values are informative
5. **Neighborhood Effects**: Location encoding can boost performance

Happy modeling! 🏠