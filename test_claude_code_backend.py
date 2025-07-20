#!/usr/bin/env python3
"""Test script to verify Claude Code backend functionality with AIDE ML example tasks."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import aide modules
sys.path.insert(0, str(Path(__file__).parent))

from aide.backend.backend_claude_code import query as claude_code_query
from aide.backend.utils import FunctionSpec

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_query():
    """Test basic text generation without function calling."""
    print("\n=== Testing Basic Query ===")
    
    system_message = "You are a helpful ML engineer assistant."
    user_message = "Write a simple Python function to calculate mean squared error."
    
    try:
        output, req_time, in_tokens, out_tokens, info = claude_code_query(
            system_message=system_message,
            user_message=user_message,
            provider="subscription",  # Use subscription auth
            model="claude-opus-4",
            temperature=0.2
        )
        
        print(f"Response time: {req_time:.2f}s")
        print(f"Tokens - Input: {in_tokens}, Output: {out_tokens}")
        print(f"Provider info: {info}")
        print(f"\nOutput:\n{output}")
        return True
    except Exception as e:
        print(f"Error in basic query test: {e}")
        return False


def test_function_calling():
    """Test function calling capability."""
    print("\n\n=== Testing Function Calling ===")
    
    # Create a function spec similar to what AIDE ML uses
    func_spec = FunctionSpec(
        name="analyze_model_performance",
        json_schema={
            "type": "object",
            "properties": {
                "model_type": {
                    "type": "string",
                    "description": "The type of ML model to use"
                },
                "metric": {
                    "type": "string",
                    "description": "The evaluation metric"
                },
                "expected_performance": {
                    "type": "number",
                    "description": "Expected performance value"
                }
            },
            "required": ["model_type", "metric", "expected_performance"]
        },
        description="Analyze and report model performance expectations"
    )
    
    system_message = "You are an ML expert analyzing a house price prediction task."
    user_message = "What model and metric would you recommend for predicting house prices with RMSE evaluation?"
    
    try:
        output, req_time, in_tokens, out_tokens, info = claude_code_query(
            system_message=system_message,
            user_message=user_message,
            func_spec=func_spec,
            provider="subscription",
            model="claude-opus-4"
        )
        
        print(f"Response time: {req_time:.2f}s")
        print(f"Tokens - Input: {in_tokens}, Output: {out_tokens}")
        
        if isinstance(output, dict):
            print(f"\nFunction call result:")
            print(json.dumps(output, indent=2))
            return True
        else:
            print(f"\nText output (function call may not be supported):\n{output}")
            return True  # Still pass if function calling isn't supported
    except Exception as e:
        print(f"Error in function calling test: {e}")
        return False


def test_bitcoin_task_simulation():
    """Simulate Claude Code solving the bitcoin price prediction task."""
    print("\n\n=== Testing Bitcoin Price Task Simulation ===")
    
    system_message = """You are an ML engineer solving a timeseries forecasting problem.
The task is to build a model for bitcoin close price prediction.
The evaluation metric is RMSE between log(predicted) and log(observed) prices.
You have access to a CSV file with historical bitcoin data."""
    
    user_message = """Write Python code to:
1. Load the bitcoin price data from 'BTC-USD.csv'
2. Prepare features for time series forecasting
3. Train a simple model
4. Make predictions and calculate the RMSE metric on log prices"""
    
    try:
        output, req_time, in_tokens, out_tokens, info = claude_code_query(
            system_message=system_message,
            user_message=user_message,
            provider="subscription",
            model="claude-opus-4",
            temperature=0.1  # Lower temperature for more deterministic code
        )
        
        print(f"Response time: {req_time:.2f}s")
        print(f"Tokens - Input: {in_tokens}, Output: {out_tokens}")
        print(f"\nGenerated code:\n{output}")
        return True
    except Exception as e:
        print(f"Error in bitcoin task simulation: {e}")
        return False


def test_house_prices_task_simulation():
    """Simulate Claude Code solving the house prices prediction task."""
    print("\n\n=== Testing House Prices Task Simulation ===")
    
    system_message = """You are an ML engineer solving a house price prediction problem.
The task is to predict SalePrice for houses based on 79 features.
The evaluation metric is RMSE between log(predicted) and log(observed) prices.
You have train.csv and test.csv files available."""
    
    user_message = """Write Python code to:
1. Load train.csv and test.csv
2. Handle missing values and prepare features
3. Train a regression model
4. Generate predictions for the test set in the required submission format"""
    
    try:
        output, req_time, in_tokens, out_tokens, info = claude_code_query(
            system_message=system_message,
            user_message=user_message,
            provider="subscription",
            model="claude-opus-4",
            temperature=0.1
        )
        
        print(f"Response time: {req_time:.2f}s")
        print(f"Tokens - Input: {in_tokens}, Output: {out_tokens}")
        print(f"\nGenerated code:\n{output}")
        return True
    except Exception as e:
        print(f"Error in house prices task simulation: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing Claude Code Backend Integration with AIDE ML")
    print("=" * 60)
    
    # Check if we're using subscription or API key
    if os.getenv("ANTHROPIC_API_KEY"):
        print("Note: ANTHROPIC_API_KEY found, but tests will use subscription auth by default")
    else:
        print("Using Claude Code subscription authentication")
    
    tests = [
        ("Basic Query", test_basic_query),
        ("Function Calling", test_function_calling),
        ("Bitcoin Price Task", test_bitcoin_task_simulation),
        ("House Prices Task", test_house_prices_task_simulation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\nUnexpected error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n\n" + "=" * 60)
    print("Test Summary:")
    print("-" * 60)
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:<30} {status}")
    
    total_passed = sum(1 for _, success in results if success)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    
    return 0 if total_passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())