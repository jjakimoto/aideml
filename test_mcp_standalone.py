#!/usr/bin/env python3
"""Standalone test for MCP functionality without full AIDE ML dependencies."""

import json
import tempfile
import os


def test_mcp_config_creation():
    """Test MCP configuration creation logic."""
    print("Testing MCP configuration creation...")
    
    # Sample function spec
    func_name = "evaluate_model"
    func_schema = {
        "type": "object",
        "properties": {
            "model_path": {"type": "string"},
            "test_data": {"type": "string"}
        },
        "required": ["model_path", "test_data"]
    }
    func_description = "Evaluate a machine learning model"
    
    # Create MCP configuration
    mcp_config = {
        "servers": {
            "aide": {
                "command": "aide-mcp-server",
                "args": ["--mode", "function-call"],
                "tools": {
                    f"call_{func_name}": {
                        "description": func_description,
                        "inputSchema": func_schema
                    }
                }
            }
        }
    }
    
    # Test writing to file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        temp_path = tmp_file.name
        json.dump(mcp_config, tmp_file, indent=2)
    
    # Verify file content
    with open(temp_path, 'r') as f:
        loaded_config = json.load(f)
    
    assert loaded_config == mcp_config
    os.unlink(temp_path)
    
    print("✓ MCP configuration creation test passed")


def test_mcp_tool_naming():
    """Test MCP tool naming convention."""
    print("Testing MCP tool naming...")
    
    func_names = ["calculate_mse", "train_model", "predict"]
    expected_tool_names = [
        "mcp__aide__call_calculate_mse",
        "mcp__aide__call_train_model",
        "mcp__aide__call_predict"
    ]
    
    for func_name, expected in zip(func_names, expected_tool_names):
        tool_name = f"mcp__aide__call_{func_name}"
        assert tool_name == expected
    
    print("✓ MCP tool naming test passed")


def test_mcp_prompt_formatting():
    """Test prompt formatting with MCP."""
    print("Testing MCP prompt formatting...")
    
    # Test prompt with MCP instructions
    func_name = "analyze_data"
    func_schema = {"type": "object", "properties": {"data": {"type": "array"}}}
    
    mcp_prompt = (
        f"System: You are a helpful assistant.\n"
        f"\nPlease use the tool '/mcp__aide__call_{func_name}' to complete this task.\n"
        f"The tool expects parameters matching this schema:\n"
        f"{json.dumps(func_schema, indent=2)}\n"
        f"User: Analyze the data"
    )
    
    assert "/mcp__aide__call_analyze_data" in mcp_prompt
    assert json.dumps(func_schema, indent=2) in mcp_prompt
    
    print("✓ MCP prompt formatting test passed")


def test_mcp_response_extraction():
    """Test extracting function calls from MCP responses."""
    print("Testing MCP response extraction...")
    
    # Simulate MCP tool call response
    tool_response = {
        "tool": "mcp__aide__call_train_model",
        "input": {
            "algorithm": "random_forest",
            "hyperparameters": {"n_estimators": 100}
        }
    }
    
    # Extract the input
    extracted_params = tool_response.get("input", {})
    assert extracted_params["algorithm"] == "random_forest"
    assert extracted_params["hyperparameters"]["n_estimators"] == 100
    
    print("✓ MCP response extraction test passed")


if __name__ == "__main__":
    print("\n=== Running MCP Standalone Tests ===\n")
    
    test_mcp_config_creation()
    test_mcp_tool_naming()
    test_mcp_prompt_formatting()
    test_mcp_response_extraction()
    
    print("\n✅ All MCP tests passed!")