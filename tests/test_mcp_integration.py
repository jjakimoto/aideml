#!/usr/bin/env python3
"""Test MCP (Model Context Protocol) integration in Claude Code backend."""

import json
import logging
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add the parent directory to the path so we can import aide modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from aide.backend.backend_claude_code import (
    query as claude_code_query,
    _convert_func_spec_to_mcp_tool,
    _create_mcp_config_for_func_spec,
    _extract_function_call,
)
from aide.backend.utils import FunctionSpec
from aide.backend.mcp_server import AideMCPServer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestMCPIntegration:
    """Test MCP integration functionality."""
    
    def test_convert_func_spec_to_mcp_tool(self):
        """Test converting FunctionSpec to MCP tool name."""
        func_spec = FunctionSpec(
            name="calculate_mse",
            json_schema={
                "type": "object",
                "properties": {
                    "predictions": {"type": "array", "items": {"type": "number"}},
                    "targets": {"type": "array", "items": {"type": "number"}}
                },
                "required": ["predictions", "targets"]
            },
            description="Calculate mean squared error"
        )
        
        tool_name = _convert_func_spec_to_mcp_tool(func_spec)
        assert tool_name == "mcp__aide__call_calculate_mse"
    
    def test_create_mcp_config(self):
        """Test creating MCP configuration from FunctionSpec."""
        func_spec = FunctionSpec(
            name="evaluate_model",
            json_schema={
                "type": "object",
                "properties": {
                    "model_path": {"type": "string"},
                    "test_data": {"type": "string"}
                },
                "required": ["model_path", "test_data"]
            },
            description="Evaluate a machine learning model"
        )
        
        # Test without file path (in-memory)
        config = _create_mcp_config_for_func_spec(func_spec)
        assert "servers" in config
        assert "aide" in config["servers"]
        assert "tools" in config["servers"]["aide"]
        assert "call_evaluate_model" in config["servers"]["aide"]["tools"]
        
        # Test with file path
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        try:
            config = _create_mcp_config_for_func_spec(func_spec, temp_path)
            
            # Check file was created
            assert os.path.exists(temp_path)
            
            # Check file content
            with open(temp_path, 'r') as f:
                saved_config = json.load(f)
            
            assert saved_config == config
        finally:
            os.unlink(temp_path)
    
    def test_extract_function_call_with_mcp(self):
        """Test extracting function calls from MCP tool responses."""
        func_spec = FunctionSpec(
            name="train_model",
            json_schema={
                "type": "object",
                "properties": {
                    "algorithm": {"type": "string"},
                    "hyperparameters": {"type": "object"}
                }
            },
            description="Train a model"
        )
        
        # Mock message with MCP tool call
        mock_message = Mock()
        mock_message.tool_calls = [{
            "tool": "mcp__aide__call_train_model",
            "input": {
                "algorithm": "random_forest",
                "hyperparameters": {"n_estimators": 100}
            }
        }]
        
        result = _extract_function_call("", func_spec, [mock_message])
        assert result == {
            "algorithm": "random_forest",
            "hyperparameters": {"n_estimators": 100}
        }
    
    def test_extract_function_call_fallback(self):
        """Test fallback to text extraction when no MCP tool calls."""
        func_spec = FunctionSpec(
            name="predict",
            json_schema={"type": "object"},
            description="Make predictions"
        )
        
        # Test with JSON in code block
        response_text = """
        Here's the function call:
        ```json
        {"data": [1, 2, 3], "model": "linear"}
        ```
        """
        
        result = _extract_function_call(response_text, func_spec, [])
        assert result == {"data": [1, 2, 3], "model": "linear"}
    
    @patch('aide.backend.backend_claude_code.claude_query')
    def test_query_with_mcp_enabled(self, mock_claude_query):
        """Test query function with MCP enabled."""
        # Mock the async generator response
        async def mock_query_gen(prompt, options):
            mock_msg = Mock()
            mock_msg.content = '{"result": "success"}'
            mock_msg.tool_calls = [{
                "tool": "mcp__aide__call_test_func",
                "input": {"param": "value"}
            }]
            yield mock_msg
        
        mock_claude_query.return_value = mock_query_gen("", None)
        
        func_spec = FunctionSpec(
            name="test_func",
            json_schema={
                "type": "object",
                "properties": {"param": {"type": "string"}}
            },
            description="Test function"
        )
        
        # Test with MCP enabled
        output, req_time, in_tokens, out_tokens, info = claude_code_query(
            system_message="Test system message",
            user_message="Test user message",
            func_spec=func_spec,
            use_mcp=True,
            provider="subscription"
        )
        
        # Check that MCP was enabled in info
        assert info["mcp_enabled"] is True
        
        # Check that the function was called with MCP options
        mock_claude_query.assert_called_once()
        call_args = mock_claude_query.call_args
        options = call_args[1]["options"]
        
        # Should have allowed_tools set
        assert hasattr(options, "allowed_tools")
    
    def test_mcp_server_basic(self):
        """Test basic MCP server functionality."""
        server = AideMCPServer()
        
        # Register a test function
        server.register_function(
            "test_function",
            {
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                },
                "required": ["input"]
            }
        )
        
        # Test tools/list
        request = {
            "method": "tools/list",
            "params": {}
        }
        response = server.handle_request(request)
        
        assert "tools" in response
        assert len(response["tools"]) == 1
        assert response["tools"][0]["name"] == "call_test_function"
        
        # Test tools/call
        request = {
            "method": "tools/call",
            "params": {
                "name": "call_test_function",
                "arguments": {"input": "test value"}
            }
        }
        response = server.handle_request(request)
        
        assert "content" in response
        content = json.loads(response["content"][0]["text"])
        assert content["status"] == "success"
        assert content["result"]["input"] == "test value"
    
    def test_mcp_prompt_formatting(self):
        """Test that prompts are formatted correctly when MCP is enabled."""
        from aide.backend.backend_claude_code import _format_prompt_for_claude_code
        
        func_spec = FunctionSpec(
            name="analyze_data",
            json_schema={
                "type": "object",
                "properties": {
                    "data_path": {"type": "string"}
                }
            },
            description="Analyze data"
        )
        
        # Test with MCP disabled
        prompt_no_mcp = _format_prompt_for_claude_code(
            "System message",
            "User message",
            func_spec,
            use_mcp=False
        )
        
        assert "You must call the function" in prompt_no_mcp
        assert "/mcp__aide__call_" not in prompt_no_mcp
        
        # Test with MCP enabled
        prompt_with_mcp = _format_prompt_for_claude_code(
            "System message",
            "User message",
            func_spec,
            use_mcp=True
        )
        
        assert "/mcp__aide__call_analyze_data" in prompt_with_mcp
        assert "You must call the function" not in prompt_with_mcp


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])