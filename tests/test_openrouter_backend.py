"""Tests for OpenRouter backend including function calling support."""

import json
import pytest
from unittest.mock import patch, MagicMock, call
from aide.backend.backend_openrouter import query
from aide.backend.utils import FunctionSpec


class TestOpenRouterBackend:
    """Test OpenRouter backend functionality."""
    
    def test_basic_query_without_function(self):
        """Test basic text generation without function calling."""
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock_completion.usage.prompt_tokens = 50
        mock_completion.usage.completion_tokens = 100
        mock_completion.system_fingerprint = "test-fingerprint"
        mock_completion.model = "test-model"
        mock_completion.created = 1234567890
        
        with patch('aide.backend.backend_openrouter._setup_openrouter_client'):
            with patch('aide.backend.backend_openrouter.backoff_create', return_value=mock_completion):
                output, req_time, in_tokens, out_tokens, info = query(
                    system_message="You are a helpful assistant",
                    user_message="Hello",
                    model="gpt-3.5-turbo"
                )
                
                assert output == "Test response"
                assert in_tokens == 50
                assert out_tokens == 100
                assert info["model"] == "test-model"
    
    def test_function_calling_basic(self):
        """Test basic function calling capability."""
        func_spec = FunctionSpec(
            name="analyze_data",
            description="Analyze data and return results",
            json_schema={
                "type": "object",
                "properties": {
                    "analysis_type": {"type": "string"},
                    "result": {"type": "number"}
                },
                "required": ["analysis_type", "result"]
            }
        )
        
        # Mock response with function call
        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [
            MagicMock(
                id="call_123",
                function=MagicMock(
                    name="analyze_data",
                    arguments='{"analysis_type": "regression", "result": 0.95}'
                )
            )
        ]
        
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=mock_message)]
        mock_completion.usage.prompt_tokens = 100
        mock_completion.usage.completion_tokens = 50
        mock_completion.system_fingerprint = "test-fingerprint"
        mock_completion.model = "gpt-3.5-turbo"
        mock_completion.created = 1234567890
        
        with patch('aide.backend.backend_openrouter._setup_openrouter_client'):
            with patch('aide.backend.backend_openrouter.backoff_create', return_value=mock_completion) as mock_create:
                output, req_time, in_tokens, out_tokens, info = query(
                    system_message="You are a data analyst",
                    user_message="Analyze this regression model",
                    func_spec=func_spec,
                    model="gpt-3.5-turbo"
                )
                
                # Verify the API was called with tools
                mock_create.assert_called_once()
                call_args = mock_create.call_args[1]
                assert "tools" in call_args
                assert len(call_args["tools"]) == 1
                assert call_args["tools"][0]["function"]["name"] == "analyze_data"
                
                # Verify the output
                assert isinstance(output, dict)
                assert output["analysis_type"] == "regression"
                assert output["result"] == 0.95
    
    def test_function_calling_with_text_fallback(self):
        """Test function calling with fallback to text when no function is called."""
        func_spec = FunctionSpec(
            name="process_text",
            description="Process text data",
            json_schema={
                "type": "object",
                "properties": {
                    "processed": {"type": "string"}
                }
            }
        )
        
        # Mock response without function call (just text)
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(
            content="I'll process the text for you",
            tool_calls=None
        ))]
        mock_completion.usage.prompt_tokens = 75
        mock_completion.usage.completion_tokens = 25
        mock_completion.system_fingerprint = "test-fingerprint"
        mock_completion.model = "gpt-3.5-turbo"
        mock_completion.created = 1234567890
        
        with patch('aide.backend.backend_openrouter._setup_openrouter_client'):
            with patch('aide.backend.backend_openrouter.backoff_create', return_value=mock_completion):
                output, req_time, in_tokens, out_tokens, info = query(
                    system_message="You are a text processor",
                    user_message="Process this text",
                    func_spec=func_spec,
                    model="gpt-3.5-turbo"
                )
                
                # When no function is called, should return the text
                assert output == "I'll process the text for you"
    
    def test_function_calling_json_decode_error(self):
        """Test handling of malformed JSON in function arguments."""
        func_spec = FunctionSpec(
            name="test_func",
            description="Test function",
            json_schema={"type": "object", "properties": {"value": {"type": "string"}}}
        )
        
        # Mock response with malformed JSON
        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [
            MagicMock(
                id="call_123",
                function=MagicMock(
                    name="test_func",
                    arguments='{"value": invalid json}'  # Invalid JSON
                )
            )
        ]
        
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=mock_message)]
        mock_completion.usage.prompt_tokens = 50
        mock_completion.usage.completion_tokens = 50
        
        with patch('aide.backend.backend_openrouter._setup_openrouter_client'):
            with patch('aide.backend.backend_openrouter.backoff_create', return_value=mock_completion):
                with pytest.raises(json.JSONDecodeError):
                    query(
                        system_message="Test",
                        user_message="Test",
                        func_spec=func_spec,
                        model="gpt-3.5-turbo"
                    )
    
    def test_function_calling_name_mismatch(self):
        """Test handling when returned function name doesn't match expected."""
        func_spec = FunctionSpec(
            name="expected_func",
            description="Expected function",
            json_schema={"type": "object", "properties": {"value": {"type": "string"}}}
        )
        
        # Mock response with different function name
        mock_message = MagicMock()
        mock_message.content = "Fallback text response"
        mock_message.tool_calls = [
            MagicMock(
                id="call_123",
                function=MagicMock(
                    name="different_func",  # Different name
                    arguments='{"value": "test"}'
                )
            )
        ]
        
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=mock_message)]
        mock_completion.usage.prompt_tokens = 50
        mock_completion.usage.completion_tokens = 50
        mock_completion.system_fingerprint = "test"
        mock_completion.model = "gpt-3.5-turbo"
        mock_completion.created = 123
        
        with patch('aide.backend.backend_openrouter._setup_openrouter_client'):
            with patch('aide.backend.backend_openrouter.backoff_create', return_value=mock_completion):
                output, req_time, in_tokens, out_tokens, info = query(
                    system_message="Test",
                    user_message="Test",
                    func_spec=func_spec,
                    model="gpt-3.5-turbo"
                )
                
                # Should fall back to text content when function name doesn't match
                assert output == "Fallback text response"
    
    def test_multiple_provider_configuration(self):
        """Test that provider configuration is maintained with function calling."""
        func_spec = FunctionSpec(
            name="test_func",
            description="Test function",
            json_schema={"type": "object", "properties": {"value": {"type": "string"}}}
        )
        
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="Test"))]
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 10
        mock_completion.system_fingerprint = "test"
        mock_completion.model = "test"
        mock_completion.created = 123
        
        with patch('aide.backend.backend_openrouter._setup_openrouter_client'):
            with patch('aide.backend.backend_openrouter.backoff_create', return_value=mock_completion) as mock_create:
                query(
                    system_message="Test",
                    user_message="Test",
                    func_spec=func_spec,
                    model="gpt-3.5-turbo"
                )
                
                # Verify extra_body with provider configuration is still included
                call_args = mock_create.call_args[1]
                assert "extra_body" in call_args
                assert "provider" in call_args["extra_body"]
                assert call_args["extra_body"]["provider"]["order"] == ["Fireworks"]