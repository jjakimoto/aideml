"""Tests for the hybrid backend module."""

import pytest
from unittest.mock import patch, MagicMock

from aide.backend.backend_hybrid import (
    HybridConfig,
    set_hybrid_config,
    get_hybrid_config,
    detect_task_type,
    query,
    configure_hybrid_backend
)
from aide.backend.utils import FunctionSpec


class TestHybridConfig:
    """Test the HybridConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = HybridConfig()
        assert config.code_backend == "claude_code"
        assert config.code_model == "claude-opus-4"
        assert config.analysis_backend is None
        assert config.analysis_model == "gpt-4o"
        assert config.default_backend is None
        assert config.default_model == "gpt-4o"
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = HybridConfig(
            code_backend="openai",
            code_model="gpt-4",
            analysis_backend="anthropic",
            analysis_model="claude-3",
            default_backend="gemini",
            default_model="gemini-pro"
        )
        assert config.code_backend == "openai"
        assert config.code_model == "gpt-4"
        assert config.analysis_backend == "anthropic"
        assert config.analysis_model == "claude-3"
        assert config.default_backend == "gemini"
        assert config.default_model == "gemini-pro"


class TestConfigManagement:
    """Test configuration management functions."""
    
    def test_set_and_get_config(self):
        """Test setting and getting hybrid configuration."""
        config = HybridConfig(code_backend="test_backend")
        set_hybrid_config(config)
        
        retrieved = get_hybrid_config()
        assert retrieved.code_backend == "test_backend"
    
    def test_configure_hybrid_backend(self):
        """Test the convenience configuration function."""
        configure_hybrid_backend(
            code_backend="custom_code",
            code_model="custom_model",
            analysis_backend="custom_analysis"
        )
        
        config = get_hybrid_config()
        assert config.code_backend == "custom_code"
        assert config.code_model == "custom_model"
        assert config.analysis_backend == "custom_analysis"


class TestTaskDetection:
    """Test task type detection."""
    
    def test_detect_code_generation(self):
        """Test detecting code generation tasks."""
        code_prompts = [
            "Write code to solve this problem",
            "Implement a function that calculates",
            "Create a Python script for data analysis",
            "Write a class to handle user authentication"
        ]
        
        for prompt in code_prompts:
            assert detect_task_type(prompt, None, None) == 'code_generation'
    
    def test_detect_analysis(self):
        """Test detecting analysis tasks."""
        analysis_prompts = [
            "Analyze the results of the execution",
            "Review the output and evaluate performance",
            "What is the accuracy of the model?",
            "Explain the findings from the experiment"
        ]
        
        for prompt in analysis_prompts:
            assert detect_task_type(prompt, None, None) == 'analysis'
    
    def test_detect_analysis_with_function(self):
        """Test detecting analysis with function spec."""
        func_spec = FunctionSpec(
            name="submit_review",
            json_schema={},
            description="Submit a review"
        )
        
        assert detect_task_type("Any message", None, func_spec) == 'analysis'
    
    def test_detect_general(self):
        """Test detecting general tasks."""
        general_prompts = [
            "Hello, how are you?",
            "What is the weather today?",
            "Tell me about machine learning"
        ]
        
        for prompt in general_prompts:
            assert detect_task_type(prompt, None, None) == 'general'


class TestHybridQuery:
    """Test the hybrid query function."""
    
    @patch('aide.backend.backend_hybrid.base_query')
    def test_query_code_generation(self, mock_base_query):
        """Test query routing for code generation."""
        # Setup
        mock_base_query.return_value = "generated code"
        configure_hybrid_backend(
            code_backend="claude_code",
            code_model="claude-opus-4"
        )
        
        # Execute
        result = query(
            system_message="Write code to implement a sorting algorithm",
            user_message=None,
            task_type="code_generation"
        )
        
        # Verify
        mock_base_query.assert_called_once()
        call_args = mock_base_query.call_args
        assert call_args.kwargs['backend'] == "claude_code"
        assert call_args.kwargs['model'] == "claude-opus-4"
        assert call_args.kwargs['task_type'] == "code_generation"
    
    @patch('aide.backend.backend_hybrid.base_query')
    def test_query_analysis(self, mock_base_query):
        """Test query routing for analysis."""
        # Setup
        mock_base_query.return_value = "analysis result"
        configure_hybrid_backend(
            analysis_backend="openai",
            analysis_model="gpt-4o"
        )
        
        # Execute
        result = query(
            system_message="Analyze the performance metrics",
            user_message=None,
            task_type="analysis"
        )
        
        # Verify
        mock_base_query.assert_called_once()
        call_args = mock_base_query.call_args
        assert call_args.kwargs['backend'] == "openai"
        assert call_args.kwargs['model'] == "gpt-4o"
        assert call_args.kwargs['task_type'] == "analysis"
    
    @patch('aide.backend.backend_hybrid.base_query')
    def test_query_auto_detect(self, mock_base_query):
        """Test query with automatic task detection."""
        # Setup
        mock_base_query.return_value = "result"
        configure_hybrid_backend()
        
        # Execute code generation
        query(
            system_message="Write a Python function to calculate factorial",
            user_message=None
        )
        
        # Verify
        call_args = mock_base_query.call_args
        assert call_args.kwargs['task_type'] == "code_generation"
        assert call_args.kwargs['backend'] == "claude_code"
    
    @patch('aide.backend.backend_hybrid.base_query')
    def test_query_with_function_spec(self, mock_base_query):
        """Test query with function spec (should be analysis)."""
        # Setup
        mock_base_query.return_value = {"is_bug": False, "summary": "All good"}
        configure_hybrid_backend(
            analysis_backend="anthropic",
            analysis_model="claude-3"
        )
        
        func_spec = FunctionSpec(
            name="submit_review",
            json_schema={},
            description="Submit review"
        )
        
        # Execute
        query(
            system_message="Review the code execution",
            user_message=None,
            func_spec=func_spec
        )
        
        # Verify
        call_args = mock_base_query.call_args
        assert call_args.kwargs['task_type'] == "analysis"
        assert call_args.kwargs['backend'] == "anthropic"
        assert call_args.kwargs['model'] == "claude-3"
    
    @patch('aide.backend.backend_hybrid.logger')
    @patch('aide.backend.backend_hybrid.base_query')
    def test_query_logging(self, mock_base_query, mock_logger):
        """Test that routing decisions are logged."""
        # Setup
        mock_base_query.return_value = "result"
        
        # Execute
        query(
            system_message="Write code",
            user_message=None
        )
        
        # Verify logging
        mock_logger.info.assert_called()
        log_message = mock_logger.info.call_args[0][0]
        assert "Routing code generation task" in log_message


if __name__ == "__main__":
    pytest.main([__file__])