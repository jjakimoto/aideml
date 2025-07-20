"""Tests for async support in Claude Code backend."""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from aide.backend.backend_claude_code import query_async, query
from aide.backend.utils import FunctionSpec


class TestAsyncSupport:
    """Test async functionality in Claude Code backend."""
    
    @pytest.mark.asyncio
    async def test_query_async_basic(self):
        """Test basic async query functionality."""
        # Mock the claude_query function
        mock_message = MagicMock()
        mock_message.content = "Test response"
        
        async def mock_claude_query(*args, **kwargs):
            yield mock_message
        
        with patch('aide.backend.backend_claude_code.claude_query', mock_claude_query):
            with patch('aide.backend.backend_claude_code._setup_claude_code_auth'):
                result = await query_async(
                    system_message="You are a helpful assistant",
                    user_message="Hello",
                    model="claude-opus-4"
                )
                
                output, req_time, in_tokens, out_tokens, info = result
                
                assert output == "Test response"
                assert req_time > 0
                assert in_tokens > 0
                assert out_tokens > 0
                assert info["async"] is True
                assert info["model"] == "claude-opus-4"
    
    @pytest.mark.asyncio
    async def test_query_async_with_function(self):
        """Test async query with function calling."""
        func_spec = FunctionSpec(
            name="test_function",
            description="A test function",
            json_schema={
                "type": "object",
                "properties": {
                    "arg1": {"type": "string"}
                }
            }
        )
        
        mock_message = MagicMock()
        mock_message.content = '{"arg1": "test_value"}'
        
        async def mock_claude_query(*args, **kwargs):
            yield mock_message
        
        with patch('aide.backend.backend_claude_code.claude_query', mock_claude_query):
            with patch('aide.backend.backend_claude_code._setup_claude_code_auth'):
                result = await query_async(
                    system_message="System prompt",
                    user_message="User prompt",
                    func_spec=func_spec,
                    model="claude-opus-4"
                )
                
                output, req_time, in_tokens, out_tokens, info = result
                
                assert isinstance(output, dict)
                assert output.get("arg1") == "test_value"
                assert info["async"] is True
    
    def test_sync_query_event_loop_handling(self):
        """Test that sync query handles event loops properly."""
        mock_message = MagicMock()
        mock_message.content = "Sync test response"
        
        async def mock_claude_query(*args, **kwargs):
            yield mock_message
        
        with patch('aide.backend.backend_claude_code.claude_query', mock_claude_query):
            with patch('aide.backend.backend_claude_code._setup_claude_code_auth'):
                # Test without existing event loop (normal case)
                result = query(
                    system_message="System",
                    user_message="User",
                    model="claude-opus-4"
                )
                
                output, req_time, in_tokens, out_tokens, info = result
                
                assert output == "Sync test response"
                assert req_time > 0
                assert "async" not in info  # Sync version doesn't set async flag
    
    @pytest.mark.asyncio
    async def test_async_mcp_cleanup(self):
        """Test that temporary MCP configs are cleaned up in async version."""
        import tempfile
        import os
        
        func_spec = FunctionSpec(
            name="test_mcp",
            description="Test MCP function",
            json_schema={"type": "object"}
        )
        
        mock_message = MagicMock()
        mock_message.content = "{}"
        
        async def mock_claude_query(*args, **kwargs):
            yield mock_message
        
        temp_files_created = []
        original_tempfile = tempfile.NamedTemporaryFile
        
        def track_tempfile(*args, **kwargs):
            f = original_tempfile(*args, **kwargs)
            temp_files_created.append(f.name)
            return f
        
        with patch('aide.backend.backend_claude_code.claude_query', mock_claude_query):
            with patch('aide.backend.backend_claude_code._setup_claude_code_auth'):
                with patch('aide.backend.backend_claude_code._create_mcp_config_for_func_spec'):
                    with patch('tempfile.NamedTemporaryFile', track_tempfile):
                        result = await query_async(
                            system_message="System",
                            user_message="User",
                            func_spec=func_spec,
                            use_mcp=True,
                            model="claude-opus-4"
                        )
                        
                        # Check that temp files were created
                        assert len(temp_files_created) > 0
                        
                        # Check that temp files were cleaned up
                        for temp_file in temp_files_created:
                            assert not os.path.exists(temp_file)
    
    @pytest.mark.asyncio
    async def test_concurrent_async_queries(self):
        """Test multiple concurrent async queries."""
        async def mock_claude_query(*args, **kwargs):
            # Simulate some processing time
            await asyncio.sleep(0.01)
            mock_message = MagicMock()
            mock_message.content = f"Response for prompt: {kwargs.get('prompt', '')[:20]}"
            yield mock_message
        
        with patch('aide.backend.backend_claude_code.claude_query', mock_claude_query):
            with patch('aide.backend.backend_claude_code._setup_claude_code_auth'):
                # Run multiple queries concurrently
                tasks = []
                for i in range(5):
                    task = query_async(
                        system_message=f"System {i}",
                        user_message=f"User message {i}",
                        model="claude-opus-4"
                    )
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks)
                
                # Verify all queries completed
                assert len(results) == 5
                
                # Verify each result is unique
                outputs = [result[0] for result in results]
                assert len(set(outputs)) == 5  # All outputs should be unique


if __name__ == "__main__":
    pytest.main([__file__, "-v"])