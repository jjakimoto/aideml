"""Tests for advanced MCP features in Claude Code backend."""

import pytest
import json
import tempfile
from pathlib import Path
from aide.backend.backend_claude_code import _create_mcp_config_for_func_spec
from aide.backend.utils import FunctionSpec


class TestAdvancedMCP:
    """Test advanced MCP functionality."""
    
    def test_basic_mcp_config(self):
        """Test basic MCP configuration generation."""
        func_spec = FunctionSpec(
            name="test_func",
            description="A test function",
            json_schema={
                "type": "object",
                "properties": {
                    "param1": {"type": "string"}
                }
            }
        )
        
        config = _create_mcp_config_for_func_spec(func_spec)
        
        assert "servers" in config
        assert "aide" in config["servers"]
        assert config["servers"]["aide"]["command"] == "python"
        assert "-m" in config["servers"]["aide"]["args"]
        assert "aide.backend.mcp_server" in config["servers"]["aide"]["args"]
        assert "--mode" in config["servers"]["aide"]["args"]
        assert "stdio" in config["servers"]["aide"]["args"]
    
    def test_advanced_mcp_config_stdio(self):
        """Test advanced MCP configuration with stdio mode."""
        func_spec = FunctionSpec(
            name="analyze_data",
            description="Analyze data with statistics",
            json_schema={
                "type": "object",
                "properties": {
                    "data": {"type": "array"},
                    "method": {"type": "string"}
                }
            }
        )
        
        config = _create_mcp_config_for_func_spec(
            func_spec,
            use_advanced=True,
            http_mode=False
        )
        
        assert "servers" in config
        assert "aide" in config["servers"]
        assert "aide.backend.mcp_server_advanced" in config["servers"]["aide"]["args"]
        assert "--mode" in config["servers"]["aide"]["args"]
        assert "stdio" in config["servers"]["aide"]["args"]
        assert "--verbose" in config["servers"]["aide"]["args"]
        
        # Check metadata for advanced mode
        assert "metadata" in config["servers"]["aide"]
        assert config["servers"]["aide"]["metadata"]["version"] == "2.0.0"
        assert "validation" in config["servers"]["aide"]["metadata"]["capabilities"]
        assert config["servers"]["aide"]["metadata"]["mode"] == "stdio"
    
    def test_advanced_mcp_config_http(self):
        """Test advanced MCP configuration with HTTP mode."""
        func_spec = FunctionSpec(
            name="process_text",
            description="Process text data",
            json_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "operation": {"type": "string"}
                }
            }
        )
        
        config = _create_mcp_config_for_func_spec(
            func_spec,
            use_advanced=True,
            http_mode=True,
            port=9090
        )
        
        assert "servers" in config
        assert "aide" in config["servers"]
        assert "aide.backend.mcp_server_advanced" in config["servers"]["aide"]["args"]
        assert "--mode" in config["servers"]["aide"]["args"]
        assert "http" in config["servers"]["aide"]["args"]
        assert "--port" in config["servers"]["aide"]["args"]
        assert "9090" in config["servers"]["aide"]["args"]
        assert "--verbose" in config["servers"]["aide"]["args"]
        
        # Check metadata
        assert config["servers"]["aide"]["metadata"]["mode"] == "http"
        assert "async" in config["servers"]["aide"]["metadata"]["capabilities"]
        assert "stats" in config["servers"]["aide"]["metadata"]["capabilities"]
    
    def test_mcp_config_file_creation(self):
        """Test MCP configuration file creation."""
        func_spec = FunctionSpec(
            name="save_test",
            description="Test saving config",
            json_schema={"type": "object"}
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mcp_config.json"
            
            config = _create_mcp_config_for_func_spec(
                func_spec,
                config_path=str(config_path),
                use_advanced=True
            )
            
            # Check file was created
            assert config_path.exists()
            
            # Load and verify contents
            with open(config_path) as f:
                saved_config = json.load(f)
            
            assert saved_config == config
            assert "metadata" in saved_config["servers"]["aide"]
    
    def test_mcp_tool_naming(self):
        """Test MCP tool naming convention."""
        func_spec = FunctionSpec(
            name="my_function",
            description="Test function",
            json_schema={"type": "object"}
        )
        
        config = _create_mcp_config_for_func_spec(func_spec)
        
        tools = config["servers"]["aide"]["tools"]
        assert f"call_{func_spec.name}" in tools
        assert tools[f"call_{func_spec.name}"]["description"] == func_spec.description
        assert tools[f"call_{func_spec.name}"]["inputSchema"] == func_spec.json_schema
    
    def test_advanced_features_info(self):
        """Test that advanced features are properly configured."""
        func_spec = FunctionSpec(
            name="advanced_test",
            description="Test advanced features",
            json_schema={
                "type": "object",
                "properties": {
                    "required_param": {"type": "string"}
                },
                "required": ["required_param"]
            }
        )
        
        # Test with different configurations
        configs = [
            (False, False, None),  # Basic stdio
            (True, False, None),   # Advanced stdio
            (True, True, 8888),    # Advanced HTTP
        ]
        
        for use_advanced, http_mode, port in configs:
            config = _create_mcp_config_for_func_spec(
                func_spec,
                use_advanced=use_advanced,
                http_mode=http_mode,
                port=port if port else 8080
            )
            
            if use_advanced:
                assert "aide.backend.mcp_server_advanced" in config["servers"]["aide"]["args"]
                assert "metadata" in config["servers"]["aide"]
                assert "middleware" in config["servers"]["aide"]["metadata"]["capabilities"]
                
                if http_mode:
                    assert "http" in config["servers"]["aide"]["args"]
                    assert str(port) in config["servers"]["aide"]["args"]
            else:
                assert "aide.backend.mcp_server" in config["servers"]["aide"]["args"]
                assert "metadata" not in config["servers"]["aide"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])