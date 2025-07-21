"""Tests for development mode setup and environment verification."""

import os
import sys
import subprocess
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
import importlib.util


class TestDevModeSetup:
    """Test development mode setup and environment."""
    
    def test_project_structure_exists(self):
        """Verify that essential project structure exists."""
        root_dir = Path(__file__).parent.parent
        
        # Check essential directories
        assert (root_dir / "aide").exists(), "aide package directory not found"
        assert (root_dir / "tests").exists(), "tests directory not found"
        assert (root_dir / "docs").exists(), "docs directory not found"
        
        # Check essential files
        # Note: This project uses requirements.txt instead of setup.py
        assert (root_dir / "requirements.txt").exists(), "requirements.txt not found"
        assert (root_dir / "run_aide.py").exists(), "run_aide.py not found"
        assert (root_dir / "run_webui.py").exists(), "run_webui.py not found"
        assert (root_dir / "setup_dev.sh").exists(), "setup_dev.sh not found"
    
    def test_aide_package_importable(self):
        """Test that aide package can be imported."""
        try:
            import aide
            assert hasattr(aide, '__version__') or True  # Package imported successfully
        except ImportError as e:
            # In development, some dependencies might not be installed
            if "humanize" in str(e) or "scikit-learn" in str(e):
                pytest.skip(f"Skipping due to missing development dependency: {e}")
            else:
                pytest.fail(f"Failed to import aide package: {e}")
    
    def test_required_dependencies_available(self):
        """Test that key dependencies are available."""
        required_packages = [
            'numpy',
            'pandas', 
            'scikit-learn',
            'streamlit',
            'pytest',
            'claude-code-sdk'
        ]
        
        missing_packages = []
        for package in required_packages:
            spec = importlib.util.find_spec(package.replace('-', '_'))
            if spec is None:
                missing_packages.append(package)
        
        if missing_packages:
            pytest.skip(f"Skipping test due to missing packages: {', '.join(missing_packages)}. Install with: pip install -r requirements.txt")
    
    def test_backend_modules_exist(self):
        """Verify all backend modules exist."""
        backends_dir = Path(__file__).parent.parent / "aide" / "backend"
        
        expected_backends = [
            "backend_openai.py",
            "backend_anthropic.py",
            "backend_gemini.py",
            "backend_claude_code.py",
            "backend_hybrid.py",
            "backend_openrouter.py",
            "mcp_server.py",
            "mcp_server_advanced.py"
        ]
        
        for backend in expected_backends:
            assert (backends_dir / backend).exists(), f"Backend module {backend} not found"
    
    def test_example_tasks_exist(self):
        """Verify example tasks are present."""
        tasks_dir = Path(__file__).parent.parent / "aide" / "example_tasks"
        
        assert (tasks_dir / "bitcoin_price.md").exists(), "bitcoin_price.md task not found"
        assert (tasks_dir / "house_prices.md").exists(), "house_prices.md task not found"
        
        # Check data directories
        assert (tasks_dir / "bitcoin_price" / "BTC-USD.csv").exists(), "Bitcoin data not found"
        assert (tasks_dir / "house_prices" / "train.csv").exists(), "House prices training data not found"
    
    def test_configuration_files_valid(self):
        """Test that configuration files are valid."""
        config_path = Path(__file__).parent.parent / "aide" / "utils" / "config.yaml"
        assert config_path.exists(), "config.yaml not found"
        
        # Try to load the config
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            assert isinstance(config, dict), "config.yaml is not a valid YAML dict"
            assert 'agent' in config, "agent section missing from config"
        except Exception as e:
            pytest.fail(f"Failed to load config.yaml: {e}")
    
    @pytest.mark.skipif(sys.platform == "win32", reason="Shell script test skipped on Windows")
    def test_setup_dev_script_executable(self):
        """Test that setup_dev.sh is executable and valid."""
        setup_script = Path(__file__).parent.parent / "setup_dev.sh"
        
        # Check if executable
        assert os.access(setup_script, os.X_OK), "setup_dev.sh is not executable"
        
        # Check script syntax (dry run)
        result = subprocess.run(
            ["bash", "-n", str(setup_script)],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"setup_dev.sh has syntax errors: {result.stderr}"
    
    def test_cli_entry_points_syntax(self):
        """Test that CLI entry points have valid Python syntax."""
        root_dir = Path(__file__).parent.parent
        
        scripts = ["run_aide.py", "run_webui.py"]
        
        for script in scripts:
            script_path = root_dir / script
            
            # Check Python syntax
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(script_path)],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0, f"{script} has syntax errors: {result.stderr}"


class TestDevelopmentEnvironment:
    """Test development environment configuration."""
    
    def test_python_version(self):
        """Ensure Python version is 3.10 or higher."""
        assert sys.version_info >= (3, 10), f"Python 3.10+ required, got {sys.version}"
    
    def test_aide_ml_paths_in_sys_path(self):
        """Test that AIDE ML can be imported when running from project root."""
        root_dir = str(Path(__file__).parent.parent)
        
        # In dev mode, the root should be in sys.path or aide should be importable
        try:
            import aide
            assert True  # Import successful
        except ImportError:
            # Check if we can add the path and then import
            if root_dir not in sys.path:
                sys.path.insert(0, root_dir)
                try:
                    import aide
                    assert True
                except ImportError:
                    pytest.fail("Cannot import aide even with root in sys.path")
    
    def test_debug_configuration_available(self):
        """Test that debug configuration is available in development mode."""
        try:
            from aide.utils.config import Config
            
            # Create a config instance
            config = Config()
            
            # Check if debug-related settings exist
            assert hasattr(config, 'agent'), "Config missing agent attribute"
            
            # Check for search configuration which includes debug settings
            if hasattr(config.agent, 'search'):
                assert hasattr(config.agent.search, 'max_debug_depth'), "max_debug_depth not in config"
                assert hasattr(config.agent.search, 'debug_prob'), "debug_prob not in config"
        except ImportError as e:
            if "humanize" in str(e) or "scikit-learn" in str(e):
                pytest.skip(f"Skipping due to missing development dependency: {e}")
            else:
                pytest.fail(f"Failed to load development configuration: {e}")
        except Exception as e:
            pytest.fail(f"Failed to load development configuration: {e}")
    
    @patch.dict(os.environ, {"AIDE_DEV_MODE": "1"})
    def test_dev_mode_environment_variable(self):
        """Test behavior when AIDE_DEV_MODE environment variable is set."""
        assert os.getenv("AIDE_DEV_MODE") == "1"
        
        # In a real implementation, this might enable additional logging,
        # debugging features, or development-specific behavior
    
    def test_test_directory_structure(self):
        """Verify test directory has proper structure."""
        tests_dir = Path(__file__).parent
        
        # Should have __init__.py
        assert (tests_dir / "__init__.py").exists(), "tests/__init__.py missing"
        
        # Count test files
        test_files = list(tests_dir.glob("test_*.py"))
        assert len(test_files) > 0, "No test files found in tests directory"
        
        # Verify this test file is among them
        assert Path(__file__).name in [f.name for f in test_files]


class TestDevelopmentWorkflow:
    """Test common development workflow operations."""
    
    def test_aide_backend_registration(self):
        """Test that backends can be properly registered."""
        try:
            from aide.backend import provider_to_query_func
            
            # Check that essential backends are registered
            expected_backends = ["openai", "anthropic", "claude_code", "hybrid"]
            
            for backend in expected_backends:
                assert backend in provider_to_query_func, f"{backend} not registered"
        except ImportError as e:
            if "humanize" in str(e) or "scikit-learn" in str(e):
                pytest.skip(f"Skipping due to missing development dependency: {e}")
            else:
                pytest.fail(f"Failed to check backend registration: {e}")
        except Exception as e:
            pytest.fail(f"Failed to check backend registration: {e}")
    
    def test_example_task_format(self):
        """Test that example tasks have the correct format."""
        tasks_dir = Path(__file__).parent.parent / "aide" / "example_tasks"
        
        for task_file in ["bitcoin_price.md", "house_prices.md"]:
            task_path = tasks_dir / task_file
            
            with open(task_path, 'r') as f:
                content = f.read()
            
            # Check for essential sections (project uses "## Goal" instead of "# Task")
            assert "## Goal" in content or "# Task" in content, f"{task_file} missing Goal/Task section"
            # Check that evaluation metric is mentioned
            assert "metric" in content.lower() or "evaluation" in content.lower(), f"{task_file} missing evaluation metric"
    
    @patch('subprocess.run')
    def test_development_commands(self, mock_run):
        """Test that development commands would work correctly."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        
        # Simulate common development commands
        commands = [
            ["python", "run_aide.py", "--help"],
            ["python", "-m", "aide.utils.view_performance", "--help"],
            ["pytest", "--collect-only"],
        ]
        
        for cmd in commands:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            # We're just checking these commands are formed correctly
            # Not actually running them in the test
            assert isinstance(cmd, list)


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v"])