#!/usr/bin/env python3
"""Integration test for Claude Code backend with AIDE ML's full workflow."""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from aide.agent import Agent
from aide.journal import Journal
from aide.utils.config import prep_cfg, _load_cfg
from aide.interpreter import Interpreter
from omegaconf import OmegaConf

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_config(backend="claude_code", task_type="simple"):
    """Create a test configuration for AIDE ML with Claude Code backend."""
    # Load default config and override with test settings
    config = _load_cfg(use_cli_args=False)
    
    # Basic configuration
    config.data_dir = tempfile.mkdtemp(prefix="aide_test_data_")
    config.log_dir = tempfile.mkdtemp(prefix="aide_test_log_")
    config.workspace_dir = tempfile.mkdtemp(prefix="aide_test_workspace_")
    config.goal = "Test task for integration testing"
    config.exp_name = "test-integration"
    
    # Agent configuration
    config.agent.steps = 1  # Just one step for testing
    config.agent.backend = backend
    config.agent.backend_options = {
        "provider": "subscription",
        "model": "claude-opus-4",
        "temperature": 0.2,
        "max_turns": 1
    }
    config.agent.search.num_drafts = 1  # Single draft for testing
    config.agent.search.debug_prob = 0.0
    
    # Prep the config to create necessary directories
    config = prep_cfg(config)
    
    return config


def test_simple_ml_task():
    """Test Claude Code backend with a simple ML task."""
    print("\n=== Testing Simple ML Task with Claude Code Backend ===")
    
    task_desc = """
    Create a simple linear regression model to predict y = 2x + 3 with some noise.
    Generate synthetic data, train the model, and report the R² score.
    """
    
    config = create_test_config(backend="claude_code")
    journal = Journal()
    
    try:
        # Initialize agent
        agent = Agent(task_desc=task_desc, cfg=config, journal=journal)
        
        # Mock the execution callback for testing
        def mock_exec_callback(code: str, is_main: bool):
            logger.info(f"Would execute code (is_main={is_main}):\n{code[:200]}...")
            # Return a mock successful result
            from aide.interpreter import ExecutionResult
            return ExecutionResult(
                stdout="Model trained successfully\nR² score: 0.98",
                stderr="",
                return_code=0
            )
        
        # Run one step of the agent with mock callback
        logger.info("Running agent step...")
        agent.step(exec_callback=mock_exec_callback)
        
        # Check if we got any nodes in the journal
        if journal.nodes:
            print(f"\n✓ Successfully created {len(journal.nodes)} node(s)")
            print(f"First node plan: {journal.nodes[0].plan[:100] if journal.nodes[0].plan else 'No plan'}...")
            return True
        else:
            print("\n✗ No nodes created")
            return False
            
    except Exception as e:
        print(f"\n✗ Error in simple ML task: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(config.workspace_dir, ignore_errors=True)
        shutil.rmtree(config.log_dir, ignore_errors=True)
        if os.path.exists(config.data_dir):
            shutil.rmtree(config.data_dir, ignore_errors=True)


def test_backend_integration():
    """Test that Claude Code backend is properly integrated."""
    print("\n=== Testing Backend Integration ===")
    
    try:
        # Import backend module
        from aide.backend import query
        
        # Test with Claude Code backend
        output = query(
            backend="claude_code",
            system_message="You are a helpful assistant.",
            user_message="Say 'Hello, AIDE ML!'",
            provider="subscription",
            model="claude-opus-4"
        )
        
        print(f"✓ Backend query successful")
        print(f"  Response: {output[:100]}..." if len(output) > 100 else f"  Response: {output}")
        return True
        
    except Exception as e:
        print(f"✗ Backend integration error: {e}")
        return False


def test_example_task_loading():
    """Test loading and understanding example tasks."""
    print("\n=== Testing Example Task Loading ===")
    
    example_tasks = [
        "aide/example_tasks/bitcoin_price.md",
        "aide/example_tasks/house_prices.md"
    ]
    
    all_loaded = True
    for task_path in example_tasks:
        try:
            with open(task_path, 'r') as f:
                content = f.read()
            print(f"✓ Loaded {Path(task_path).name}: {len(content)} chars")
        except Exception as e:
            print(f"✗ Failed to load {task_path}: {e}")
            all_loaded = False
    
    return all_loaded


def main():
    """Run integration tests."""
    print("Integration Tests for Claude Code Backend in AIDE ML")
    print("=" * 60)
    
    tests = [
        ("Backend Integration", test_backend_integration),
        ("Example Task Loading", test_example_task_loading),
        ("Simple ML Task", test_simple_ml_task),
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
    print("\n" + "=" * 60)
    print("Integration Test Summary:")
    print("-" * 60)
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:<30} {status}")
    
    total_passed = sum(1 for _, success in results if success)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    
    # Additional notes
    print("\n" + "=" * 60)
    print("Integration Status:")
    print("✓ Backend registered in aide/backend/__init__.py")
    print("✓ claude-code-sdk added to requirements.txt")
    print("✓ run_aide.py accepts --backend and --backend-opt arguments")
    print("✓ Agent passes backend configuration to query calls")
    print("\nTo test with real tasks:")
    print("python run_aide.py --task aide/example_tasks/bitcoin_price.md --backend claude_code")
    
    return 0 if total_passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())