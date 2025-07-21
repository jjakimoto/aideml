#!/usr/bin/env python3
"""
Bitcoin Price Prediction Example Runner
This script runs the AIDE ML agent on the Bitcoin price prediction task
with support for both Claude Code and Anthropic backends.

IMPORTANT: This script demonstrates the correct way to import from AIDE ML.
We add the parent of aideml to the Python path, then import from the 'aideml.aide' package.

Example of direct AIDE ML usage in your own scripts:
    import sys
    from pathlib import Path
    
    # Add parent of aideml to path
    aideml_parent = Path(__file__).parent.parent.parent  # Adjust based on location
    sys.path.insert(0, str(aideml_parent))
    
    # Import from aideml.aide package
    from aideml.aide.run import run_experiment
    from aideml.aide.backend import query_with_backend
    from aideml.aide.utils.config import load_config
"""

import os
import sys
import argparse
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime


def setup_environment():
    """Ensure environment is properly configured"""
    # Add parent of aideml to Python path (so we can import as aideml.aide)
    project_root = Path(__file__).parent.parent.parent  # aideml directory
    aideml_parent = project_root.parent  # parent of aideml
    sys.path.insert(0, str(aideml_parent))
    
    # Check if aideml conda environment is active
    if os.environ.get("CONDA_DEFAULT_ENV") != "aideml":
        print("⚠️  Warning: 'aideml' conda environment is not activated")
        print("   Run: conda activate aideml")
    
    return project_root


def check_api_keys(backend):
    """Check if required API keys are set"""
    if backend == "anthropic":
        # Check for Anthropic API key
        if not os.environ.get("ANTHROPIC_API_KEY"):
            # Try to load from ANTHROPIC_API_KEY_TILDE
            tilde_key = os.environ.get("ANTHROPIC_API_KEY_TILDE")
            if tilde_key:
                os.environ["ANTHROPIC_API_KEY"] = tilde_key
                print("✅ Using ANTHROPIC_API_KEY from ANTHROPIC_API_KEY_TILDE")
            else:
                print("❌ Error: ANTHROPIC_API_KEY not set")
                print("   Run: python ../scripts/setup_api_keys.py")
                return False
    elif backend == "claude_code":
        # Claude Code SDK should handle auth automatically
        print("✅ Using Claude Code backend (SDK handles authentication)")
    
    return True


def build_command(args, project_root, output_dir):
    """Build the command to run AIDE ML"""
    task_path = project_root / "aide" / "example_tasks" / "bitcoin_price.md"
    
    cmd = [
        "python", "-m", "aideml.aide.run",
        "--task", str(task_path),
        "--log-dir", str(output_dir),
        "--backend", args.backend
    ]
    
    # Add model if specified
    if args.model:
        cmd.extend(["--backend-opt", f"model={args.model}"])
    
    # Add MCP support for Claude Code
    if args.backend == "claude_code" and args.use_mcp:
        cmd.extend(["--backend-opt", "use_mcp=true"])
    
    # Add timeout
    cmd.extend(["--backend-opt", f"timeout={args.timeout}"])
    
    # Add number of workers
    cmd.extend(["--num-workers", str(args.workers)])
    
    # Add iterations
    cmd.extend(["--num-iterations", str(args.iterations)])
    
    # Add debug flag
    if args.debug:
        cmd.append("--debug")
    
    # Add specialized prompts (enabled by default)
    cmd.extend(["--backend-opt", "use_specialized_prompts=true"])
    
    return cmd


def run_aide_ml(args, project_root):
    """Run AIDE ML with the specified configuration"""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"bitcoin_price_{args.backend}_{timestamp}"
    output_dir = Path(args.output_dir) / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}")
    
    # Build command
    cmd = build_command(args, project_root, output_dir)
    
    print(f"\n🚀 Running AIDE ML with {args.backend} backend...")
    print(f"📋 Command: {' '.join(cmd)}")
    
    # Run the command
    start_time = time.time()
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode != 0:
            print(f"\n❌ AIDE ML exited with code {process.returncode}")
            return None
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        process.terminate()
        return None
    except Exception as e:
        print(f"\n❌ Error running AIDE ML: {e}")
        return None
    
    elapsed_time = time.time() - start_time
    print(f"\n✅ Completed in {elapsed_time:.1f} seconds")
    
    return {
        "backend": args.backend,
        "output_dir": str(output_dir),
        "elapsed_time": elapsed_time,
        "timestamp": timestamp
    }


def compare_backends(args, project_root):
    """Run the task with both backends and compare results"""
    print("🔄 Running comparison between Claude Code and Anthropic backends...")
    
    results = []
    
    # Run with Claude Code
    args_claude = argparse.Namespace(**vars(args))
    args_claude.backend = "claude_code"
    if check_api_keys("claude_code"):
        result = run_aide_ml(args_claude, project_root)
        if result:
            results.append(result)
    
    # Run with Anthropic
    args_anthropic = argparse.Namespace(**vars(args))
    args_anthropic.backend = "anthropic"
    if check_api_keys("anthropic"):
        result = run_aide_ml(args_anthropic, project_root)
        if result:
            results.append(result)
    
    if len(results) == 2:
        print("\n" + "="*60)
        print("📊 COMPARISON RESULTS")
        print("="*60)
        
        for result in results:
            print(f"\n{result['backend'].upper()}:")
            print(f"  Time: {result['elapsed_time']:.1f} seconds")
            print(f"  Output: {result['output_dir']}")
            
            # Try to read the best score from journal
            journal_path = Path(result['output_dir']) / 'journal.json'
            if journal_path.exists():
                try:
                    with open(journal_path, 'r') as f:
                        journal = json.load(f)
                        if 'best_score' in journal:
                            print(f"  Best Score: {journal['best_score']:.4f}")
                except:
                    pass
        
        # Performance comparison
        if results[0]['elapsed_time'] < results[1]['elapsed_time']:
            faster = results[0]['backend']
            speedup = (results[1]['elapsed_time'] / results[0]['elapsed_time'] - 1) * 100
            print(f"\n⚡ {faster} was {speedup:.1f}% faster")
        else:
            faster = results[1]['backend']
            speedup = (results[0]['elapsed_time'] / results[1]['elapsed_time'] - 1) * 100
            print(f"\n⚡ {faster} was {speedup:.1f}% faster")


def main():
    parser = argparse.ArgumentParser(
        description="Run AIDE ML on Bitcoin price prediction task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with Claude Code backend
  python run_bitcoin.py --backend claude_code
  
  # Run with Anthropic backend and specific model
  python run_bitcoin.py --backend anthropic --model claude-3-5-sonnet-20241022
  
  # Run with MCP enabled
  python run_bitcoin.py --backend claude_code --use-mcp
  
  # Compare both backends
  python run_bitcoin.py --compare
        """
    )
    
    parser.add_argument(
        "--backend",
        choices=["claude_code", "anthropic"],
        default="claude_code",
        help="Backend to use (default: claude_code)"
    )
    
    parser.add_argument(
        "--model",
        help="Model name to use (e.g., claude-opus-4, claude-3-5-sonnet-20241022)"
    )
    
    parser.add_argument(
        "--use-mcp",
        action="store_true",
        help="Enable MCP for Claude Code backend"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds (default: 300)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations (default: 10)"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run with both backends and compare results"
    )
    
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for outputs (default: outputs/)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup environment
    project_root = setup_environment()
    
    print("🚀 Bitcoin Price Prediction with AIDE ML")
    print("="*50)
    
    if args.compare:
        # Run comparison
        compare_backends(args, project_root)
    else:
        # Check API keys
        if not check_api_keys(args.backend):
            sys.exit(1)
        
        # Run single backend
        run_aide_ml(args, project_root)
    
    print("\n✨ Done! Check the output directory for results.")


if __name__ == "__main__":
    main()