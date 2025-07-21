#!/usr/bin/env python3
"""
House Price Prediction Example Runner
This script runs the AIDE ML agent on the house price prediction task
with support for both Claude Code and Anthropic backends.
"""

import os
import sys
import argparse
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import pandas as pd


def setup_environment():
    """Ensure environment is properly configured"""
    # Add project root to Python path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
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


def add_custom_prompt(args):
    """Add custom prompts based on feature engineering level"""
    prompts = []
    
    if args.feature_eng == "advanced":
        prompts.append(
            "Focus on advanced feature engineering including polynomial features, "
            "interaction terms between key variables (especially area and quality features), "
            "and domain-specific transformations for real estate data."
        )
    elif args.feature_eng == "auto":
        prompts.append(
            "Use automated feature engineering techniques like featuretools or "
            "category_encoders with target encoding. Explore feature selection methods."
        )
    
    if args.custom_prompt:
        prompts.append(args.custom_prompt)
    
    return " ".join(prompts) if prompts else None


def build_command(args, project_root, output_dir):
    """Build the command to run AIDE ML"""
    task_path = project_root / "aide" / "example_tasks" / "house_prices.md"
    
    cmd = [
        "python", "-m", "aide.run",
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
    
    # Add timeout (higher for house prices due to complexity)
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
    
    # Add custom prompt if provided
    custom_prompt = add_custom_prompt(args)
    if custom_prompt:
        cmd.extend(["--prompt-addon", custom_prompt])
    
    return cmd


def analyze_results(output_dir):
    """Analyze the results and extract key metrics"""
    metrics = {}
    
    # Try to read submission file
    submission_path = output_dir / "submission.csv"
    if submission_path.exists():
        try:
            submission = pd.read_csv(submission_path)
            metrics['num_predictions'] = len(submission)
            metrics['submission_exists'] = True
            
            # Check for reasonable price ranges
            if 'SalePrice' in submission.columns:
                prices = submission['SalePrice']
                metrics['price_min'] = prices.min()
                metrics['price_max'] = prices.max()
                metrics['price_mean'] = prices.mean()
        except Exception as e:
            print(f"⚠️  Could not analyze submission: {e}")
    
    # Try to read journal for best score
    journal_path = output_dir / "journal.json"
    if journal_path.exists():
        try:
            with open(journal_path, 'r') as f:
                journal = json.load(f)
                if 'best_score' in journal:
                    metrics['best_score'] = journal['best_score']
                
                # Count number of iterations
                if 'iterations' in journal:
                    metrics['iterations_completed'] = len(journal['iterations'])
        except:
            pass
    
    return metrics


def run_aide_ml(args, project_root):
    """Run AIDE ML with the specified configuration"""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"house_prices_{args.backend}_{timestamp}"
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
    print(f"\n✅ Completed in {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    
    # Analyze results
    metrics = analyze_results(output_dir)
    
    return {
        "backend": args.backend,
        "output_dir": str(output_dir),
        "elapsed_time": elapsed_time,
        "timestamp": timestamp,
        "metrics": metrics
    }


def compare_backends(args, project_root):
    """Run the task with both backends and compare results"""
    print("🔄 Running comparison between Claude Code and Anthropic backends...")
    print("   Note: House prices task is complex and may take 1-2 hours total\n")
    
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
            print(f"  Time: {result['elapsed_time']:.1f} seconds ({result['elapsed_time']/60:.1f} minutes)")
            print(f"  Output: {result['output_dir']}")
            
            # Print metrics
            metrics = result.get('metrics', {})
            if 'best_score' in metrics:
                print(f"  Best Score (RMSE): {metrics['best_score']:.4f}")
            if 'iterations_completed' in metrics:
                print(f"  Iterations: {metrics['iterations_completed']}")
            if 'num_predictions' in metrics:
                print(f"  Predictions: {metrics['num_predictions']}")
        
        # Performance comparison
        if results[0]['elapsed_time'] < results[1]['elapsed_time']:
            faster = results[0]['backend']
            speedup = (results[1]['elapsed_time'] / results[0]['elapsed_time'] - 1) * 100
            print(f"\n⚡ {faster} was {speedup:.1f}% faster")
        else:
            faster = results[1]['backend']
            speedup = (results[0]['elapsed_time'] / results[1]['elapsed_time'] - 1) * 100
            print(f"\n⚡ {faster} was {speedup:.1f}% faster")
        
        # Score comparison
        score1 = results[0].get('metrics', {}).get('best_score')
        score2 = results[1].get('metrics', {}).get('best_score')
        if score1 and score2:
            if score1 < score2:
                print(f"🏆 {results[0]['backend']} achieved better score ({score1:.4f} vs {score2:.4f})")
            else:
                print(f"🏆 {results[1]['backend']} achieved better score ({score2:.4f} vs {score1:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="Run AIDE ML on house price prediction task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with Claude Code backend
  python run_house_prices.py --backend claude_code
  
  # Run with advanced feature engineering
  python run_house_prices.py --backend claude_code --feature-eng advanced
  
  # Run with Anthropic backend and more iterations
  python run_house_prices.py --backend anthropic --iterations 20
  
  # Compare both backends
  python run_house_prices.py --compare
  
  # Custom instructions
  python run_house_prices.py --backend claude_code \\
    --custom-prompt "Focus on tree-based models with careful hyperparameter tuning"
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
        default=400,
        help="Request timeout in seconds (default: 400)"
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
        default=15,
        help="Number of iterations (default: 15)"
    )
    
    parser.add_argument(
        "--feature-eng",
        choices=["basic", "advanced", "auto"],
        default="basic",
        help="Level of feature engineering to suggest"
    )
    
    parser.add_argument(
        "--custom-prompt",
        help="Additional instructions for AIDE ML"
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
    
    print("🏠 House Price Prediction with AIDE ML")
    print("="*50)
    print("📊 This task has 79 features and requires sophisticated ML techniques")
    print("⏱️  Expected runtime: 3-8 minutes per iteration")
    
    if args.compare:
        # Run comparison
        compare_backends(args, project_root)
    else:
        # Check API keys
        if not check_api_keys(args.backend):
            sys.exit(1)
        
        # Run single backend
        result = run_aide_ml(args, project_root)
        
        if result and result.get('metrics'):
            print("\n📊 Results Summary:")
            metrics = result['metrics']
            if 'best_score' in metrics:
                print(f"   Best RMSE: {metrics['best_score']:.4f}")
            if 'num_predictions' in metrics:
                print(f"   Predictions generated: {metrics['num_predictions']}")
    
    print("\n✨ Done! Check the output directory for:")
    print("   - solution.py: The complete ML pipeline")
    print("   - submission.csv: Predictions for the test set")
    print("   - journal.json: Detailed reasoning log")
    print("   - tree.html: Interactive solution exploration")


if __name__ == "__main__":
    main()