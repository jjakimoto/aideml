#!/usr/bin/env python3
"""
Backend Performance Comparison Script
This script runs AIDE ML tasks with different backends and provides
a comprehensive performance comparison.
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
from typing import Dict, List, Any


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


def check_all_api_keys():
    """Check which backends have API keys configured"""
    available_backends = []
    
    # Check Anthropic
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY_TILDE"):
        available_backends.append("anthropic")
        print("✅ Anthropic backend available")
    else:
        print("❌ Anthropic API key not found")
    
    # Check Claude Code (assume it's available if SDK is installed)
    try:
        result = subprocess.run(["which", "claude"], capture_output=True)
        if result.returncode == 0:
            available_backends.append("claude_code")
            print("✅ Claude Code backend available")
        else:
            print("❌ Claude Code CLI not found")
    except:
        print("❌ Claude Code CLI not found")
    
    # Could add checks for other backends (OpenAI, Gemini, etc.)
    
    return available_backends


def run_task_with_backend(task: str, backend: str, project_root: Path, 
                         iterations: int = 5, timeout: int = 300) -> Dict[str, Any]:
    """Run a specific task with a specific backend"""
    print(f"\n🚀 Running {task} with {backend} backend...")
    
    # Prepare output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"comparison_outputs/{task}_{backend}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    task_path = project_root / "aide" / "example_tasks" / f"{task}.md"
    cmd = [
        "python", "-m", "aide.run",
        "--task", str(task_path),
        "--log-dir", str(output_dir),
        "--backend", backend,
        "--num-iterations", str(iterations),
        "--backend-opt", f"timeout={timeout}",
        "--backend-opt", "use_specialized_prompts=true"
    ]
    
    # Handle API key for Anthropic
    env = os.environ.copy()
    if backend == "anthropic" and not env.get("ANTHROPIC_API_KEY"):
        if env.get("ANTHROPIC_API_KEY_TILDE"):
            env["ANTHROPIC_API_KEY"] = env["ANTHROPIC_API_KEY_TILDE"]
    
    # Run the task
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout * iterations * 2  # Overall timeout
        )
        
        elapsed_time = time.time() - start_time
        
        # Extract metrics from output
        metrics = extract_metrics(output_dir, result.stdout)
        
        return {
            "success": result.returncode == 0,
            "elapsed_time": elapsed_time,
            "output_dir": str(output_dir),
            "metrics": metrics,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "elapsed_time": time.time() - start_time,
            "output_dir": str(output_dir),
            "error": "Timeout exceeded"
        }
    except Exception as e:
        return {
            "success": False,
            "elapsed_time": time.time() - start_time,
            "output_dir": str(output_dir),
            "error": str(e)
        }


def extract_metrics(output_dir: Path, stdout: str) -> Dict[str, Any]:
    """Extract performance metrics from output"""
    metrics = {}
    
    # Try to read journal file
    journal_path = output_dir / "journal.json"
    if journal_path.exists():
        try:
            with open(journal_path, 'r') as f:
                journal = json.load(f)
                if 'best_score' in journal:
                    metrics['best_score'] = journal['best_score']
                if 'iterations' in journal:
                    metrics['num_iterations'] = len(journal.get('iterations', []))
        except:
            pass
    
    # Extract token usage from stdout if available
    if "tokens" in stdout.lower():
        # Simple pattern matching for token counts
        import re
        token_pattern = r'(\d+)\s*tokens'
        matches = re.findall(token_pattern, stdout)
        if matches:
            metrics['total_tokens'] = sum(int(m) for m in matches)
    
    # Check if submission file exists (for house prices)
    submission_path = output_dir / "submission.csv"
    if submission_path.exists():
        metrics['submission_created'] = True
        try:
            df = pd.read_csv(submission_path)
            metrics['num_predictions'] = len(df)
        except:
            pass
    
    return metrics


def generate_comparison_report(results: Dict[str, Dict[str, Any]], output_path: Path):
    """Generate a comprehensive comparison report"""
    report = ["# AIDE ML Backend Comparison Report"]
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Summary table
    report.append("## Summary\n")
    report.append("| Task | Backend | Success | Time (s) | Best Score | Iterations |")
    report.append("|------|---------|---------|----------|------------|------------|")
    
    for task in results:
        for backend in results[task]:
            r = results[task][backend]
            success = "✅" if r['success'] else "❌"
            time_str = f"{r['elapsed_time']:.1f}"
            score = r['metrics'].get('best_score', 'N/A')
            if isinstance(score, float):
                score = f"{score:.4f}"
            iterations = r['metrics'].get('num_iterations', 'N/A')
            
            report.append(f"| {task} | {backend} | {success} | {time_str} | {score} | {iterations} |")
    
    # Detailed comparison by task
    report.append("\n## Detailed Comparison\n")
    
    for task in results:
        report.append(f"### {task.replace('_', ' ').title()}\n")
        
        backends = list(results[task].keys())
        if len(backends) >= 2:
            # Time comparison
            times = {b: results[task][b]['elapsed_time'] for b in backends if results[task][b]['success']}
            if times:
                fastest = min(times, key=times.get)
                report.append(f"**Fastest Backend**: {fastest} ({times[fastest]:.1f}s)\n")
            
            # Score comparison
            scores = {b: results[task][b]['metrics'].get('best_score') 
                     for b in backends 
                     if results[task][b]['success'] and results[task][b]['metrics'].get('best_score')}
            if scores:
                best = min(scores, key=scores.get)
                report.append(f"**Best Score**: {best} ({scores[best]:.4f})\n")
        
        # Individual backend details
        for backend in backends:
            r = results[task][backend]
            report.append(f"\n#### {backend}")
            if r['success']:
                report.append(f"- Time: {r['elapsed_time']:.1f}s")
                if 'best_score' in r['metrics']:
                    report.append(f"- Best Score: {r['metrics']['best_score']:.4f}")
                if 'total_tokens' in r['metrics']:
                    report.append(f"- Total Tokens: {r['metrics']['total_tokens']:,}")
                report.append(f"- Output: `{r['output_dir']}`")
            else:
                report.append(f"- Status: Failed")
                if 'error' in r:
                    report.append(f"- Error: {r['error']}")
    
    # Performance insights
    report.append("\n## Performance Insights\n")
    
    # Calculate overall statistics
    total_times = {}
    success_rates = {}
    
    for backend in ['claude_code', 'anthropic']:
        times = []
        successes = []
        for task in results:
            if backend in results[task]:
                r = results[task][backend]
                times.append(r['elapsed_time'])
                successes.append(r['success'])
        
        if times:
            total_times[backend] = sum(times)
            success_rates[backend] = sum(successes) / len(successes) * 100
    
    if total_times:
        report.append("### Overall Performance")
        for backend in total_times:
            report.append(f"- **{backend}**: {total_times[backend]:.1f}s total, "
                         f"{success_rates[backend]:.0f}% success rate")
    
    # Recommendations
    report.append("\n## Recommendations\n")
    report.append("Based on this comparison:\n")
    
    if 'claude_code' in total_times and 'anthropic' in total_times:
        if total_times['claude_code'] < total_times['anthropic']:
            report.append("- Claude Code backend is generally faster")
        else:
            report.append("- Anthropic backend is generally faster")
    
    report.append("- Consider using hybrid backend to leverage strengths of each")
    report.append("- Enable MCP for Claude Code backend for enhanced capabilities")
    report.append("- Use specialized prompts for better task-specific performance")
    
    # Write report
    report_content = "\n".join(report)
    with open(output_path, 'w') as f:
        f.write(report_content)
    
    # Also save raw results as JSON
    json_path = output_path.parent / f"{output_path.stem}_data.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return report_content


def main():
    parser = argparse.ArgumentParser(
        description="Compare AIDE ML backend performance across tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all available backends on all tasks
  python compare_backends.py
  
  # Compare specific backends
  python compare_backends.py --backends claude_code anthropic
  
  # Run specific task only
  python compare_backends.py --task bitcoin_price
  
  # Quick test with fewer iterations
  python compare_backends.py --iterations 3 --quick
        """
    )
    
    parser.add_argument(
        "--backends",
        nargs="+",
        help="Backends to compare (default: all available)"
    )
    
    parser.add_argument(
        "--task",
        choices=["bitcoin_price", "house_prices", "all"],
        default="all",
        help="Task to run (default: all)"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of iterations per task (default: 5)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per request in seconds (default: 300)"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with reduced iterations and timeout"
    )
    
    parser.add_argument(
        "--output",
        default="comparison_report.md",
        help="Output report filename (default: comparison_report.md)"
    )
    
    args = parser.parse_args()
    
    # Setup environment
    project_root = setup_environment()
    
    print("🔬 AIDE ML Backend Performance Comparison")
    print("="*50)
    
    # Check available backends
    available_backends = check_all_api_keys()
    if not available_backends:
        print("\n❌ No backends available. Please configure API keys.")
        sys.exit(1)
    
    # Determine which backends to use
    if args.backends:
        backends_to_test = [b for b in args.backends if b in available_backends]
        if not backends_to_test:
            print(f"\n❌ None of the specified backends are available: {args.backends}")
            sys.exit(1)
    else:
        backends_to_test = available_backends
    
    print(f"\n📊 Testing backends: {', '.join(backends_to_test)}")
    
    # Determine tasks
    if args.task == "all":
        tasks = ["bitcoin_price", "house_prices"]
    else:
        tasks = [args.task]
    
    # Quick mode adjustments
    if args.quick:
        args.iterations = min(args.iterations, 3)
        args.timeout = min(args.timeout, 180)
        print("⚡ Quick mode: reduced iterations and timeout")
    
    print(f"📋 Tasks: {', '.join(tasks)}")
    print(f"🔄 Iterations: {args.iterations}")
    print(f"⏱️  Timeout: {args.timeout}s per request")
    
    # Run comparisons
    results = {}
    total_start = time.time()
    
    for task in tasks:
        results[task] = {}
        print(f"\n\n{'='*50}")
        print(f"📈 Task: {task.replace('_', ' ').title()}")
        print(f"{'='*50}")
        
        for backend in backends_to_test:
            result = run_task_with_backend(
                task, backend, project_root, 
                args.iterations, args.timeout
            )
            results[task][backend] = result
            
            if result['success']:
                print(f"✅ {backend}: Completed in {result['elapsed_time']:.1f}s")
            else:
                print(f"❌ {backend}: Failed - {result.get('error', 'Unknown error')}")
    
    total_time = time.time() - total_start
    print(f"\n\n✅ All comparisons completed in {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    # Generate report
    output_path = Path(args.output)
    report = generate_comparison_report(results, output_path)
    
    print(f"\n📄 Report saved to: {output_path}")
    print(f"📊 Raw data saved to: {output_path.stem}_data.json")
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(report.split("## Performance Insights")[0])


if __name__ == "__main__":
    main()