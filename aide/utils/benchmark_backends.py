"""Systematic performance benchmarking for LLM backends.

This module provides functionality to systematically benchmark different LLM
backends by running the same tasks across multiple providers and comparing
their performance, accuracy, and cost-effectiveness.
"""

import time
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import concurrent.futures

from aide.utils.performance_monitor import get_performance_monitor
from aide.utils.config import Config


class BackendBenchmark:
    """Systematic benchmarking of LLM backends."""
    
    def __init__(self, tasks_dir: Path = None, output_dir: Path = None):
        """Initialize benchmarking system.
        
        Args:
            tasks_dir: Directory containing benchmark tasks.
            output_dir: Directory to save benchmark results.
        """
        self.tasks_dir = tasks_dir or Path(__file__).parent.parent / "example_tasks"
        self.output_dir = output_dir or Path.home() / '.aide_ml' / 'benchmarks'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.performance_monitor = get_performance_monitor()
        
        # Define available backends
        self.available_backends = {'openai', 'anthropic', 'openrouter', 'gemini', 'claude_code', 'hybrid'}
        
        # Define benchmark configurations
        self.backend_configs = {
            'openai': {
                'backend': 'openai',
                'model': 'gpt-4-turbo-preview',
                'temperature': 0.7
            },
            'anthropic': {
                'backend': 'anthropic', 
                'model': 'claude-3-sonnet-20240229',
                'temperature': 0.7
            },
            'claude_code': {
                'backend': 'claude_code',
                'model': 'claude-opus-4',
                'temperature': 0.2,
                'provider': 'subscription'
            },
            'gemini': {
                'backend': 'gemini',
                'model': 'gemini-pro',
                'temperature': 0.7
            },
            'hybrid': {
                'backend': 'hybrid',
                'agent.hybrid.code_backend': 'claude_code',
                'agent.hybrid.code_model': 'claude-opus-4',
                'agent.hybrid.analysis_backend': 'openai',
                'agent.hybrid.analysis_model': 'gpt-4-turbo-preview'
            }
        }
    
    def get_benchmark_tasks(self) -> List[Path]:
        """Get list of benchmark tasks."""
        tasks = []
        if self.tasks_dir.exists():
            # Use existing example tasks
            tasks.extend(self.tasks_dir.glob("*.md"))
        
        # Add some standard benchmark tasks if not enough
        if len(tasks) < 2:
            # Create minimal benchmark tasks
            self._create_minimal_benchmark_tasks()
            tasks = list(self.tasks_dir.glob("*.md"))
        
        return tasks[:5]  # Limit to 5 tasks for benchmarking
    
    def _create_minimal_benchmark_tasks(self):
        """Create minimal benchmark tasks if they don't exist."""
        # This is a fallback - the example tasks should already exist
        pass
    
    def _validate_backends(self, backends: List[str]) -> List[str]:
        """Validate that backends are available and configured.
        
        Args:
            backends: List of backend names to validate.
            
        Returns:
            List of valid backend names.
            
        Raises:
            ValueError: If no valid backends are found.
        """
        valid_backends = []
        invalid_backends = []
        
        for backend in backends:
            if backend in self.available_backends:
                if backend in self.backend_configs:
                    valid_backends.append(backend)
                else:
                    invalid_backends.append(f"{backend} (not configured)")
            else:
                invalid_backends.append(f"{backend} (not available)")
        
        if invalid_backends:
            print(f"Warning: Skipping invalid backends: {', '.join(invalid_backends)}")
        
        if not valid_backends:
            available = ', '.join(sorted(self.available_backends))
            raise ValueError(f"No valid backends found. Available backends: {available}")
        
        return valid_backends
    
    def run_benchmark(self, backends: List[str] = None, tasks: List[Path] = None,
                     max_workers: int = 1) -> Dict[str, Any]:
        """Run systematic benchmark across backends and tasks.
        
        Args:
            backends: List of backend names to benchmark. If None, uses all.
            tasks: List of task files to run. If None, uses default benchmark tasks.
            max_workers: Maximum number of parallel workers.
            
        Returns:
            Dictionary containing benchmark results.
        """
        backends = backends or list(self.backend_configs.keys())
        backends = self._validate_backends(backends)  # Validate backends before use
        tasks = tasks or self.get_benchmark_tasks()
        
        print(f"Starting benchmark with {len(backends)} backends and {len(tasks)} tasks...")
        
        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'backends': backends,
                'tasks': [str(t) for t in tasks],
                'num_tasks': len(tasks),
                'num_backends': len(backends)
            },
            'results': {},
            'summary': {}
        }
        
        # Run benchmarks
        for backend in backends:
            print(f"\n--- Benchmarking {backend} ---")
            backend_results = []
            
            for task in tasks:
                print(f"  Running task: {task.name}")
                result = self._run_single_benchmark(backend, task)
                backend_results.append(result)
                
                # Save intermediate results
                self._save_results(results)
            
            results['results'][backend] = backend_results
        
        # Generate summary
        results['summary'] = self._generate_summary(results['results'])
        
        # Save final results
        self._save_results(results)
        self._generate_report(results)
        
        return results
    
    def _run_single_benchmark(self, backend: str, task: Path) -> Dict[str, Any]:
        """Run a single benchmark task with a specific backend.
        
        Args:
            backend: Backend name.
            task: Task file path.
            
        Returns:
            Dictionary containing benchmark results for this run.
        """
        # Validate inputs
        if not task.exists():
            return {
                'backend': backend,
                'task': task.name,
                'start_time': time.time(),
                'end_time': time.time(),
                'duration': 0,
                'success': False,
                'error': f"Task file not found: {task}",
                'metrics': {}
            }
        
        config = self.backend_configs.get(backend, {})
        start_time = time.time()
        
        # Build command with input validation
        cmd = ['python', 'run_aide.py', '--task', str(task)]
        
        # Add backend configuration with validation
        if 'backend' in config:
            backend_value = str(config['backend']).strip()
            if backend_value and not any(char in backend_value for char in ['&', '|', ';', '`']):
                cmd.extend(['--backend', backend_value])
        
        # Add backend options with input validation
        for key, value in config.items():
            if key != 'backend':
                # Validate key and value to prevent injection
                key_str = str(key).strip()
                value_str = str(value).strip()
                if (key_str and value_str and 
                    not any(char in key_str for char in ['&', '|', ';', '`', '=']) and
                    not any(char in value_str for char in ['&', '|', ';', '`'])):
                    cmd.extend(['--backend-opt', f'{key_str}={value_str}'])
        
        # Add benchmark-specific options using OmegaConf format
        cmd.extend([
            'agent.steps=3'  # Limit iterations for benchmarking
        ])
        
        # Initialize result structure
        result = {
            'backend': backend,
            'task': task.name,
            'start_time': start_time,
            'success': False,
            'error': None,
            'metrics': {},
            'memory_usage': None
        }
        
        try:
            # Run AIDE ML with the task (increased timeout for complex tasks)
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout per task
                cwd=Path(__file__).parent.parent.parent  # Run from project root
            )
            
            result['success'] = process.returncode == 0
            
            # Store only essential output to manage memory
            if len(process.stdout) > 50000:  # Truncate very long outputs
                result['stdout'] = process.stdout[-50000:]  # Keep last 50k chars
                result['output_truncated'] = True
            else:
                result['stdout'] = process.stdout
                result['output_truncated'] = False
            
            if len(process.stderr) > 10000:  # Keep stderr smaller
                result['stderr'] = process.stderr[-10000:]
                result['stderr_truncated'] = True
            else:
                result['stderr'] = process.stderr
                result['stderr_truncated'] = False
            
            if not result['success']:
                # Extract meaningful error from stderr
                error_lines = process.stderr.split('\n')[-10:]  # Last 10 lines
                result['error'] = f"Process failed with code {process.returncode}. Error: {' '.join(error_lines).strip()}"
            
        except subprocess.TimeoutExpired as e:
            result['error'] = "Task timed out after 10 minutes"
            # Try to get partial output
            if hasattr(e, 'stdout') and e.stdout:
                result['stdout'] = e.stdout[-10000:] if len(e.stdout) > 10000 else e.stdout
            if hasattr(e, 'stderr') and e.stderr:
                result['stderr'] = e.stderr[-5000:] if len(e.stderr) > 5000 else e.stderr
        except FileNotFoundError:
            result['error'] = "run_aide.py not found. Make sure you're running from the correct directory."
        except PermissionError:
            result['error'] = "Permission denied running benchmark command"
        except Exception as e:
            result['error'] = f"Unexpected error: {str(e)}"
        
        result['end_time'] = time.time()
        result['duration'] = result['end_time'] - result['start_time']
        
        # Extract performance metrics from logs if available
        result['metrics'] = self._extract_metrics_from_output(result.get('stdout', ''))
        
        return result
    
    def _extract_metrics_from_output(self, output: str) -> Dict[str, Any]:
        """Extract performance metrics from AIDE ML output."""
        metrics = {
            'iterations': 0,
            'total_tokens': 0,
            'solutions_generated': 0,
            'best_score': None
        }
        
        # Simple parsing - this would need to be enhanced based on actual output format
        for line in output.split('\n'):
            if 'Iteration' in line:
                metrics['iterations'] += 1
            elif 'Total tokens' in line:
                try:
                    tokens = int(line.split(':')[-1].strip())
                    metrics['total_tokens'] = tokens
                except:
                    pass
            elif 'Best score' in line:
                try:
                    score = float(line.split(':')[-1].strip())
                    metrics['best_score'] = score
                except:
                    pass
        
        return metrics
    
    def _generate_summary(self, results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Generate summary statistics from benchmark results."""
        summary = {}
        
        for backend, backend_results in results.items():
            successful_runs = [r for r in backend_results if r['success']]
            
            if successful_runs:
                durations = [r['duration'] for r in successful_runs]
                avg_duration = sum(durations) / len(durations)
                
                summary[backend] = {
                    'success_rate': len(successful_runs) / len(backend_results),
                    'avg_duration': avg_duration,
                    'min_duration': min(durations),
                    'max_duration': max(durations),
                    'total_runs': len(backend_results),
                    'successful_runs': len(successful_runs),
                    'failed_runs': len(backend_results) - len(successful_runs),
                    'errors': [r['error'] for r in backend_results if r.get('error')]
                }
                
                # Add metrics summary if available
                metrics_summary = {}
                for metric_key in ['iterations', 'total_tokens', 'best_score']:
                    values = [r['metrics'].get(metric_key, 0) for r in successful_runs 
                             if r['metrics'].get(metric_key) is not None]
                    if values:
                        metrics_summary[metric_key] = {
                            'mean': sum(values) / len(values),
                            'min': min(values),
                            'max': max(values)
                        }
                
                summary[backend]['metrics'] = metrics_summary
            else:
                summary[backend] = {
                    'success_rate': 0,
                    'total_runs': len(backend_results),
                    'errors': [r['error'] for r in backend_results if r.get('error')]
                }
        
        # Identify best performers
        if summary:
            best_success_rate = max(summary.items(), 
                                  key=lambda x: x[1].get('success_rate', 0))
            best_speed = min((k, v) for k, v in summary.items() 
                           if v.get('avg_duration') is not None,
                           key=lambda x: x[1]['avg_duration'])
            
            summary['best_performers'] = {
                'highest_success_rate': best_success_rate[0],
                'fastest_average': best_speed[0] if best_speed else None
            }
        
        return summary
    
    def _save_results(self, results: Dict[str, Any]):
        """Save benchmark results to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f'benchmark_{timestamp}.json'
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Also save as latest
        latest_file = self.output_dir / 'benchmark_latest.json'
        with open(latest_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def _generate_report_header(self, results: Dict[str, Any]) -> str:
        """Generate the report header section."""
        header = "# AIDE ML Backend Benchmark Report\n\n"
        header += f"**Generated:** {results['metadata']['timestamp']}\n\n"
        header += f"**Backends tested:** {', '.join(results['metadata']['backends'])}\n"
        header += f"**Number of tasks:** {results['metadata']['num_tasks']}\n\n"
        return header
    
    def _generate_performance_table(self, results: Dict[str, Any]) -> str:
        """Generate the performance comparison table."""
        table = "### Performance Comparison\n\n"
        table += "| Backend | Success Rate | Avg Duration (s) | Total Runs |\n"
        table += "|---------|--------------|------------------|------------|\n"
        
        for backend, stats in results['summary'].items():
            if backend != 'best_performers':
                success_rate = stats.get('success_rate', 0) * 100
                avg_duration = stats.get('avg_duration', 'N/A')
                if isinstance(avg_duration, float):
                    avg_duration = f"{avg_duration:.2f}"
                
                table += f"| {backend} | {success_rate:.1f}% | {avg_duration} | {stats['total_runs']} |\n"
        
        return table
    
    def _generate_best_performers_section(self, results: Dict[str, Any]) -> str:
        """Generate the best performers section."""
        section = "\n### Best Performers\n\n"
        best = results['summary'].get('best_performers', {})
        
        if best.get('highest_success_rate'):
            section += f"- **Highest Success Rate:** {best['highest_success_rate']}\n"
        if best.get('fastest_average'):
            section += f"- **Fastest Average:** {best['fastest_average']}\n"
        
        return section
    
    def _generate_detailed_results(self, results: Dict[str, Any]) -> str:
        """Generate the detailed results section."""
        section = "\n## Detailed Results\n\n"
        
        for backend, backend_results in results['results'].items():
            section += f"### {backend}\n\n"
            
            for result in backend_results:
                section += f"**Task:** {result['task']}\n"
                section += f"- Success: {'✓' if result['success'] else '✗'}\n"
                section += f"- Duration: {result['duration']:.2f}s\n"
                
                if result.get('error'):
                    section += f"- Error: {result['error']}\n"
                
                if result['metrics']:
                    section += "- Metrics:\n"
                    for key, value in result['metrics'].items():
                        if value is not None:
                            section += f"  - {key}: {value}\n"
                
                section += "\n"
        
        return section
    
    def _generate_report(self, results: Dict[str, Any]):
        """Generate human-readable benchmark report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / f'benchmark_report_{timestamp}.md'
        
        # Generate report content using helper methods
        content = self._generate_report_header(results)
        content += "## Summary\n\n"
        content += self._generate_performance_table(results)
        content += self._generate_best_performers_section(results)
        content += self._generate_detailed_results(results)
        
        # Write to file
        with open(report_file, 'w') as f:
            f.write(content)
        
        # Also save as latest report
        latest_report = self.output_dir / 'benchmark_report_latest.md'
        with open(latest_report, 'w') as f:
            f.write(content)
    
    def compare_with_historical(self, current_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current benchmark results with historical data."""
        historical_file = self.output_dir / 'benchmark_history.json'
        
        history = []
        if historical_file.exists():
            with open(historical_file, 'r') as f:
                history = json.load(f)
        
        # Add current results to history
        history.append({
            'timestamp': current_results['metadata']['timestamp'],
            'summary': current_results['summary']
        })
        
        # Keep last 10 benchmarks
        history = history[-10:]
        
        # Save updated history
        with open(historical_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Generate trend analysis
        trends = {}
        for backend in current_results['summary']:
            if backend != 'best_performers':
                backend_history = [
                    h['summary'].get(backend, {}) 
                    for h in history 
                    if backend in h['summary']
                ]
                
                if len(backend_history) > 1:
                    # Calculate trends
                    success_rates = [h.get('success_rate', 0) for h in backend_history]
                    durations = [h.get('avg_duration', 0) for h in backend_history if h.get('avg_duration')]
                    
                    trends[backend] = {
                        'success_rate_trend': 'improving' if success_rates[-1] > success_rates[0] else 'declining',
                        'speed_trend': 'improving' if durations and durations[-1] < durations[0] else 'declining',
                        'history_length': len(backend_history)
                    }
        
        return trends


def main():
    """CLI interface for backend benchmarking."""
    parser = argparse.ArgumentParser(description='Benchmark AIDE ML backends')
    parser.add_argument('--backends', nargs='+', 
                       help='Backends to benchmark (default: all)')
    parser.add_argument('--tasks-dir', type=Path,
                       help='Directory containing benchmark tasks')
    parser.add_argument('--output-dir', type=Path,
                       help='Directory to save results')
    parser.add_argument('--parallel', type=int, default=1,
                       help='Number of parallel workers')
    
    args = parser.parse_args()
    
    benchmark = BackendBenchmark(
        tasks_dir=args.tasks_dir,
        output_dir=args.output_dir
    )
    
    results = benchmark.run_benchmark(
        backends=args.backends,
        max_workers=args.parallel
    )
    
    # Display summary
    print("\n=== Benchmark Summary ===")
    for backend, stats in results['summary'].items():
        if backend != 'best_performers':
            print(f"\n{backend}:")
            print(f"  Success Rate: {stats.get('success_rate', 0) * 100:.1f}%")
            if stats.get('avg_duration'):
                print(f"  Avg Duration: {stats['avg_duration']:.2f}s")
    
    print(f"\nResults saved to: {benchmark.output_dir}")


if __name__ == '__main__':
    main()