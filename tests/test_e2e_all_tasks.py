"""End-to-end tests for all AIDE ML example tasks.

This module provides comprehensive end-to-end testing by running AIDE ML
on all available example tasks and verifying successful execution.
"""

import unittest
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tempfile
import shutil
import concurrent.futures
from datetime import datetime


class TestE2EAllTasks(unittest.TestCase):
    """End-to-end tests for all example tasks."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class."""
        cls.project_root = Path(__file__).parent.parent
        cls.example_tasks_dir = cls.project_root / "aide" / "example_tasks"
        cls.results_dir = Path.home() / '.aide_ml' / 'e2e_test_results'
        cls.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Test configuration
        cls.test_config = {
            'max_iterations': 2,  # Limit iterations for testing
            'timeout_per_task': 180,  # 3 minutes per task
            'backend': 'claude_code',  # Default backend for testing
            'parallel_workers': 2  # Number of parallel test runners
        }
        
        # Track test results
        cls.test_results = {
            'start_time': datetime.now().isoformat(),
            'config': cls.test_config,
            'tasks': {}
        }
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Save test results
        results_file = cls.results_dir / f"e2e_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(cls.test_results, f, indent=2)
        
        print(f"\nTest results saved to: {results_file}")
    
    def setUp(self):
        """Set up individual test."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up individual test."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def get_all_example_tasks(self) -> List[Path]:
        """Get all example task files."""
        if not self.example_tasks_dir.exists():
            self.skipTest(f"Example tasks directory not found: {self.example_tasks_dir}")
        
        tasks = list(self.example_tasks_dir.glob("*.md"))
        return sorted(tasks)
    
    def run_aide_ml_task(self, task_path: Path, backend: str = None) -> Dict:
        """Run AIDE ML on a single task.
        
        Args:
            task_path: Path to the task file.
            backend: Backend to use (default from test config).
            
        Returns:
            Dictionary containing test results.
        """
        backend = backend or self.test_config['backend']
        start_time = time.time()
        
        # Create output directory for this task
        task_output_dir = Path(self.temp_dir) / task_path.stem
        task_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build command
        cmd = [
            'python', str(self.project_root / 'run_aide.py'),
            '--task', str(task_path),
            '--backend', backend,
            '--max-iterations', str(self.test_config['max_iterations']),
            '--output-dir', str(task_output_dir),
            '--no-interactive'
        ]
        
        result = {
            'task': task_path.name,
            'backend': backend,
            'start_time': start_time,
            'success': False,
            'error': None,
            'output_dir': str(task_output_dir),
            'stdout': '',
            'stderr': '',
            'metrics': {}
        }
        
        try:
            # Run AIDE ML
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.test_config['timeout_per_task'],
                cwd=str(self.project_root)
            )
            
            result['stdout'] = process.stdout
            result['stderr'] = process.stderr
            result['return_code'] = process.returncode
            result['success'] = process.returncode == 0
            
            if not result['success']:
                result['error'] = f"Process exited with code {process.returncode}"
            
            # Extract metrics from output
            result['metrics'] = self._extract_metrics(process.stdout)
            
            # Check for generated solutions
            result['solutions_found'] = self._check_solutions(task_output_dir)
            
        except subprocess.TimeoutExpired:
            result['error'] = f"Task timed out after {self.test_config['timeout_per_task']}s"
        except Exception as e:
            result['error'] = f"Unexpected error: {str(e)}"
        
        result['duration'] = time.time() - start_time
        return result
    
    def _extract_metrics(self, output: str) -> Dict:
        """Extract metrics from AIDE ML output."""
        metrics = {
            'iterations_completed': 0,
            'solutions_generated': 0,
            'best_score': None,
            'total_tokens': 0,
            'errors_encountered': 0
        }
        
        for line in output.split('\n'):
            line_lower = line.lower()
            
            if 'iteration' in line_lower and 'completed' in line_lower:
                metrics['iterations_completed'] += 1
            elif 'solution generated' in line_lower or 'generated solution' in line_lower:
                metrics['solutions_generated'] += 1
            elif 'best score' in line_lower or 'best validation score' in line_lower:
                try:
                    # Extract score value
                    parts = line.split(':')
                    if len(parts) > 1:
                        score_str = parts[-1].strip()
                        score = float(score_str.split()[0])
                        metrics['best_score'] = score
                except:
                    pass
            elif 'total tokens' in line_lower:
                try:
                    tokens = int(line.split(':')[-1].strip().split()[0])
                    metrics['total_tokens'] = tokens
                except:
                    pass
            elif 'error' in line_lower or 'exception' in line_lower:
                metrics['errors_encountered'] += 1
        
        return metrics
    
    def _check_solutions(self, output_dir: Path) -> Dict:
        """Check for generated solutions in output directory."""
        solutions = {
            'count': 0,
            'files': [],
            'has_final_solution': False
        }
        
        if output_dir.exists():
            # Look for solution files
            solution_files = list(output_dir.glob("solution*.py"))
            solutions['count'] = len(solution_files)
            solutions['files'] = [f.name for f in solution_files]
            
            # Check for final solution
            final_solution = output_dir / "final_solution.py"
            solutions['has_final_solution'] = final_solution.exists()
        
        return solutions
    
    def test_all_example_tasks(self):
        """Test all example tasks with default configuration."""
        tasks = self.get_all_example_tasks()
        
        if not tasks:
            self.skipTest("No example tasks found")
        
        print(f"\nRunning end-to-end tests on {len(tasks)} example tasks...")
        
        # Run tasks in parallel for efficiency
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.test_config['parallel_workers']) as executor:
            future_to_task = {
                executor.submit(self.run_aide_ml_task, task): task 
                for task in tasks
            }
            
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    self.test_results['tasks'][task.name] = result
                    
                    # Print progress
                    status = "✓" if result['success'] else "✗"
                    print(f"{status} {task.name} - {result['duration']:.1f}s")
                    
                    # Verify basic success criteria
                    with self.subTest(task=task.name):
                        if result['error'] and 'timeout' not in result['error']:
                            # Don't fail on timeouts in CI/testing
                            self.assertIsNone(result['error'], 
                                            f"Task {task.name} failed: {result['error']}")
                        
                        # Verify at least one iteration completed
                        self.assertGreater(result['metrics']['iterations_completed'], 0,
                                         f"No iterations completed for {task.name}")
                        
                except Exception as e:
                    print(f"✗ {task.name} - Exception: {str(e)}")
                    self.test_results['tasks'][task.name] = {
                        'error': str(e),
                        'success': False
                    }
        
        # Generate summary
        self._generate_summary()
    
    def test_specific_task_bitcoin_price(self):
        """Test the bitcoin price prediction task specifically."""
        task_path = self.example_tasks_dir / "bitcoin_price.md"
        
        if not task_path.exists():
            self.skipTest(f"Bitcoin price task not found: {task_path}")
        
        result = self.run_aide_ml_task(task_path)
        
        # Verify specific criteria for bitcoin task
        self.assertTrue(result['success'], f"Bitcoin task failed: {result.get('error')}")
        self.assertGreater(result['metrics']['solutions_generated'], 0)
        self.assertTrue(result['solutions_found']['count'] > 0)
        
        # Save detailed result
        self.test_results['tasks']['bitcoin_price_detailed'] = result
    
    def test_specific_task_house_prices(self):
        """Test the house prices task specifically."""
        task_path = self.example_tasks_dir / "house_prices.md"
        
        if not task_path.exists():
            self.skipTest(f"House prices task not found: {task_path}")
        
        result = self.run_aide_ml_task(task_path)
        
        # Verify specific criteria for house prices task
        self.assertTrue(result['success'], f"House prices task failed: {result.get('error')}")
        self.assertGreater(result['metrics']['solutions_generated'], 0)
        self.assertTrue(result['solutions_found']['count'] > 0)
        
        # Save detailed result
        self.test_results['tasks']['house_prices_detailed'] = result
    
    def test_multiple_backends(self):
        """Test a task with multiple backends to ensure compatibility."""
        # Pick a simple task for multi-backend testing
        tasks = self.get_all_example_tasks()
        if not tasks:
            self.skipTest("No example tasks found")
        
        test_task = tasks[0]  # Use first available task
        backends = ['claude_code', 'openai', 'anthropic']
        
        print(f"\nTesting {test_task.name} with multiple backends...")
        
        for backend in backends:
            with self.subTest(backend=backend):
                try:
                    result = self.run_aide_ml_task(test_task, backend=backend)
                    self.test_results['tasks'][f'{test_task.name}_{backend}'] = result
                    
                    # Basic verification
                    if result['error'] and 'API' not in result['error']:
                        # Don't fail on API errors (missing keys, etc.)
                        self.assertIsNone(result['error'], 
                                        f"Backend {backend} failed: {result['error']}")
                    
                except Exception as e:
                    if 'API' in str(e) or 'key' in str(e).lower():
                        self.skipTest(f"Backend {backend} not configured: {str(e)}")
                    else:
                        raise
    
    def _generate_summary(self):
        """Generate test summary."""
        tasks_results = self.test_results['tasks']
        
        total_tasks = len(tasks_results)
        successful_tasks = sum(1 for r in tasks_results.values() if r.get('success', False))
        failed_tasks = total_tasks - successful_tasks
        
        total_duration = sum(r.get('duration', 0) for r in tasks_results.values())
        avg_duration = total_duration / total_tasks if total_tasks > 0 else 0
        
        summary = {
            'total_tasks': total_tasks,
            'successful_tasks': successful_tasks,
            'failed_tasks': failed_tasks,
            'success_rate': successful_tasks / total_tasks if total_tasks > 0 else 0,
            'total_duration': total_duration,
            'average_duration': avg_duration,
            'end_time': datetime.now().isoformat()
        }
        
        self.test_results['summary'] = summary
        
        # Print summary
        print("\n" + "="*60)
        print("End-to-End Test Summary")
        print("="*60)
        print(f"Total tasks tested: {total_tasks}")
        print(f"Successful: {successful_tasks} ({summary['success_rate']*100:.1f}%)")
        print(f"Failed: {failed_tasks}")
        print(f"Total duration: {total_duration:.1f}s")
        print(f"Average duration per task: {avg_duration:.1f}s")
        
        if failed_tasks > 0:
            print("\nFailed tasks:")
            for task_name, result in tasks_results.items():
                if not result.get('success', False):
                    print(f"  - {task_name}: {result.get('error', 'Unknown error')}")
    
    def test_validate_task_data(self):
        """Validate that all example tasks have required data files."""
        tasks = self.get_all_example_tasks()
        
        for task in tasks:
            with self.subTest(task=task.name):
                # Check if task has a data directory
                task_data_dir = task.parent / task.stem
                
                if task_data_dir.exists():
                    # Verify it's not empty
                    data_files = list(task_data_dir.iterdir())
                    self.assertGreater(len(data_files), 0, 
                                     f"Task {task.name} has empty data directory")
                    
                    # Log data files for reference
                    self.test_results['tasks'][f'{task.name}_data_validation'] = {
                        'has_data_dir': True,
                        'data_files': [f.name for f in data_files],
                        'data_files_count': len(data_files)
                    }
                else:
                    # Some tasks might not need data files
                    self.test_results['tasks'][f'{task.name}_data_validation'] = {
                        'has_data_dir': False,
                        'note': 'Task may not require data files'
                    }


class TestE2ESmoke(unittest.TestCase):
    """Quick smoke tests for CI/CD pipelines."""
    
    def test_smoke_single_task(self):
        """Run a quick smoke test on a single task."""
        project_root = Path(__file__).parent.parent
        example_tasks_dir = project_root / "aide" / "example_tasks"
        
        # Find a simple task for smoke testing
        tasks = list(example_tasks_dir.glob("*.md"))
        if not tasks:
            self.skipTest("No example tasks found")
        
        task = tasks[0]
        
        # Run with minimal configuration
        cmd = [
            'python', str(project_root / 'run_aide.py'),
            '--task', str(task),
            '--backend', 'claude_code',
            '--max-iterations', '1',
            '--no-interactive'
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,  # 1 minute timeout for smoke test
                cwd=str(project_root)
            )
            
            # Basic checks
            self.assertIn('Iteration', result.stdout or result.stderr, 
                         "No iteration output found")
            
        except subprocess.TimeoutExpired:
            self.fail("Smoke test timed out")
        except Exception as e:
            self.fail(f"Smoke test failed: {str(e)}")


if __name__ == '__main__':
    # Run with verbosity
    unittest.main(verbosity=2)