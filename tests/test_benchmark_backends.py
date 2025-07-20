"""Tests for backend benchmarking functionality."""

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import json
import tempfile

from aide.utils.benchmark_backends import BackendBenchmark


class TestBackendBenchmark(unittest.TestCase):
    """Test cases for BackendBenchmark class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.benchmark = BackendBenchmark(
            tasks_dir=self.temp_path / "tasks",
            output_dir=self.temp_path / "output"
        )
        
        # Create test task
        self.benchmark.tasks_dir.mkdir(parents=True, exist_ok=True)
        test_task = self.benchmark.tasks_dir / "test_task.md"
        test_task.write_text("""# Test Task
        
This is a simple test task for benchmarking.

## Task
Predict the value of y = 2x + 1 for x = 5.
""")
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test benchmark initialization."""
        self.assertTrue(self.benchmark.output_dir.exists())
        self.assertIsNotNone(self.benchmark.performance_monitor)
        self.assertIn('claude_code', self.benchmark.backend_configs)
        self.assertIn('openai', self.benchmark.backend_configs)
        
        # Test available backends
        expected_backends = {'openai', 'anthropic', 'openrouter', 'gemini', 'claude_code', 'hybrid'}
        self.assertEqual(self.benchmark.available_backends, expected_backends)
    
    def test_get_benchmark_tasks(self):
        """Test getting benchmark tasks."""
        tasks = self.benchmark.get_benchmark_tasks()
        self.assertGreater(len(tasks), 0)
        self.assertTrue(all(t.suffix == '.md' for t in tasks))
    
    @patch('subprocess.run')
    def test_run_single_benchmark(self, mock_run):
        """Test running a single benchmark."""
        # Mock subprocess output
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "Iteration 1\nTotal tokens: 1000\nBest score: 0.95"
        mock_process.stderr = ""
        mock_run.return_value = mock_process
        
        task = self.benchmark.tasks_dir / "test_task.md"
        result = self.benchmark._run_single_benchmark('claude_code', task)
        
        self.assertEqual(result['backend'], 'claude_code')
        self.assertEqual(result['task'], 'test_task.md')
        self.assertTrue(result['success'])
        self.assertIsNone(result['error'])
        self.assertIn('duration', result)
        self.assertGreater(result['duration'], 0)
        
        # Check new fields added in improvements
        self.assertIn('memory_usage', result)
        self.assertIn('output_truncated', result)
        self.assertFalse(result['output_truncated'])  # Should be False for short output
        
        # Check metrics extraction
        self.assertEqual(result['metrics']['iterations'], 1)
        self.assertEqual(result['metrics']['total_tokens'], 1000)
        self.assertEqual(result['metrics']['best_score'], 0.95)
    
    def test_extract_metrics_from_output(self):
        """Test metric extraction from output."""
        output = """
Starting AIDE ML...
Iteration 1
Generated solution...
Total tokens: 1500
Iteration 2
Best score: 0.87
Total tokens: 2000
"""
        metrics = self.benchmark._extract_metrics_from_output(output)
        
        self.assertEqual(metrics['iterations'], 2)
        self.assertEqual(metrics['total_tokens'], 2000)  # Should get the last value
        self.assertEqual(metrics['best_score'], 0.87)
    
    def test_generate_summary(self):
        """Test summary generation."""
        results = {
            'backend1': [
                {
                    'success': True,
                    'duration': 10.5,
                    'metrics': {'iterations': 3, 'total_tokens': 1000}
                },
                {
                    'success': True,
                    'duration': 12.0,
                    'metrics': {'iterations': 4, 'total_tokens': 1200}
                },
                {
                    'success': False,
                    'duration': 5.0,
                    'error': 'Timeout',
                    'metrics': {}
                }
            ],
            'backend2': [
                {
                    'success': True,
                    'duration': 8.0,
                    'metrics': {'iterations': 2, 'total_tokens': 800}
                },
                {
                    'success': True,
                    'duration': 9.0,
                    'metrics': {'iterations': 3, 'total_tokens': 900}
                }
            ]
        }
        
        summary = self.benchmark._generate_summary(results)
        
        # Check backend1 summary
        self.assertAlmostEqual(summary['backend1']['success_rate'], 2/3)
        self.assertAlmostEqual(summary['backend1']['avg_duration'], 11.25)  # (10.5 + 12.0) / 2
        self.assertEqual(summary['backend1']['total_runs'], 3)
        self.assertEqual(summary['backend1']['successful_runs'], 2)
        self.assertIn('Timeout', summary['backend1']['errors'])
        
        # Check backend2 summary
        self.assertEqual(summary['backend2']['success_rate'], 1.0)
        self.assertAlmostEqual(summary['backend2']['avg_duration'], 8.5)  # (8.0 + 9.0) / 2
        
        # Check best performers
        self.assertEqual(summary['best_performers']['highest_success_rate'], 'backend2')
        self.assertEqual(summary['best_performers']['fastest_average'], 'backend2')
    
    def test_save_and_load_results(self):
        """Test saving and loading results."""
        test_results = {
            'metadata': {
                'timestamp': '2025-07-20T10:00:00',
                'backends': ['test_backend'],
                'tasks': ['test_task.md']
            },
            'results': {
                'test_backend': [{'success': True, 'duration': 10.0}]
            },
            'summary': {
                'test_backend': {'success_rate': 1.0}
            }
        }
        
        self.benchmark._save_results(test_results)
        
        # Check that files were created
        latest_file = self.benchmark.output_dir / 'benchmark_latest.json'
        self.assertTrue(latest_file.exists())
        
        # Load and verify
        with open(latest_file, 'r') as f:
            loaded = json.load(f)
        
        self.assertEqual(loaded['metadata']['backends'], ['test_backend'])
        self.assertEqual(loaded['summary']['test_backend']['success_rate'], 1.0)
    
    def test_generate_report(self):
        """Test report generation."""
        test_results = {
            'metadata': {
                'timestamp': '2025-07-20T10:00:00',
                'backends': ['backend1', 'backend2'],
                'tasks': ['task1.md', 'task2.md'],
                'num_tasks': 2,
                'num_backends': 2
            },
            'results': {
                'backend1': [
                    {
                        'task': 'task1.md',
                        'success': True,
                        'duration': 10.0,
                        'metrics': {'iterations': 3}
                    },
                    {
                        'task': 'task2.md',
                        'success': False,
                        'duration': 5.0,
                        'error': 'Failed',
                        'metrics': {}
                    }
                ],
                'backend2': [
                    {
                        'task': 'task1.md',
                        'success': True,
                        'duration': 8.0,
                        'metrics': {'iterations': 2}
                    },
                    {
                        'task': 'task2.md',
                        'success': True,
                        'duration': 9.0,
                        'metrics': {'iterations': 3}
                    }
                ]
            },
            'summary': {
                'backend1': {
                    'success_rate': 0.5,
                    'avg_duration': 10.0,
                    'total_runs': 2,
                    'successful_runs': 1,
                    'failed_runs': 1
                },
                'backend2': {
                    'success_rate': 1.0,
                    'avg_duration': 8.5,
                    'total_runs': 2,
                    'successful_runs': 2,
                    'failed_runs': 0
                },
                'best_performers': {
                    'highest_success_rate': 'backend2',
                    'fastest_average': 'backend2'
                }
            }
        }
        
        self.benchmark._generate_report(test_results)
        
        # Check report file exists
        report_files = list(self.benchmark.output_dir.glob('benchmark_report_*.md'))
        self.assertGreater(len(report_files), 0)
        
        # Check latest report
        latest_report = self.benchmark.output_dir / 'benchmark_report_latest.md'
        self.assertTrue(latest_report.exists())
        
        # Verify content
        content = latest_report.read_text()
        self.assertIn('# AIDE ML Backend Benchmark Report', content)
        self.assertIn('backend1', content)
        self.assertIn('backend2', content)
        self.assertIn('50.0%', content)  # backend1 success rate
        self.assertIn('100.0%', content)  # backend2 success rate
    
    def test_report_generation_methods(self):
        """Test the refactored report generation methods."""
        test_results = {
            'metadata': {
                'timestamp': '2025-07-20T10:00:00',
                'backends': ['backend1'],
                'tasks': ['task1.md'],
                'num_tasks': 1,
                'num_backends': 1
            },
            'results': {
                'backend1': [
                    {
                        'task': 'task1.md',
                        'success': True,
                        'duration': 10.0,
                        'metrics': {'iterations': 3}
                    }
                ]
            },
            'summary': {
                'backend1': {
                    'success_rate': 1.0,
                    'avg_duration': 10.0,
                    'total_runs': 1
                },
                'best_performers': {
                    'highest_success_rate': 'backend1',
                    'fastest_average': 'backend1'
                }
            }
        }
        
        # Test individual report generation methods
        header = self.benchmark._generate_report_header(test_results)
        self.assertIn('AIDE ML Backend Benchmark Report', header)
        self.assertIn('2025-07-20T10:00:00', header)
        
        table = self.benchmark._generate_performance_table(test_results)
        self.assertIn('Performance Comparison', table)
        self.assertIn('backend1', table)
        self.assertIn('100.0%', table)  # Success rate
        
        best_performers = self.benchmark._generate_best_performers_section(test_results)
        self.assertIn('Best Performers', best_performers)
        self.assertIn('backend1', best_performers)
        
        detailed = self.benchmark._generate_detailed_results(test_results)
        self.assertIn('Detailed Results', detailed)
        self.assertIn('task1.md', detailed)
    
    def test_compare_with_historical(self):
        """Test historical comparison."""
        current_results = {
            'metadata': {'timestamp': '2025-07-20T12:00:00'},
            'summary': {
                'backend1': {
                    'success_rate': 0.9,
                    'avg_duration': 10.0
                }
            }
        }
        
        # Create some historical data
        history_file = self.benchmark.output_dir / 'benchmark_history.json'
        history = [
            {
                'timestamp': '2025-07-20T10:00:00',
                'summary': {
                    'backend1': {
                        'success_rate': 0.8,
                        'avg_duration': 12.0
                    }
                }
            }
        ]
        
        with open(history_file, 'w') as f:
            json.dump(history, f)
        
        trends = self.benchmark.compare_with_historical(current_results)
        
        # Check trends
        self.assertEqual(trends['backend1']['success_rate_trend'], 'improving')
        self.assertEqual(trends['backend1']['speed_trend'], 'improving')
        self.assertEqual(trends['backend1']['history_length'], 2)
    
    @patch('aide.utils.benchmark_backends.BackendBenchmark._run_single_benchmark')
    def test_run_benchmark(self, mock_run_single):
        """Test full benchmark run."""
        # Mock single benchmark runs
        mock_run_single.return_value = {
            'backend': 'test_backend',
            'task': 'test_task.md',
            'success': True,
            'duration': 10.0,
            'metrics': {'iterations': 3}
        }
        
        # Run benchmark with one backend and one task
        results = self.benchmark.run_benchmark(
            backends=['claude_code'],
            tasks=[self.benchmark.tasks_dir / 'test_task.md']
        )
        
        # Verify results structure
        self.assertIn('metadata', results)
        self.assertIn('results', results)
        self.assertIn('summary', results)
        
        # Verify metadata
        self.assertEqual(results['metadata']['backends'], ['claude_code'])
        self.assertEqual(results['metadata']['num_tasks'], 1)
        self.assertEqual(results['metadata']['num_backends'], 1)
        
        # Verify that benchmark was called
        mock_run_single.assert_called_once()
    
    def test_validate_backends_success(self):
        """Test successful backend validation."""
        valid_backends = ['claude_code', 'openai']
        result = self.benchmark._validate_backends(valid_backends)
        self.assertEqual(result, valid_backends)
    
    def test_validate_backends_invalid(self):
        """Test backend validation with invalid backends."""
        invalid_backends = ['invalid_backend', 'another_invalid']
        with self.assertRaises(ValueError) as cm:
            self.benchmark._validate_backends(invalid_backends)
        
        self.assertIn("No valid backends found", str(cm.exception))
        self.assertIn("Available backends:", str(cm.exception))
    
    def test_validate_backends_mixed(self):
        """Test backend validation with mixed valid/invalid backends."""
        mixed_backends = ['claude_code', 'invalid_backend', 'openai']
        
        # Capture stdout to check warning messages
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            result = self.benchmark._validate_backends(mixed_backends)
            self.assertEqual(set(result), {'claude_code', 'openai'})
            
            # Check warning was printed
            output = captured_output.getvalue()
            self.assertIn("Warning: Skipping invalid backends", output)
            self.assertIn("invalid_backend", output)
        finally:
            sys.stdout = sys.__stdout__
    
    def test_validate_backends_unconfigured(self):
        """Test backend validation with unconfigured but available backends."""
        # Remove a backend from configs to test this case
        original_config = self.benchmark.backend_configs.pop('openai', None)
        
        try:
            backends = ['openai', 'claude_code']
            result = self.benchmark._validate_backends(backends)
            # Should only return claude_code since openai is not configured
            self.assertEqual(result, ['claude_code'])
        finally:
            # Restore the config
            if original_config:
                self.benchmark.backend_configs['openai'] = original_config


if __name__ == '__main__':
    unittest.main()