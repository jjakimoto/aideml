"""
Tests for the Performance Dashboard functionality.
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aide.utils.performance_monitor import PerformanceMonitor, BackendMetrics


class TestPerformanceDashboard:
    """Test suite for performance dashboard functionality."""
    
    @pytest.fixture
    def temp_metrics_dir(self):
        """Create temporary metrics directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_dir = Path(tmpdir) / ".aide_ml" / "metrics"
            metrics_dir.mkdir(parents=True)
            yield metrics_dir
    
    @pytest.fixture
    def sample_metrics(self):
        """Generate sample metrics data."""
        base_time = datetime.now()
        metrics = []
        
        # Create metrics for different backends
        backends = ["openai", "anthropic", "claude_code"]
        models = {
            "openai": ["gpt-4o", "gpt-4o-mini"],
            "anthropic": ["claude-3-5-sonnet-20241022"],
            "claude_code": ["claude-opus-4"],
        }
        
        for i, backend in enumerate(backends):
            for j, model in enumerate(models[backend]):
                for k in range(5):  # 5 queries per model
                    metric = BackendMetrics(
                        backend=backend,
                        model=model,
                        timestamp=(base_time - timedelta(hours=i*2 + j + k/10)).isoformat(),
                        duration=2.5 + i + j * 0.5 + k * 0.1,
                        success=k != 2,  # Make one query fail
                        prompt_tokens=1000 + i * 100 + j * 50,
                        completion_tokens=500 + i * 50 + j * 25,
                        total_tokens=1500 + i * 150 + j * 75,
                        error="Timeout error" if k == 2 else None,
                    )
                    metrics.append(metric)
        
        return metrics
    
    def test_metrics_file_creation(self, temp_metrics_dir, sample_metrics):
        """Test creating metrics files in the correct structure."""
        # Write metrics to files
        for metric in sample_metrics:
            backend_dir = temp_metrics_dir / metric.backend
            backend_dir.mkdir(exist_ok=True)
            
            # Create unique filename based on timestamp
            timestamp_str = metric.timestamp.replace(":", "-").replace(".", "-")
            metric_file = backend_dir / f"metric_{timestamp_str}.json"
            
            with open(metric_file, 'w') as f:
                json.dump(metric.__dict__, f)
        
        # Verify files were created
        for backend in ["openai", "anthropic", "claude_code"]:
            backend_dir = temp_metrics_dir / backend
            assert backend_dir.exists()
            assert len(list(backend_dir.glob("*.json"))) > 0
    
    def test_load_metrics_data(self, temp_metrics_dir, sample_metrics):
        """Test loading metrics data from files."""
        # Write sample metrics
        for metric in sample_metrics:
            backend_dir = temp_metrics_dir / metric.backend
            backend_dir.mkdir(exist_ok=True)
            timestamp_str = metric.timestamp.replace(":", "-").replace(".", "-")
            metric_file = backend_dir / f"metric_{timestamp_str}.json"
            with open(metric_file, 'w') as f:
                json.dump(metric.__dict__, f)
        
        # Simulate loading metrics (similar to dashboard function)
        all_metrics = []
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        for backend_dir in temp_metrics_dir.iterdir():
            if backend_dir.is_dir():
                backend_name = backend_dir.name
                for metric_file in backend_dir.glob("*.json"):
                    with open(metric_file, 'r') as f:
                        metric = json.load(f)
                        metric['backend'] = backend_name
                        metric['timestamp'] = datetime.fromisoformat(metric['timestamp'])
                        if metric['timestamp'] > cutoff_time:
                            all_metrics.append(metric)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_metrics)
        
        # Verify data loaded correctly
        assert not df.empty
        assert len(df) == len(sample_metrics)
        assert set(df['backend'].unique()) == {"openai", "anthropic", "claude_code"}
    
    def test_performance_summary_calculation(self, sample_metrics):
        """Test calculation of performance summary statistics."""
        # Convert metrics to DataFrame
        df = pd.DataFrame([m.__dict__ for m in sample_metrics])
        
        # Calculate summary statistics
        summary = df.groupby('backend').agg({
            'duration': ['mean', 'min', 'max', 'count'],
            'success': 'mean',
            'total_tokens': 'sum',
            'prompt_tokens': 'sum',
            'completion_tokens': 'sum'
        }).round(2)
        
        # Verify summary calculations
        assert not summary.empty
        assert len(summary) == 3  # Three backends
        
        # Check success rate calculation
        for backend in summary.index:
            backend_data = df[df['backend'] == backend]
            expected_success_rate = backend_data['success'].mean()
            actual_success_rate = summary.loc[backend, ('success', 'mean')]
            assert abs(expected_success_rate - actual_success_rate) < 0.01
    
    def test_token_usage_aggregation(self, sample_metrics):
        """Test token usage aggregation by backend."""
        df = pd.DataFrame([m.__dict__ for m in sample_metrics])
        
        # Aggregate token usage
        token_summary = df.groupby('backend')[['prompt_tokens', 'completion_tokens']].sum()
        
        # Verify aggregation
        assert not token_summary.empty
        assert all(token_summary['prompt_tokens'] > 0)
        assert all(token_summary['completion_tokens'] > 0)
        
        # Verify totals match
        for backend in token_summary.index:
            backend_data = df[df['backend'] == backend]
            assert token_summary.loc[backend, 'prompt_tokens'] == backend_data['prompt_tokens'].sum()
            assert token_summary.loc[backend, 'completion_tokens'] == backend_data['completion_tokens'].sum()
    
    def test_time_filtering(self, sample_metrics):
        """Test filtering metrics by time range."""
        df = pd.DataFrame([m.__dict__ for m in sample_metrics])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Test different time ranges
        time_ranges = [1, 6, 12, 24]
        
        for hours in time_ranges:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            filtered_df = df[df['timestamp'] > cutoff_time]
            
            # All recent metrics should be included
            if hours >= 6:
                assert len(filtered_df) == len(df)
            else:
                assert len(filtered_df) < len(df)
    
    def test_model_comparison_data(self, sample_metrics):
        """Test data preparation for model comparison."""
        df = pd.DataFrame([m.__dict__ for m in sample_metrics])
        
        # Group by backend and model
        model_stats = df.groupby(['backend', 'model']).agg({
            'duration': 'mean',
            'success': 'mean',
            'total_tokens': 'mean'
        }).round(2)
        
        # Verify model statistics
        assert not model_stats.empty
        assert len(model_stats) == 4  # Total number of unique backend-model combinations
        
        # Check multi-index structure
        assert model_stats.index.nlevels == 2
        assert 'openai' in model_stats.index.get_level_values(0)
        assert 'gpt-4o' in model_stats.index.get_level_values(1)
    
    def test_export_functionality(self, sample_metrics, tmp_path):
        """Test exporting metrics data to CSV."""
        df = pd.DataFrame([m.__dict__ for m in sample_metrics])
        
        # Export to CSV
        export_path = tmp_path / "metrics_export.csv"
        df.to_csv(export_path, index=False)
        
        # Verify export
        assert export_path.exists()
        
        # Read back and verify
        imported_df = pd.read_csv(export_path)
        assert len(imported_df) == len(df)
        assert list(imported_df.columns) == list(df.columns)
    
    def test_dashboard_with_empty_metrics(self, temp_metrics_dir):
        """Test dashboard behavior with no metrics data."""
        # Ensure metrics directory is empty
        assert not list(temp_metrics_dir.iterdir())
        
        # Simulate loading empty metrics
        all_metrics = []
        for backend_dir in temp_metrics_dir.iterdir():
            if backend_dir.is_dir():
                for metric_file in backend_dir.glob("*.json"):
                    with open(metric_file, 'r') as f:
                        all_metrics.append(json.load(f))
        
        df = pd.DataFrame(all_metrics)
        assert df.empty
    
    def test_failure_filtering(self, sample_metrics):
        """Test filtering to show only failed queries."""
        df = pd.DataFrame([m.__dict__ for m in sample_metrics])
        
        # Filter failures
        failures_df = df[~df['success']]
        
        # Verify failures
        assert not failures_df.empty
        assert all(~failures_df['success'])
        assert all(failures_df['error'].notna())
        
        # Check error messages
        assert all(failures_df['error'] == "Timeout error")