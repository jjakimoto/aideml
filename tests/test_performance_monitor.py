"""Tests for the performance monitoring module."""

import pytest
import time
import tempfile
from pathlib import Path
from datetime import datetime

from aide.utils.performance_monitor import (
    PerformanceMonitor, 
    BackendMetrics,
    get_performance_monitor,
    monitor_query
)


class TestBackendMetrics:
    """Test the BackendMetrics dataclass."""
    
    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = BackendMetrics(
            backend="openai",
            model="gpt-4",
            timestamp=datetime.now(),
            response_time=1.5,
            total_tokens=100,
            prompt_tokens=50,
            completion_tokens=50,
            success=True,
            task_type="classification"
        )
        
        data = metrics.to_dict()
        assert data["backend"] == "openai"
        assert data["model"] == "gpt-4"
        assert data["response_time"] == 1.5
        assert data["total_tokens"] == 100
        assert isinstance(data["timestamp"], str)
    
    def test_from_dict(self):
        """Test creating metrics from dictionary."""
        now = datetime.now()
        data = {
            "backend": "anthropic",
            "model": "claude-3",
            "timestamp": now.isoformat(),
            "response_time": 2.0,
            "total_tokens": 200,
            "prompt_tokens": 100,
            "completion_tokens": 100,
            "success": True,
            "error": None,
            "task_type": "regression",
            "additional_data": {}
        }
        
        metrics = BackendMetrics.from_dict(data)
        assert metrics.backend == "anthropic"
        assert metrics.model == "claude-3"
        assert metrics.response_time == 2.0
        assert metrics.task_type == "regression"


class TestPerformanceMonitor:
    """Test the PerformanceMonitor class."""
    
    def test_initialization(self):
        """Test monitor initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = PerformanceMonitor(Path(tmpdir))
            assert monitor.log_dir == Path(tmpdir)
            assert monitor.metrics == []
            assert monitor.current_log_file.parent == Path(tmpdir)
    
    def test_record_query(self):
        """Test recording a query."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = PerformanceMonitor(Path(tmpdir))
            
            start_time = time.time()
            time.sleep(0.1)  # Simulate query time
            end_time = time.time()
            
            metrics = monitor.record_query(
                backend="openai",
                model="gpt-4",
                start_time=start_time,
                end_time=end_time,
                total_tokens=150,
                prompt_tokens=50,
                completion_tokens=100,
                success=True,
                task_type="classification"
            )
            
            assert len(monitor.metrics) == 1
            assert metrics.backend == "openai"
            assert metrics.response_time >= 0.1
            assert metrics.total_tokens == 150
            
            # Check that it was saved to file
            assert monitor.current_log_file.exists()
    
    def test_load_metrics(self):
        """Test loading metrics from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = PerformanceMonitor(Path(tmpdir))
            
            # Record some metrics
            for i in range(3):
                monitor.record_query(
                    backend="openai",
                    model="gpt-4",
                    start_time=time.time(),
                    end_time=time.time() + 0.5,
                    total_tokens=100 + i * 10,
                    prompt_tokens=50,
                    completion_tokens=50 + i * 10,
                    success=True
                )
            
            # Load from file
            loaded = monitor.load_metrics()
            assert len(loaded) == 3
            assert all(m.backend == "openai" for m in loaded)
    
    def test_get_backend_summary(self):
        """Test getting backend performance summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = PerformanceMonitor(Path(tmpdir))
            
            # Record metrics for different backends
            for backend in ["openai", "anthropic", "openai"]:
                monitor.record_query(
                    backend=backend,
                    model="test-model",
                    start_time=time.time(),
                    end_time=time.time() + 1.0,
                    total_tokens=100,
                    prompt_tokens=50,
                    completion_tokens=50,
                    success=True
                )
            
            # Get summary for openai
            summary = monitor.get_backend_summary("openai")
            assert summary["num_queries"] == 2
            assert summary["success_rate"] == 1.0
            assert "response_time" in summary
            assert "tokens" in summary
    
    def test_compare_backends(self):
        """Test comparing performance across backends."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = PerformanceMonitor(Path(tmpdir))
            
            # Record metrics for different backends
            backends_data = [
                ("openai", 0.5, True),
                ("anthropic", 0.3, True),
                ("openai", 0.6, False),
                ("anthropic", 0.4, True),
            ]
            
            for backend, response_time, success in backends_data:
                monitor.record_query(
                    backend=backend,
                    model="test-model",
                    start_time=time.time(),
                    end_time=time.time() + response_time,
                    total_tokens=100,
                    prompt_tokens=50,
                    completion_tokens=50,
                    success=success
                )
            
            comparison = monitor.compare_backends()
            assert "openai" in comparison["backends"]
            assert "anthropic" in comparison["backends"]
            assert comparison["best_response_time"] == "anthropic"  # Lower avg response time
            assert comparison["best_success_rate"] == "anthropic"  # 100% success rate
    
    def test_get_recent_performance(self):
        """Test getting recent performance metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = PerformanceMonitor(Path(tmpdir))
            
            # Record some metrics
            for i in range(5):
                monitor.record_query(
                    backend="openai",
                    model="gpt-4",
                    start_time=time.time(),
                    end_time=time.time() + 0.5,
                    total_tokens=100,
                    prompt_tokens=50,
                    completion_tokens=50,
                    success=True
                )
            
            recent = monitor.get_recent_performance("openai", hours=1)
            assert recent["backend"] == "openai"
            assert recent["total_queries"] == 5
            assert recent["overall_success_rate"] == 1.0
            assert len(recent["hourly_stats"]) > 0


class TestMonitorDecorator:
    """Test the monitor_query decorator."""
    
    def test_decorator_success(self):
        """Test decorator with successful function."""
        monitor = get_performance_monitor()
        initial_count = len(monitor.metrics)
        
        @monitor_query("test_backend", "test_model", "test_task")
        def test_function():
            time.sleep(0.1)
            return {"total_tokens": 100, "prompt_tokens": 40, "completion_tokens": 60}
        
        result = test_function()
        
        assert len(monitor.metrics) > initial_count
        last_metric = monitor.metrics[-1]
        assert last_metric.backend == "test_backend"
        assert last_metric.model == "test_model"
        assert last_metric.task_type == "test_task"
        assert last_metric.success is True
        assert last_metric.response_time >= 0.1
    
    def test_decorator_failure(self):
        """Test decorator with failing function."""
        monitor = get_performance_monitor()
        initial_count = len(monitor.metrics)
        
        @monitor_query("test_backend", "test_model", "test_task")
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_function()
        
        assert len(monitor.metrics) > initial_count
        last_metric = monitor.metrics[-1]
        assert last_metric.success is False
        assert "Test error" in last_metric.error


if __name__ == "__main__":
    pytest.main([__file__])