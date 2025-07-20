"""Performance monitoring for LLM backends.

This module provides functionality to track and compare performance metrics
across different LLM backends, including response times, token usage, and
success rates.
"""

import time
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict


@dataclass
class BackendMetrics:
    """Container for backend performance metrics."""
    backend: str
    model: str
    timestamp: datetime
    response_time: float  # seconds
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    success: bool
    error: Optional[str] = None
    task_type: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BackendMetrics':
        """Create metrics from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class PerformanceMonitor:
    """Monitor and analyze backend performance."""
    
    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize performance monitor.
        
        Args:
            log_dir: Directory to save performance logs. If None, uses default location.
        """
        self.log_dir = log_dir or Path.home() / '.aide_ml' / 'performance_logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: List[BackendMetrics] = []
        self.current_log_file = self.log_dir / f"perf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    
    def record_query(self, 
                     backend: str,
                     model: str,
                     start_time: float,
                     end_time: float,
                     total_tokens: int,
                     prompt_tokens: int,
                     completion_tokens: int,
                     success: bool,
                     error: Optional[str] = None,
                     task_type: Optional[str] = None,
                     **kwargs) -> BackendMetrics:
        """Record metrics for a single query.
        
        Returns:
            BackendMetrics object containing the recorded data.
        """
        metrics = BackendMetrics(
            backend=backend,
            model=model,
            timestamp=datetime.now(),
            response_time=end_time - start_time,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            success=success,
            error=error,
            task_type=task_type,
            additional_data=kwargs
        )
        
        self.metrics.append(metrics)
        self._save_metrics(metrics)
        return metrics
    
    def _save_metrics(self, metrics: BackendMetrics):
        """Save metrics to log file."""
        with open(self.current_log_file, 'a') as f:
            f.write(json.dumps(metrics.to_dict()) + '\n')
    
    def load_metrics(self, log_file: Optional[Path] = None) -> List[BackendMetrics]:
        """Load metrics from log file."""
        file_to_load = log_file or self.current_log_file
        loaded_metrics = []
        
        if file_to_load.exists():
            with open(file_to_load, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        loaded_metrics.append(BackendMetrics.from_dict(data))
        
        return loaded_metrics
    
    def get_backend_summary(self, backend: Optional[str] = None, 
                           task_type: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary for a backend.
        
        Args:
            backend: Backend name to filter by. If None, includes all backends.
            task_type: Task type to filter by. If None, includes all tasks.
            
        Returns:
            Dictionary containing performance statistics.
        """
        filtered_metrics = self.metrics
        
        if backend:
            filtered_metrics = [m for m in filtered_metrics if m.backend == backend]
        
        if task_type:
            filtered_metrics = [m for m in filtered_metrics if m.task_type == task_type]
        
        if not filtered_metrics:
            return {
                'backend': backend,
                'task_type': task_type,
                'num_queries': 0,
                'message': 'No metrics found for the specified filters'
            }
        
        response_times = [m.response_time for m in filtered_metrics]
        total_tokens = [m.total_tokens for m in filtered_metrics]
        success_rate = sum(1 for m in filtered_metrics if m.success) / len(filtered_metrics)
        
        return {
            'backend': backend,
            'task_type': task_type,
            'num_queries': len(filtered_metrics),
            'success_rate': success_rate,
            'response_time': {
                'mean': statistics.mean(response_times),
                'median': statistics.median(response_times),
                'min': min(response_times),
                'max': max(response_times),
                'stdev': statistics.stdev(response_times) if len(response_times) > 1 else 0
            },
            'tokens': {
                'mean': statistics.mean(total_tokens),
                'median': statistics.median(total_tokens),
                'total': sum(total_tokens)
            },
            'errors': [m.error for m in filtered_metrics if m.error]
        }
    
    def compare_backends(self, backends: Optional[List[str]] = None,
                        task_type: Optional[str] = None) -> Dict[str, Any]:
        """Compare performance across multiple backends.
        
        Args:
            backends: List of backend names to compare. If None, compares all.
            task_type: Task type to filter by.
            
        Returns:
            Dictionary containing comparative statistics.
        """
        if backends is None:
            backends = list(set(m.backend for m in self.metrics))
        
        comparisons = {}
        for backend in backends:
            comparisons[backend] = self.get_backend_summary(backend, task_type)
        
        # Find best performers
        best_response_time = min(comparisons.items(), 
                               key=lambda x: x[1].get('response_time', {}).get('mean', float('inf')))
        best_success_rate = max(comparisons.items(),
                              key=lambda x: x[1].get('success_rate', 0))
        
        return {
            'backends': comparisons,
            'best_response_time': best_response_time[0],
            'best_success_rate': best_success_rate[0],
            'task_type': task_type
        }
    
    def get_recent_performance(self, backend: str, hours: int = 24) -> Dict[str, Any]:
        """Get performance metrics for recent queries.
        
        Args:
            backend: Backend name.
            hours: Number of hours to look back.
            
        Returns:
            Performance summary for recent period.
        """
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        recent_metrics = [
            m for m in self.metrics 
            if m.backend == backend and m.timestamp.timestamp() > cutoff_time
        ]
        
        if not recent_metrics:
            return {
                'backend': backend,
                'hours': hours,
                'message': f'No metrics found for {backend} in the last {hours} hours'
            }
        
        # Group by hour for trend analysis
        hourly_groups = {}
        for metric in recent_metrics:
            hour_key = metric.timestamp.strftime('%Y-%m-%d %H:00')
            if hour_key not in hourly_groups:
                hourly_groups[hour_key] = []
            hourly_groups[hour_key].append(metric)
        
        hourly_stats = {}
        for hour, metrics in hourly_groups.items():
            response_times = [m.response_time for m in metrics]
            hourly_stats[hour] = {
                'num_queries': len(metrics),
                'avg_response_time': statistics.mean(response_times),
                'success_rate': sum(1 for m in metrics if m.success) / len(metrics)
            }
        
        return {
            'backend': backend,
            'hours': hours,
            'total_queries': len(recent_metrics),
            'hourly_stats': hourly_stats,
            'overall_success_rate': sum(1 for m in recent_metrics if m.success) / len(recent_metrics)
        }


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor(log_dir: Optional[Path] = None) -> PerformanceMonitor:
    """Get or create the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor(log_dir)
    return _performance_monitor


def monitor_query(backend: str, model: str, task_type: Optional[str] = None):
    """Decorator to monitor backend query performance.
    
    Usage:
        @monitor_query('claude_code', 'claude-opus-4', 'code_generation')
        def query_function(...):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            start_time = time.time()
            success = True
            error = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                end_time = time.time()
                
                # Extract token usage from result if available
                total_tokens = 0
                prompt_tokens = 0
                completion_tokens = 0
                
                if 'result' in locals():
                    # Try to extract token counts from result
                    if isinstance(result, dict):
                        total_tokens = result.get('total_tokens', 0)
                        prompt_tokens = result.get('prompt_tokens', 0)
                        completion_tokens = result.get('completion_tokens', 0)
                
                monitor.record_query(
                    backend=backend,
                    model=model,
                    start_time=start_time,
                    end_time=end_time,
                    total_tokens=total_tokens,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    success=success,
                    error=error,
                    task_type=task_type
                )
        
        return wrapper
    return decorator