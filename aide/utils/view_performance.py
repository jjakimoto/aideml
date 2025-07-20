#!/usr/bin/env python3
"""CLI tool to view and analyze backend performance metrics."""

import argparse
import json
from pathlib import Path
from datetime import datetime
from tabulate import tabulate

from aide.utils.performance_monitor import get_performance_monitor, BackendMetrics


def print_summary(summary: dict, backend: str):
    """Print a formatted summary of backend performance."""
    print(f"\n=== Performance Summary for {backend} ===")
    print(f"Total queries: {summary['num_queries']}")
    print(f"Success rate: {summary['success_rate']:.2%}")
    
    if summary['num_queries'] > 0:
        print(f"\nResponse Time (seconds):")
        print(f"  Mean: {summary['response_time']['mean']:.3f}")
        print(f"  Median: {summary['response_time']['median']:.3f}")
        print(f"  Min: {summary['response_time']['min']:.3f}")
        print(f"  Max: {summary['response_time']['max']:.3f}")
        print(f"  Std Dev: {summary['response_time']['stdev']:.3f}")
        
        print(f"\nToken Usage:")
        print(f"  Mean: {summary['tokens']['mean']:.0f}")
        print(f"  Median: {summary['tokens']['median']:.0f}")
        print(f"  Total: {summary['tokens']['total']:,}")
        
        if summary['errors']:
            print(f"\nErrors ({len(summary['errors'])}):")
            for i, error in enumerate(summary['errors'][:5]):
                print(f"  {i+1}. {error[:100]}...")


def print_comparison(comparison: dict):
    """Print a comparison table of multiple backends."""
    print("\n=== Backend Comparison ===")
    
    # Prepare data for table
    headers = ["Backend", "Queries", "Success Rate", "Avg Response (s)", "Avg Tokens"]
    rows = []
    
    for backend, stats in comparison['backends'].items():
        if stats['num_queries'] > 0:
            rows.append([
                backend,
                stats['num_queries'],
                f"{stats['success_rate']:.2%}",
                f"{stats['response_time']['mean']:.3f}",
                f"{stats['tokens']['mean']:.0f}"
            ])
    
    if rows:
        print(tabulate(rows, headers=headers, tablefmt="grid"))
        print(f"\nBest response time: {comparison['best_response_time']}")
        print(f"Best success rate: {comparison['best_success_rate']}")
    else:
        print("No data available for comparison.")


def print_recent_performance(recent: dict):
    """Print recent performance trends."""
    print(f"\n=== Recent Performance for {recent['backend']} (last {recent['hours']} hours) ===")
    
    if 'message' in recent:
        print(recent['message'])
        return
    
    print(f"Total queries: {recent['total_queries']}")
    print(f"Overall success rate: {recent['overall_success_rate']:.2%}")
    
    if recent['hourly_stats']:
        print("\nHourly breakdown:")
        headers = ["Hour", "Queries", "Avg Response (s)", "Success Rate"]
        rows = []
        
        for hour in sorted(recent['hourly_stats'].keys()):
            stats = recent['hourly_stats'][hour]
            rows.append([
                hour,
                stats['num_queries'],
                f"{stats['avg_response_time']:.3f}",
                f"{stats['success_rate']:.2%}"
            ])
        
        print(tabulate(rows, headers=headers, tablefmt="grid"))


def main():
    parser = argparse.ArgumentParser(description="View AIDE ML backend performance metrics")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Summary command
    summary_parser = subparsers.add_parser('summary', help='Show performance summary')
    summary_parser.add_argument('--backend', help='Backend to analyze (default: all)')
    summary_parser.add_argument('--task-type', help='Filter by task type')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare backend performance')
    compare_parser.add_argument('--backends', nargs='+', help='Backends to compare (default: all)')
    compare_parser.add_argument('--task-type', help='Filter by task type')
    
    # Recent command
    recent_parser = subparsers.add_parser('recent', help='Show recent performance')
    recent_parser.add_argument('backend', help='Backend to analyze')
    recent_parser.add_argument('--hours', type=int, default=24, help='Number of hours to look back')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export metrics to JSON')
    export_parser.add_argument('output', help='Output file path')
    export_parser.add_argument('--backend', help='Filter by backend')
    
    # Load command
    load_parser = subparsers.add_parser('load', help='Load metrics from specific log file')
    load_parser.add_argument('log_file', help='Log file to load')
    
    args = parser.parse_args()
    
    # Get performance monitor
    monitor = get_performance_monitor()
    
    if args.command == 'summary':
        summary = monitor.get_backend_summary(args.backend, args.task_type)
        print_summary(summary, args.backend or "all backends")
        
    elif args.command == 'compare':
        comparison = monitor.compare_backends(args.backends, args.task_type)
        print_comparison(comparison)
        
    elif args.command == 'recent':
        recent = monitor.get_recent_performance(args.backend, args.hours)
        print_recent_performance(recent)
        
    elif args.command == 'export':
        # Export metrics to JSON
        metrics_to_export = monitor.metrics
        if args.backend:
            metrics_to_export = [m for m in metrics_to_export if m.backend == args.backend]
        
        with open(args.output, 'w') as f:
            json.dump([m.to_dict() for m in metrics_to_export], f, indent=2)
        print(f"Exported {len(metrics_to_export)} metrics to {args.output}")
        
    elif args.command == 'load':
        # Load metrics from file
        loaded = monitor.load_metrics(Path(args.log_file))
        monitor.metrics.extend(loaded)
        print(f"Loaded {len(loaded)} metrics from {args.log_file}")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()