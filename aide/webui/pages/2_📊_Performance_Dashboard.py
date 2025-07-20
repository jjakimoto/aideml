"""
Performance Monitoring Dashboard for AIDE ML
This page provides real-time visualization of backend performance metrics.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
from datetime import datetime, timedelta
import os
from aide.utils.performance_monitor import PerformanceMonitor

# Set page config
st.set_page_config(
    page_title="AIDE Performance Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize performance monitor
@st.cache_resource
def get_performance_monitor():
    """Get or create a PerformanceMonitor instance."""
    return PerformanceMonitor()

def load_metrics_data(monitor, hours=24):
    """Load performance metrics from the monitor."""
    # Get metrics file path
    metrics_dir = Path.home() / ".aide_ml" / "metrics"
    
    if not metrics_dir.exists():
        return pd.DataFrame()
    
    # Load recent metrics
    all_metrics = []
    cutoff_time = datetime.now() - timedelta(hours=hours)
    
    for backend_dir in metrics_dir.iterdir():
        if backend_dir.is_dir():
            backend_name = backend_dir.name
            for metric_file in backend_dir.glob("*.json"):
                try:
                    with open(metric_file, 'r') as f:
                        metric = json.load(f)
                        metric['backend'] = backend_name
                        metric['timestamp'] = datetime.fromisoformat(metric['timestamp'])
                        if metric['timestamp'] > cutoff_time:
                            all_metrics.append(metric)
                except Exception as e:
                    st.warning(f"Error loading metric file {metric_file}: {e}")
    
    if not all_metrics:
        return pd.DataFrame()
    
    return pd.DataFrame(all_metrics)

def create_performance_summary(df):
    """Create a performance summary table."""
    if df.empty:
        return pd.DataFrame()
    
    summary = df.groupby('backend').agg({
        'duration': ['mean', 'min', 'max', 'count'],
        'success': 'mean',
        'total_tokens': 'sum',
        'prompt_tokens': 'sum',
        'completion_tokens': 'sum'
    }).round(2)
    
    summary.columns = ['Avg Duration (s)', 'Min Duration (s)', 'Max Duration (s)', 
                      'Total Queries', 'Success Rate', 'Total Tokens', 
                      'Prompt Tokens', 'Completion Tokens']
    summary['Success Rate'] = (summary['Success Rate'] * 100).round(1).astype(str) + '%'
    
    return summary

def create_timeline_chart(df):
    """Create a timeline chart of query durations."""
    if df.empty:
        return None
    
    fig = px.scatter(df, x='timestamp', y='duration', color='backend',
                    title='Query Duration Over Time',
                    labels={'duration': 'Duration (seconds)', 'timestamp': 'Time'},
                    hover_data=['model', 'success', 'total_tokens'])
    
    fig.update_layout(height=400)
    return fig

def create_token_usage_chart(df):
    """Create a token usage chart by backend."""
    if df.empty:
        return None
    
    token_summary = df.groupby('backend')[['prompt_tokens', 'completion_tokens']].sum()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Prompt Tokens', x=token_summary.index, y=token_summary['prompt_tokens']))
    fig.add_trace(go.Bar(name='Completion Tokens', x=token_summary.index, y=token_summary['completion_tokens']))
    
    fig.update_layout(
        title='Token Usage by Backend',
        xaxis_title='Backend',
        yaxis_title='Tokens',
        barmode='stack',
        height=400
    )
    return fig

def create_success_rate_chart(df):
    """Create a success rate chart by backend."""
    if df.empty:
        return None
    
    success_rates = df.groupby('backend')['success'].agg(['mean', 'count'])
    success_rates['success_rate'] = success_rates['mean'] * 100
    
    fig = go.Figure(data=[
        go.Bar(
            x=success_rates.index,
            y=success_rates['success_rate'],
            text=[f"{rate:.1f}%<br>({count} queries)" 
                  for rate, count in zip(success_rates['success_rate'], success_rates['count'])],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Success Rate by Backend',
        xaxis_title='Backend',
        yaxis_title='Success Rate (%)',
        yaxis_range=[0, 105],
        height=400
    )
    return fig

def create_model_comparison_chart(df):
    """Create a comparison chart for different models."""
    if df.empty or 'model' not in df.columns:
        return None
    
    model_stats = df.groupby(['backend', 'model']).agg({
        'duration': 'mean',
        'success': 'mean',
        'total_tokens': 'mean'
    }).round(2)
    
    if model_stats.empty:
        return None
    
    # Create subplot figure
    fig = go.Figure()
    
    for backend in model_stats.index.get_level_values(0).unique():
        backend_data = model_stats.loc[backend]
        fig.add_trace(go.Scatter(
            x=backend_data.index,
            y=backend_data['duration'],
            mode='markers+lines',
            name=backend,
            marker=dict(size=10)
        ))
    
    fig.update_layout(
        title='Average Duration by Model',
        xaxis_title='Model',
        yaxis_title='Duration (seconds)',
        height=400
    )
    return fig

# Main dashboard
st.title("📊 AIDE ML Performance Dashboard")
st.markdown("Real-time monitoring of LLM backend performance metrics")

# Sidebar controls
with st.sidebar:
    st.header("Dashboard Controls")
    
    # Time range selector
    time_range = st.selectbox(
        "Time Range",
        options=[1, 6, 12, 24, 48, 168],
        format_func=lambda x: f"Last {x} hours" if x < 168 else "Last week",
        index=3
    )
    
    # Refresh button
    if st.button("🔄 Refresh Data"):
        st.cache_resource.clear()
        st.rerun()
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("Auto-refresh (every 30s)")
    if auto_refresh:
        st.empty()  # Placeholder for auto-refresh logic

# Load performance data
monitor = get_performance_monitor()
df = load_metrics_data(monitor, hours=time_range)

if df.empty:
    st.warning("No performance data available. Run some AIDE ML experiments to see metrics here.")
    st.info("Performance metrics are automatically collected when you run experiments with different backends.")
else:
    # Performance summary
    st.header("Performance Summary")
    summary = create_performance_summary(df)
    if not summary.empty:
        st.dataframe(summary, use_container_width=True)
    
    # Visualizations
    st.header("Performance Visualizations")
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Timeline chart
        timeline_fig = create_timeline_chart(df)
        if timeline_fig:
            st.plotly_chart(timeline_fig, use_container_width=True)
        
        # Success rate chart
        success_fig = create_success_rate_chart(df)
        if success_fig:
            st.plotly_chart(success_fig, use_container_width=True)
    
    with col2:
        # Token usage chart
        token_fig = create_token_usage_chart(df)
        if token_fig:
            st.plotly_chart(token_fig, use_container_width=True)
        
        # Model comparison chart
        model_fig = create_model_comparison_chart(df)
        if model_fig:
            st.plotly_chart(model_fig, use_container_width=True)
    
    # Detailed metrics table
    with st.expander("View Detailed Metrics"):
        st.subheader("Raw Performance Data")
        
        # Add filters
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            selected_backends = st.multiselect(
                "Filter by Backend",
                options=df['backend'].unique(),
                default=df['backend'].unique()
            )
        
        with filter_col2:
            if 'model' in df.columns:
                selected_models = st.multiselect(
                    "Filter by Model",
                    options=df['model'].dropna().unique(),
                    default=df['model'].dropna().unique()
                )
            else:
                selected_models = []
        
        with filter_col3:
            show_failures = st.checkbox("Show failures only", value=False)
        
        # Apply filters
        filtered_df = df[df['backend'].isin(selected_backends)]
        if selected_models and 'model' in df.columns:
            filtered_df = filtered_df[filtered_df['model'].isin(selected_models)]
        if show_failures:
            filtered_df = filtered_df[~filtered_df['success']]
        
        # Display filtered data
        display_columns = ['timestamp', 'backend', 'model', 'duration', 'success', 
                          'prompt_tokens', 'completion_tokens', 'total_tokens']
        display_columns = [col for col in display_columns if col in filtered_df.columns]
        
        st.dataframe(
            filtered_df[display_columns].sort_values('timestamp', ascending=False),
            use_container_width=True
        )
        
        # Export button
        if st.button("📥 Export Data as CSV"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"aide_performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown("Performance metrics are automatically collected during AIDE ML experiments. "
           "Use different backends to compare their performance characteristics.")

# Auto-refresh implementation
if auto_refresh:
    import time
    time.sleep(30)
    st.rerun()