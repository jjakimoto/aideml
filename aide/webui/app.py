"""
AIDE ML Web UI - Main Application Entry Point

This is the main entry point for the AIDE ML multi-page Streamlit application.
It provides a landing page and navigation to different features.
"""

import streamlit as st
from pathlib import Path


def main():
    """
    Main page for AIDE ML multi-page application.
    """
    st.set_page_config(
        page_title="AIDE: Machine Learning Engineer Agent",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Load custom CSS if available
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Main content
    st.markdown(
        """
        <h1 style='text-align: center;'>🤖 AIDE: Machine Learning Engineer Agent</h1>
        <p style='text-align: center; font-size: 1.2em;'>The open-source AI agent that generates solutions for machine learning tasks</p>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("---")
    
    # Introduction section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            ## Welcome to AIDE ML
            
            AIDE ML is an LLM-powered agent that writes, evaluates, and improves machine learning code 
            to solve data science problems. It uses a tree search algorithm to explore different solutions 
            and find the best approach for your specific task.
            
            ### 🚀 Getting Started
            
            Use the sidebar to navigate between different features:
            
            - **🔬 Experiments**: Run ML experiments with AI assistance
            - **📊 Performance Dashboard**: Monitor and analyze backend performance in real-time
            
            ### 🎯 Key Features
            
            - **Multi-Backend Support**: Choose from OpenAI, Anthropic, Claude Code, Gemini, and more
            - **Tree Search Algorithm**: Systematically explores solution space
            - **Automatic Code Generation**: Generates complete ML solutions
            - **Performance Monitoring**: Track and compare backend performance
            - **Hybrid Backend**: Intelligently route queries to different providers
            
            ### 📚 Resources
            
            - [Documentation](https://github.com/WecoAI/aideml)
            - [Example Tasks](https://github.com/WecoAI/aideml/tree/main/aide/example_tasks)
            - [Research Paper](https://arxiv.org/abs/2405.20309)
            """
        )
    
    # Quick stats
    st.markdown("---")
    st.markdown("### 📈 Quick Stats")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Supported Backends", "6+", "Claude Code, OpenAI, Anthropic...")
    
    with col2:
        st.metric("Example Tasks", "10+", "Classification, Regression, NLP...")
    
    with col3:
        st.metric("Success Rate", "High", "Tree search optimization")
    
    with col4:
        st.metric("Open Source", "✓", "MIT License")
    
    # Latest updates
    st.markdown("---")
    st.markdown("### 🆕 Latest Updates")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(
            """
            **New Performance Dashboard** 🎉
            
            Monitor your LLM backend performance in real-time:
            - Track query durations and success rates
            - Compare performance across different backends
            - Visualize token usage and costs
            - Export metrics for further analysis
            
            Navigate to **📊 Performance Dashboard** in the sidebar to explore!
            """
        )
    
    with col2:
        st.success(
            """
            **Enhanced Backend Support** 🚀
            
            Now featuring:
            - Claude Code SDK integration with MCP
            - Hybrid backend for intelligent routing
            - Performance monitoring for all backends
            - Specialized prompts for ML tasks
            
            Configure your preferred backend in the **🔬 Experiments** page!
            """
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <p style='text-align: center; color: gray;'>
        AIDE ML - Empowering data scientists with AI-driven solutions
        </p>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()