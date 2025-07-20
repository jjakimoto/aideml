"""Hybrid backend that routes queries to different providers based on task type.

This backend enables using specialized models for different tasks:
- Claude Code for code generation
- Other models for analysis, review, and general tasks
"""

import logging
from typing import Dict, Optional, Tuple, Any

from .utils import FunctionSpec
from . import query as base_query

logger = logging.getLogger("aide.backend.hybrid")


class HybridConfig:
    """Configuration for hybrid backend routing."""
    
    def __init__(self,
                 code_backend: str = "claude_code",
                 code_model: str = "claude-opus-4",
                 analysis_backend: Optional[str] = None,
                 analysis_model: str = "gpt-4o",
                 default_backend: Optional[str] = None,
                 default_model: str = "gpt-4o"):
        """Initialize hybrid configuration.
        
        Args:
            code_backend: Backend to use for code generation tasks
            code_model: Model to use for code generation
            analysis_backend: Backend to use for analysis tasks (if None, uses default)
            analysis_model: Model to use for analysis
            default_backend: Default backend for other tasks
            default_model: Default model for other tasks
        """
        self.code_backend = code_backend
        self.code_model = code_model
        self.analysis_backend = analysis_backend
        self.analysis_model = analysis_model
        self.default_backend = default_backend
        self.default_model = default_model


# Global hybrid configuration
_hybrid_config: Optional[HybridConfig] = None


def set_hybrid_config(config: HybridConfig):
    """Set the global hybrid configuration."""
    global _hybrid_config
    _hybrid_config = config


def get_hybrid_config() -> HybridConfig:
    """Get the global hybrid configuration."""
    global _hybrid_config
    if _hybrid_config is None:
        _hybrid_config = HybridConfig()
    return _hybrid_config


def detect_task_type(system_message: Optional[str], 
                    user_message: Optional[str],
                    func_spec: Optional[FunctionSpec]) -> str:
    """Detect the type of task from the query content.
    
    Returns:
        'code_generation', 'analysis', or 'general'
    """
    # If there's a function spec, it's likely an analysis task (like review)
    if func_spec is not None:
        if func_spec.name == "submit_review":
            return 'analysis'
    
    # Check message content for code generation indicators
    messages = []
    if system_message:
        messages.append(system_message.lower())
    if user_message:
        messages.append(user_message.lower())
    
    combined_text = ' '.join(messages)
    
    # Code generation indicators
    code_indicators = [
        'write code', 'implement', 'create a function', 'create a class',
        'write a script', 'write python', 'code solution', 'program',
        'def ', 'class ', 'import ', 'python code', 'task description:',
        'implementation guideline', 'code:', 'plan and code'
    ]
    
    # Analysis indicators
    analysis_indicators = [
        'analyze', 'review', 'evaluate', 'assess', 'explain',
        'what is', 'describe', 'summary', 'findings', 'results',
        'execution output', 'output log', 'metric', 'validation'
    ]
    
    code_score = sum(1 for indicator in code_indicators if indicator in combined_text)
    analysis_score = sum(1 for indicator in analysis_indicators if indicator in combined_text)
    
    if code_score > analysis_score:
        return 'code_generation'
    elif analysis_score > 0:
        return 'analysis'
    else:
        return 'general'


def query(
    system_message: Optional[str],
    user_message: Optional[str],
    model: Optional[str] = None,
    backend: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    func_spec: Optional[FunctionSpec] = None,
    **model_kwargs,
) -> Tuple[Any, float, int, int, Dict[str, Any]]:
    """Hybrid query that routes to different backends based on task type.
    
    This function automatically detects the task type and routes to the
    appropriate backend. You can override this by setting task_type in
    model_kwargs.
    """
    config = get_hybrid_config()
    
    # Check if task type is explicitly specified
    task_type = model_kwargs.get('task_type')
    if task_type is None:
        # Auto-detect task type
        task_type = detect_task_type(system_message, user_message, func_spec)
    
    # Determine which backend and model to use
    if task_type == 'code_generation':
        selected_backend = config.code_backend
        selected_model = model if model else config.code_model
        logger.info(f"Hybrid: Routing code generation task to {selected_backend} ({selected_model})")
    elif task_type == 'analysis':
        selected_backend = config.analysis_backend or config.default_backend
        selected_model = model if model else config.analysis_model
        logger.info(f"Hybrid: Routing analysis task to {selected_backend} ({selected_model})")
    else:
        selected_backend = config.default_backend
        selected_model = model if model else config.default_model
        logger.info(f"Hybrid: Routing general task to {selected_backend} ({selected_model})")
    
    # Add task type for performance monitoring
    model_kwargs['task_type'] = task_type
    
    # Make the query using the base query function
    result = base_query(
        system_message=system_message,
        user_message=user_message,
        model=selected_model,
        backend=selected_backend,
        temperature=temperature,
        max_tokens=max_tokens,
        func_spec=func_spec,
        **model_kwargs
    )
    
    # base_query returns just the output, but backends return tuple
    # We need to match the backend interface
    if isinstance(result, tuple):
        return result
    else:
        # Construct a proper response tuple
        # This is a fallback - normally base_query should return proper output
        return result, 0.0, 0, 0, {"backend": selected_backend, "model": selected_model}


def configure_hybrid_backend(code_backend: str = "claude_code",
                           code_model: str = "claude-opus-4",
                           analysis_backend: Optional[str] = None,
                           analysis_model: str = "gpt-4o",
                           default_backend: Optional[str] = None,
                           default_model: str = "gpt-4o"):
    """Convenience function to configure the hybrid backend.
    
    Example:
        configure_hybrid_backend(
            code_backend="claude_code",
            code_model="claude-opus-4",
            analysis_backend="openai",
            analysis_model="gpt-4o"
        )
    """
    config = HybridConfig(
        code_backend=code_backend,
        code_model=code_model,
        analysis_backend=analysis_backend,
        analysis_model=analysis_model,
        default_backend=default_backend,
        default_model=default_model
    )
    set_hybrid_config(config)