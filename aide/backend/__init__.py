from . import backend_anthropic, backend_openai, backend_openrouter, backend_gemini, backend_claude_code, backend_hybrid
from .utils import FunctionSpec, OutputType, PromptType, compile_prompt_to_md
from ..utils.performance_monitor import get_performance_monitor
import re
import logging
import os
import time

logger = logging.getLogger("aide")


def determine_provider(model: str) -> str:
    # Check if model matches OpenAI patterns first
    if re.match(r"^(gpt-|o\d-|codex-mini-latest$)", model):
        return "openai"
    elif model.startswith("claude-"):
        return "anthropic"
    elif model.startswith("gemini-"):
        return "gemini"
    # If OPENAI_BASE_URL is set, use openai provider for non-standard models
    elif os.getenv("OPENAI_BASE_URL"):
        return "openai"
    # all other models are handle by openrouter
    else:
        return "openrouter"


provider_to_query_func = {
    "openai": backend_openai.query,
    "anthropic": backend_anthropic.query,
    "openrouter": backend_openrouter.query,
    "gemini": backend_gemini.query,
    "claude_code": backend_claude_code.query,
    "hybrid": backend_hybrid.query,
}


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str = None,
    backend: str = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> OutputType:
    """
    General LLM query for various backends with a single system and user message.
    Supports function calling for some backends.

    Args:
        system_message (PromptType | None): Uncompiled system message (will generate a message following the OpenAI/Anthropic format)
        user_message (PromptType | None): Uncompiled user message (will generate a message following the OpenAI/Anthropic format)
        model (str): string identifier for the model to use (e.g. "gpt-4-turbo")
        backend (str | None, optional): Explicitly specify the backend to use (e.g. "claude_code"). If None, backend is determined from model name.
        temperature (float | None, optional): Temperature to sample at. Defaults to the model-specific default.
        max_tokens (int | None, optional): Maximum number of tokens to generate. Defaults to the model-specific max tokens.
        func_spec (FunctionSpec | None, optional): Optional FunctionSpec object defining a function call. If given, the return value will be a dict.

    Returns:
        OutputType: A string completion if func_spec is None, otherwise a dict with the function call details.
    """

    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Determine provider - explicit backend takes precedence
    if backend:
        provider = backend
    else:
        provider = determine_provider(model)
    
    if provider not in provider_to_query_func:
        raise ValueError(f"Unknown provider/backend: {provider}")
    
    # Get performance monitor
    monitor = get_performance_monitor()
    
    # Extract task type from model_kwargs if available
    task_type = model_kwargs.pop('task_type', None)
    
    query_func = provider_to_query_func[provider]
    start_time = time.time()
    success = True
    error = None
    
    try:
        output, req_time, in_tok_count, out_tok_count, info = query_func(
            system_message=compile_prompt_to_md(system_message) if system_message else None,
            user_message=compile_prompt_to_md(user_message) if user_message else None,
            func_spec=func_spec,
            **model_kwargs,
        )
    except Exception as e:
        success = False
        error = str(e)
        raise
    finally:
        end_time = time.time()
        
        # Record performance metrics
        if 'output' in locals():
            total_tokens = in_tok_count + out_tok_count
            monitor.record_query(
                backend=provider,
                model=model,
                start_time=start_time,
                end_time=end_time,
                total_tokens=total_tokens,
                prompt_tokens=in_tok_count,
                completion_tokens=out_tok_count,
                success=success,
                error=error,
                task_type=task_type,
                request_time=req_time,
                has_function_call=func_spec is not None
            )

    return output
