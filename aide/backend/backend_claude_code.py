"""Backend for Claude Code SDK."""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

from claude_code_sdk import query as claude_query, ClaudeCodeOptions
from .utils import FunctionSpec, OutputType, opt_messages_to_list

logger = logging.getLogger("aide")

CLAUDE_CODE_PROVIDERS = {
    "subscription": {"env_var": None, "use_env": None},  # Uses Claude Code subscription auth
    "anthropic": {"env_var": "ANTHROPIC_API_KEY", "use_env": None},
    "bedrock": {"env_var": None, "use_env": "CLAUDE_CODE_USE_BEDROCK"},
    "vertex": {"env_var": None, "use_env": "CLAUDE_CODE_USE_VERTEX"},
}


def _setup_claude_code_auth(provider: str = "subscription"):
    """Set up authentication for Claude Code based on provider.
    
    Provider options:
    - 'subscription': Uses Claude Code subscription (no API key needed)
    - 'anthropic': Uses Anthropic API key
    - 'bedrock': Uses AWS Bedrock credentials
    - 'vertex': Uses Google Vertex AI credentials
    """
    if provider not in CLAUDE_CODE_PROVIDERS:
        raise ValueError(
            f"Unknown provider: {provider}. Must be one of {list(CLAUDE_CODE_PROVIDERS.keys())}"
        )

    provider_config = CLAUDE_CODE_PROVIDERS[provider]
    
    # Set provider-specific environment variable
    if provider_config["use_env"]:
        os.environ[provider_config["use_env"]] = "1"
        logger.info(f"Claude Code configured to use {provider}")
    
    # Check for API key if needed (optional for anthropic provider)
    if provider_config["env_var"] and provider != "anthropic":
        # For non-anthropic providers that require env vars, they must be set
        if not os.getenv(provider_config["env_var"]):
            raise ValueError(
                f"Missing required environment variable: {provider_config['env_var']}"
            )
    elif provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        logger.info(
            "No ANTHROPIC_API_KEY found. Claude Code will use subscription authentication."
        )
    elif provider == "subscription":
        logger.info("Using Claude Code subscription authentication (no API key required)")


def _format_prompt_for_claude_code(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
) -> str:
    """Format the prompt for Claude Code, including function spec if provided."""
    prompt_parts = []
    
    if system_message:
        prompt_parts.append(f"System: {system_message}")
    
    if func_spec:
        # Include function specification in the prompt
        prompt_parts.append(
            f"\nYou must call the function '{func_spec.name}' with the following schema:\n"
            f"{json.dumps(func_spec.json_schema, indent=2)}\n"
            f"Description: {func_spec.description}"
        )
    
    if user_message:
        prompt_parts.append(f"\nUser: {user_message}")
    
    return "\n".join(prompt_parts)


async def _extract_function_call(response_text: str, func_spec: FunctionSpec) -> Any:
    """Extract function call from Claude Code response."""
    # Claude Code might return function calls in various formats
    # Try to extract JSON that matches our function schema
    
    # Look for JSON blocks in the response
    import re
    json_pattern = r"```json\s*(.*?)\s*```"
    json_matches = re.findall(json_pattern, response_text, re.DOTALL)
    
    if json_matches:
        for match in json_matches:
            try:
                data = json.loads(match)
                # Validate against function schema if needed
                return data
            except json.JSONDecodeError:
                continue
    
    # Try to parse the entire response as JSON
    try:
        data = json.loads(response_text)
        return data
    except json.JSONDecodeError:
        pass
    
    # If no valid JSON found, return the text
    return response_text


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """
    Query Claude Code SDK with AIDE ML compatible interface.
    
    This function provides a synchronous wrapper around Claude Code's async API
    and formats the response to match AIDE ML's expected output format.
    """
    # Extract Claude Code specific options
    provider = model_kwargs.pop("provider", "subscription")  # Default to subscription
    max_turns = model_kwargs.pop("max_turns", 1)
    temperature = model_kwargs.pop("temperature", 0.2)
    model = model_kwargs.pop("model", "claude-opus-4")
    
    # Set up authentication
    _setup_claude_code_auth(provider)
    
    # Format the prompt
    prompt = _format_prompt_for_claude_code(system_message, user_message, func_spec)
    
    logger.info(f"Claude Code API request: provider={provider}, model={model}")
    logger.debug(f"Prompt: {prompt}")
    
    # Create options for Claude Code
    options = ClaudeCodeOptions(
        max_turns=max_turns,
        # Add other options as needed based on model_kwargs
    )
    
    # Track timing
    t0 = time.time()
    
    # Run the async query in a sync context
    async def run_query():
        messages = []
        response_text = ""
        
        async for message in claude_query(prompt=prompt, options=options):
            messages.append(message)
            # Accumulate response text from messages
            # The actual message structure depends on Claude Code SDK
            if hasattr(message, "content"):
                response_text += message.content
            elif hasattr(message, "text"):
                response_text += message.text
        
        return response_text, messages
    
    # Execute the async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        response_text, messages = loop.run_until_complete(run_query())
    finally:
        loop.close()
    
    req_time = time.time() - t0
    
    # Process the output
    output: OutputType
    if func_spec is not None:
        # Try to extract function call from response
        output = asyncio.run(_extract_function_call(response_text, func_spec))
    else:
        output = response_text
    
    # Estimate token counts (Claude Code SDK might not provide exact counts)
    # This is a rough estimation based on response length
    in_tokens = len(prompt.split()) * 1.3  # Rough estimate
    out_tokens = len(response_text.split()) * 1.3  # Rough estimate
    
    info = {
        "provider": provider,
        "model": model,
        "max_turns": max_turns,
        "temperature": temperature,
    }
    
    logger.info(
        f"Claude Code API call completed - {model} - {req_time:.2f}s - "
        f"~{int(in_tokens + out_tokens)} tokens (in: ~{int(in_tokens)}, out: ~{int(out_tokens)})"
    )
    logger.debug(f"Claude Code API response: {output}")
    
    return output, req_time, int(in_tokens), int(out_tokens), info