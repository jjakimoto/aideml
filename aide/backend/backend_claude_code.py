"""Backend for Claude Code SDK."""

import asyncio
import json
import logging
import os
import time
import tempfile
from typing import Any, Dict, Optional, Tuple
from pathlib import Path

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
    use_mcp: bool = False,
) -> str:
    """Format the prompt for Claude Code, including function spec if provided."""
    prompt_parts = []
    
    if system_message:
        prompt_parts.append(f"System: {system_message}")
    
    if func_spec and not use_mcp:
        # Include function specification in the prompt (when not using MCP)
        prompt_parts.append(
            f"\nYou must call the function '{func_spec.name}' with the following schema:\n"
            f"{json.dumps(func_spec.json_schema, indent=2)}\n"
            f"Description: {func_spec.description}"
        )
    elif func_spec and use_mcp:
        # When using MCP, include instructions to use the tool
        prompt_parts.append(
            f"\nPlease use the tool '/mcp__aide__call_{func_spec.name}' to complete this task.\n"
            f"The tool expects parameters matching this schema:\n"
            f"{json.dumps(func_spec.json_schema, indent=2)}"
        )
    
    if user_message:
        prompt_parts.append(f"\nUser: {user_message}")
    
    return "\n".join(prompt_parts)


def _estimate_claude_tokens(text: str) -> int:
    """Estimate token count for Claude models using character-based heuristics.
    
    Claude's tokenizer tends to produce more tokens than GPT models.
    Based on empirical data and research:
    - Claude averages ~3.5-4 characters per token
    - Claude tokenizer has 65,000 token variations (vs GPT's 100,261)
    - Claude typically produces 20-30% more tokens than GPT for same text
    
    This estimation provides better accuracy than simple word count.
    """
    if not text:
        return 0
    
    # Calculate base estimate using character count
    # Using 3.7 chars/token as average for Claude (between 3.5-4.0)
    char_count = len(text)
    base_estimate = char_count / 3.7
    
    # Adjust for special characters and formatting that increase tokenization
    # Count special elements that typically increase token count
    newlines = text.count('\n')
    punctuation = sum(1 for c in text if c in '.,!?;:()[]{}"\'-')
    spaces = text.count(' ')
    
    # Each newline and punctuation typically adds fractional tokens
    special_adjustment = (newlines * 0.3) + (punctuation * 0.1)
    
    # Final estimate with rounding
    total_estimate = base_estimate + special_adjustment
    
    return int(round(total_estimate))


def _convert_func_spec_to_mcp_tool(func_spec: FunctionSpec) -> str:
    """Convert FunctionSpec to MCP tool name."""
    return f"mcp__aide__call_{func_spec.name}"


def _create_mcp_config_for_func_spec(
    func_spec: FunctionSpec, 
    config_path: Optional[str] = None,
    use_advanced: bool = False,
    http_mode: bool = False,
    port: int = 8080
) -> Dict[str, Any]:
    """Create MCP configuration for a function spec with advanced options.
    
    Args:
        func_spec: Function specification
        config_path: Optional path to save config file
        use_advanced: Use advanced MCP server with enhanced features
        http_mode: Use HTTP mode instead of stdio
        port: Port for HTTP mode (default: 8080)
    """
    # Choose the appropriate MCP server module
    server_module = "aide.backend.mcp_server_advanced" if use_advanced else "aide.backend.mcp_server"
    
    # Build server arguments
    args = ["-m", server_module]
    
    if use_advanced:
        args.extend(["--mode", "http" if http_mode else "stdio"])
        if http_mode:
            args.extend(["--port", str(port)])
        # Add verbose flag for better debugging in advanced mode
        args.append("--verbose")
    else:
        # Basic server only supports stdio
        args.extend(["--mode", "stdio"])
    
    # MCP configuration that defines an AIDE ML tool server
    mcp_config = {
        "servers": {
            "aide": {
                "command": "python",
                "args": args,
                "tools": {
                    f"call_{func_spec.name}": {
                        "description": func_spec.description,
                        "inputSchema": func_spec.json_schema
                    }
                }
            }
        }
    }
    
    # Add advanced server metadata if using advanced mode
    if use_advanced:
        mcp_config["servers"]["aide"]["metadata"] = {
            "version": "2.0.0",
            "capabilities": ["validation", "async", "stats", "middleware"],
            "mode": "http" if http_mode else "stdio"
        }
    
    if config_path:
        # Save the configuration to a file
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(mcp_config, f, indent=2)
    
    return mcp_config


def _extract_mcp_tool_call(messages: list, func_spec: FunctionSpec) -> Any:
    """Extract MCP tool call from messages."""
    if not messages:
        return None
    
    expected_tool = _convert_func_spec_to_mcp_tool(func_spec)
    for message in messages:
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.get('tool') == expected_tool:
                    return tool_call.get('input', {})
    return None


def _extract_json_from_text(response_text: str) -> Any:
    """Extract JSON data from text response with multiple fallback strategies."""
    import re
    
    # Strategy 1: Look for JSON blocks in code fences
    json_pattern = r"```json\s*(.*?)\s*```"
    json_matches = re.findall(json_pattern, response_text, re.DOTALL)
    
    if json_matches:
        for match in json_matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    # Strategy 2: Try to parse the entire response as JSON
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Return the original text if no valid JSON found
    return response_text


def _extract_function_call(response_text: str, func_spec: FunctionSpec, messages: list = None) -> Any:
    """Extract function call from Claude Code response.
    
    Uses a prioritized extraction strategy:
    1. MCP tool calls from messages (if available)
    2. JSON extraction from response text
    """
    # First try to extract MCP tool call
    mcp_result = _extract_mcp_tool_call(messages, func_spec)
    if mcp_result is not None:
        return mcp_result
    
    # Fallback to text-based JSON extraction
    return _extract_json_from_text(response_text)


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
    
    Additional model_kwargs:
        use_mcp (bool): Enable MCP (Model Context Protocol) for function calling.
                       When True and func_spec is provided, uses MCP tools instead
                       of text-based function specification. Default: False.
        mcp_config_path (str): Path to custom MCP configuration file. If not provided
                              and use_mcp is True, a temporary config will be created.
    """
    # Extract Claude Code specific options
    provider = model_kwargs.pop("provider", "subscription")  # Default to subscription
    max_turns = model_kwargs.pop("max_turns", 1)
    temperature = model_kwargs.pop("temperature", 0.2)
    model = model_kwargs.pop("model", "claude-opus-4")
    use_mcp = model_kwargs.pop("use_mcp", False)  # MCP opt-in flag
    mcp_config_path = model_kwargs.pop("mcp_config_path", None)  # Custom MCP config path
    use_advanced_mcp = model_kwargs.pop("use_advanced_mcp", False)
    mcp_http_mode = model_kwargs.pop("mcp_http_mode", False)
    mcp_http_port = model_kwargs.pop("mcp_http_port", 8080)
    
    # Set up authentication
    _setup_claude_code_auth(provider)
    
    # Format the prompt
    prompt = _format_prompt_for_claude_code(system_message, user_message, func_spec, use_mcp)
    
    logger.info(f"Claude Code API request: provider={provider}, model={model}")
    logger.debug(f"Prompt: {prompt}")
    
    # Create options for Claude Code
    options_dict = {
        "max_turns": max_turns,
    }
    
    # Add MCP configuration if enabled and func_spec is provided
    if use_mcp and func_spec:
        if mcp_config_path:
            # Use provided MCP config path
            options_dict["mcp_config"] = mcp_config_path
            # Create the MCP config file if it doesn't exist
            if not Path(mcp_config_path).exists():
                _create_mcp_config_for_func_spec(
                    func_spec, mcp_config_path,
                    use_advanced=use_advanced_mcp,
                    http_mode=mcp_http_mode,
                    port=mcp_http_port
                )
        else:
            # Use temporary MCP config
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                temp_config_path = tmp_file.name
                _create_mcp_config_for_func_spec(
                    func_spec, temp_config_path,
                    use_advanced=use_advanced_mcp,
                    http_mode=mcp_http_mode,
                    port=mcp_http_port
                )
                options_dict["mcp_config"] = temp_config_path
        
        # Allow the specific MCP tool
        options_dict["allowed_tools"] = [_convert_func_spec_to_mcp_tool(func_spec)]
        
        logger.info(f"MCP enabled for function: {func_spec.name}")
    
    options = ClaudeCodeOptions(**options_dict)
    
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
    
    # Execute the async function with better event loop handling
    # Avoid creating new event loops when possible
    try:
        # Check if there's already a running event loop
        loop = asyncio.get_running_loop()
        # We're in an async context, but need to run sync
        # This shouldn't happen in normal AIDE ML usage
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, run_query())
            response_text, messages = future.result()
    except RuntimeError:
        # No running event loop, safe to use asyncio.run()
        # This is more efficient than creating a new event loop
        response_text, messages = asyncio.run(run_query())
    
    req_time = time.time() - t0
    
    # Process the output
    output: OutputType
    if func_spec is not None:
        # Try to extract function call from response
        output = _extract_function_call(response_text, func_spec, messages)
    else:
        output = response_text
    
    # Estimate token counts (Claude Code SDK doesn't provide exact counts)
    # Using a more accurate character-based estimation for Claude models
    # Based on empirical data: Claude averages ~3.5-4 characters per token
    # This accounts for Claude's tokenizer producing more tokens than GPT models
    in_tokens = _estimate_claude_tokens(prompt)
    out_tokens = _estimate_claude_tokens(response_text)
    
    info = {
        "provider": provider,
        "model": model,
        "max_turns": max_turns,
        "temperature": temperature,
        "mcp_enabled": use_mcp and func_spec is not None,
        "mcp_advanced": use_advanced_mcp if use_mcp else False,
        "mcp_mode": "http" if mcp_http_mode else "stdio" if use_mcp else None,
    }
    
    # Clean up temporary MCP config if created
    if use_mcp and func_spec and not mcp_config_path and "mcp_config" in options_dict:
        try:
            os.unlink(options_dict["mcp_config"])
        except Exception:
            pass  # Best effort cleanup
    
    logger.info(
        f"Claude Code API call completed - {model} - {req_time:.2f}s - "
        f"~{int(in_tokens + out_tokens)} tokens (in: ~{int(in_tokens)}, out: ~{int(out_tokens)})"
    )
    logger.debug(f"Claude Code API response: {output}")
    
    return output, req_time, int(in_tokens), int(out_tokens), info


async def query_async(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> Tuple[OutputType, float, int, int, Dict[str, Any]]:
    """Async version of Claude Code backend query.
    
    This provides full async support without creating new event loops.
    Use this when calling from async contexts.
    
    Args:
        system_message: System prompt
        user_message: User prompt  
        func_spec: Optional function specification for tool use
        **model_kwargs: Additional arguments including:
            - model: Model name (default: claude-opus-4)
            - provider: Auth provider (default: subscription)
            - temperature: Sampling temperature
            - max_turns: Maximum conversation turns
            - use_mcp: Enable MCP for function calling
            - mcp_config_path: Path to MCP config file
    
    Returns:
        Tuple of (output, request_time, input_tokens, output_tokens, info)
    """
    # Extract parameters
    model = model_kwargs.get("model", "claude-opus-4")
    provider = model_kwargs.get("provider", "subscription")
    temperature = model_kwargs.get("temperature", 0.0)
    max_turns = model_kwargs.get("max_turns", 3)
    use_mcp = model_kwargs.get("use_mcp", False)
    mcp_config_path = model_kwargs.get("mcp_config_path")
    use_advanced_mcp = model_kwargs.get("use_advanced_mcp", False)
    mcp_http_mode = model_kwargs.get("mcp_http_mode", False)
    mcp_http_port = model_kwargs.get("mcp_http_port", 8080)
    
    # Set up authentication
    _setup_claude_code_auth(provider)
    
    # Format the prompt
    prompt = _format_prompt_for_claude_code(system_message, user_message, func_spec, use_mcp)
    
    logger.info(f"Claude Code API request (async): provider={provider}, model={model}")
    logger.debug(f"Prompt: {prompt}")
    
    # Create options for Claude Code
    options_dict = {
        "max_turns": max_turns,
    }
    
    # Add MCP configuration if enabled and func_spec is provided
    temp_mcp_config = None
    if use_mcp and func_spec:
        if mcp_config_path:
            # Use provided MCP config path
            options_dict["mcp_config"] = mcp_config_path
            # Create the MCP config file if it doesn't exist
            if not Path(mcp_config_path).exists():
                _create_mcp_config_for_func_spec(
                    func_spec, mcp_config_path,
                    use_advanced=use_advanced_mcp,
                    http_mode=mcp_http_mode,
                    port=mcp_http_port
                )
        else:
            # Use temporary MCP config
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                temp_mcp_config = tmp_file.name
                _create_mcp_config_for_func_spec(func_spec, temp_mcp_config)
                options_dict["mcp_config"] = temp_mcp_config
        
        # Allow the specific MCP tool
        options_dict["allowed_tools"] = [_convert_func_spec_to_mcp_tool(func_spec)]
        
        logger.info(f"MCP enabled for function: {func_spec.name}")
    
    options = ClaudeCodeOptions(**options_dict)
    
    # Track timing
    t0 = time.time()
    
    # Run the async query directly (no event loop needed)
    messages = []
    response_text = ""
    
    try:
        async for message in claude_query(prompt=prompt, options=options):
            messages.append(message)
            # Accumulate response text from messages
            if hasattr(message, "content"):
                response_text += message.content
            elif hasattr(message, "text"):
                response_text += message.text
    finally:
        # Clean up temporary MCP config if created
        if temp_mcp_config:
            try:
                os.unlink(temp_mcp_config)
            except Exception:
                pass  # Best effort cleanup
    
    req_time = time.time() - t0
    
    # Process the output
    output: OutputType
    if func_spec is not None:
        # Try to extract function call from response
        output = _extract_function_call(response_text, func_spec, messages)
    else:
        output = response_text
    
    # Estimate token counts using improved method
    in_tokens = _estimate_claude_tokens(prompt)
    out_tokens = _estimate_claude_tokens(response_text)
    
    info = {
        "provider": provider,
        "model": model,
        "max_turns": max_turns,
        "temperature": temperature,
        "mcp_enabled": use_mcp and func_spec is not None,
        "mcp_advanced": use_advanced_mcp if use_mcp else False,
        "mcp_mode": "http" if mcp_http_mode else "stdio" if use_mcp else None,
        "async": True,  # Mark as async execution
    }
    
    logger.info(
        f"Claude Code API call completed (async) - {model} - {req_time:.2f}s - "
        f"~{int(in_tokens + out_tokens)} tokens (in: ~{int(in_tokens)}, out: ~{int(out_tokens)})"
    )
    logger.debug(f"Claude Code API response: {output}")
    
    return output, req_time, int(in_tokens), int(out_tokens), info