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


def _convert_func_spec_to_mcp_tool(func_spec: FunctionSpec) -> str:
    """Convert FunctionSpec to MCP tool name."""
    return f"mcp__aide__call_{func_spec.name}"


def _create_mcp_config_for_func_spec(func_spec: FunctionSpec, config_path: Optional[str] = None) -> Dict[str, Any]:
    """Create MCP configuration for a function spec."""
    # MCP configuration that defines an AIDE ML tool server
    mcp_config = {
        "servers": {
            "aide": {
                "command": "aide-mcp-server",  # Hypothetical MCP server for AIDE ML
                "args": ["--mode", "function-call"],
                "tools": {
                    f"call_{func_spec.name}": {
                        "description": func_spec.description,
                        "inputSchema": func_spec.json_schema
                    }
                }
            }
        }
    }
    
    if config_path:
        # Save the configuration to a file
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(mcp_config, f, indent=2)
    
    return mcp_config


def _extract_function_call(response_text: str, func_spec: FunctionSpec, messages: list = None) -> Any:
    """Extract function call from Claude Code response."""
    # First check if we have MCP tool calls in the messages
    if messages:
        for message in messages:
            # Check for MCP tool usage in the message
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.get('tool') == _convert_func_spec_to_mcp_tool(func_spec):
                        # Return the tool call parameters
                        return tool_call.get('input', {})
    
    # Fallback to text extraction if no MCP tool calls found
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
                _create_mcp_config_for_func_spec(func_spec, mcp_config_path)
        else:
            # Use temporary MCP config
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                temp_config_path = tmp_file.name
                _create_mcp_config_for_func_spec(func_spec, temp_config_path)
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
        output = _extract_function_call(response_text, func_spec, messages)
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
        "mcp_enabled": use_mcp and func_spec is not None,
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