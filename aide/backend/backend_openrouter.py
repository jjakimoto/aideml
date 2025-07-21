"""Backend for OpenRouter API"""

import json
import logging
import os
import time

from funcy import notnone, once, select_values
import openai

from .utils import FunctionSpec, OutputType, backoff_create

logger = logging.getLogger("aide")

_client: openai.OpenAI = None  # type: ignore

OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@once
def _setup_openrouter_client():
    global _client
    _client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        max_retries=0,
    )


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_openrouter_client()
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    # Prepare messages
    # OpenRouter supports system messages, so use them when available
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    if user_message:
        messages.append({"role": "user", "content": user_message})
    
    # Add function calling support
    if func_spec is not None:
        # OpenRouter uses the same format as OpenAI for tools
        filtered_kwargs["tools"] = [{
            "type": "function",
            "function": {
                "name": func_spec.name,
                "description": func_spec.description,
                "parameters": func_spec.json_schema
            }
        }]
        filtered_kwargs["tool_choice"] = {
            "type": "function",
            "function": {"name": func_spec.name}
        }

    logger.info(f"OpenRouter API request: system={system_message}, user={user_message}")

    t0 = time.time()
    completion = backoff_create(
        _client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        extra_body={
            "provider": {
                "order": ["Fireworks"],
                "ignore": ["Together", "DeepInfra", "Hyperbolic"],
            },
        },
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    # Process the response
    message = completion.choices[0].message
    output: OutputType
    
    # Check for function calls
    if func_spec is not None and hasattr(message, 'tool_calls') and message.tool_calls:
        # Function call found
        tool_call = message.tool_calls[0]
        if hasattr(tool_call, 'function') and tool_call.function.name == func_spec.name:
            try:
                output = json.loads(tool_call.function.arguments)
                logger.info(f"OpenRouter function call extracted: {output}")
            except json.JSONDecodeError as ex:
                logger.error(
                    f"Error decoding function arguments:\n{tool_call.function.arguments}"
                )
                raise ex
        else:
            # Function name mismatch or unexpected structure, fall back to text
            logger.warning(
                f"Function name mismatch or unexpected structure. "
                f"Expected: {func_spec.name}, got tool_calls: {message.tool_calls}"
            )
            output = message.content or ""
    else:
        # No function call or no func_spec, use regular text output
        output = message.content or ""

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    logger.info(
        f"OpenRouter API call completed - {completion.model} - {req_time:.2f}s - {in_tokens + out_tokens} tokens (in: {in_tokens}, out: {out_tokens})"
    )
    logger.info(f"OpenRouter API response: {output}")

    return output, req_time, in_tokens, out_tokens, info
