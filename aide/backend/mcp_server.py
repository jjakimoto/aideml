#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Server for AIDE ML.

This server exposes AIDE ML's function calling capabilities through MCP,
allowing Claude Code to invoke functions defined by FunctionSpec objects.
"""

import argparse
import json
import logging
import sys
from typing import Any, Dict

logger = logging.getLogger("aide.mcp")


class AideMCPServer:
    """MCP server that handles function calls for AIDE ML."""
    
    def __init__(self):
        self.functions = {}
    
    def register_function(self, name: str, schema: dict, handler=None):
        """Register a function that can be called through MCP."""
        self.functions[name] = {
            "schema": schema,
            "handler": handler or self._default_handler
        }
    
    def _default_handler(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Default handler that echoes the parameters."""
        return {
            "status": "success",
            "result": params
        }
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an incoming MCP request."""
        method = request.get("method")
        params = request.get("params", {})
        
        if method == "tools/list":
            # Return available tools
            tools = []
            for name, func_info in self.functions.items():
                tools.append({
                    "name": f"call_{name}",
                    "description": func_info["schema"].get("description", f"Call {name} function"),
                    "inputSchema": func_info["schema"]
                })
            return {"tools": tools}
        
        elif method == "tools/call":
            # Execute a tool
            tool_name = params.get("name", "").replace("call_", "", 1)
            tool_input = params.get("arguments", {})
            
            if tool_name in self.functions:
                handler = self.functions[tool_name]["handler"]
                try:
                    result = handler(tool_input)
                    return {"content": [{"type": "text", "text": json.dumps(result)}]}
                except Exception as e:
                    return {"error": {"message": str(e)}}
            else:
                return {"error": {"message": f"Unknown tool: {tool_name}"}}
        
        else:
            return {"error": {"message": f"Unknown method: {method}"}}
    
    def run_stdio(self):
        """Run the MCP server in stdio mode."""
        logger.info("Starting AIDE ML MCP server in stdio mode")
        
        while True:
            try:
                # Read JSON-RPC request from stdin
                line = sys.stdin.readline()
                if not line:
                    break
                
                request = json.loads(line.strip())
                response = self.handle_request(request)
                
                # Write JSON-RPC response to stdout
                json_response = json.dumps({
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": response
                })
                print(json_response)
                sys.stdout.flush()
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
            except Exception as e:
                logger.error(f"Error handling request: {e}")


def main():
    """Main entry point for the MCP server."""
    parser = argparse.ArgumentParser(description="AIDE ML MCP Server")
    parser.add_argument("--mode", default="stdio", choices=["stdio", "http"],
                        help="Server mode (stdio or http)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    server = AideMCPServer()
    
    # In function-call mode, we would register functions based on
    # the FunctionSpec passed from AIDE ML
    # For now, register a demo function
    server.register_function(
        "example_function",
        {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Input parameter"}
            },
            "required": ["input"]
        }
    )
    
    if args.mode == "stdio":
        server.run_stdio()
    else:
        # HTTP mode not implemented yet
        raise NotImplementedError("HTTP mode not yet implemented")


if __name__ == "__main__":
    main()