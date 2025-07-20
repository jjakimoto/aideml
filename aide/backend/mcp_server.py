#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Server for AIDE ML.

This server exposes AIDE ML's function calling capabilities through MCP,
allowing Claude Code to invoke functions defined by FunctionSpec objects.

Note: This is a basic MCP server implementation that provides proof-of-concept
functionality for AIDE ML function calling. It currently supports:
- stdio mode communication
- tools/list and tools/call methods
- Basic function registration and execution

Limitations:
- HTTP mode not implemented
- Single function support (registers demo function by default)
- Basic error handling
"""

import argparse
import json
import logging
import sys
from typing import Any, Dict, Optional

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
        """Handle an incoming MCP request with robust error handling."""
        try:
            method = request.get("method")
            params = request.get("params", {})
            
            if method == "tools/list":
                return self._handle_tools_list()
            elif method == "tools/call":
                return self._handle_tools_call(params)
            else:
                return {"error": {"code": -32601, "message": f"Method not found: {method}"}}
                
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return {"error": {"code": -32603, "message": f"Internal error: {str(e)}"}}
    
    def _handle_tools_list(self) -> Dict[str, Any]:
        """Handle tools/list request."""
        tools = []
        for name, func_info in self.functions.items():
            tools.append({
                "name": f"call_{name}",
                "description": func_info["schema"].get("description", f"Call {name} function"),
                "inputSchema": func_info["schema"]
            })
        return {"tools": tools}
    
    def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name", "")
        if not tool_name.startswith("call_"):
            return {"error": {"code": -32602, "message": f"Invalid tool name format: {tool_name}"}}
        
        # Extract function name
        func_name = tool_name[5:]  # Remove "call_" prefix
        tool_input = params.get("arguments", {})
        
        if func_name not in self.functions:
            return {"error": {"code": -32602, "message": f"Unknown tool: {tool_name}"}}
        
        try:
            handler = self.functions[func_name]["handler"]
            result = handler(tool_input)
            return {"content": [{"type": "text", "text": json.dumps(result)}]}
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"error": {"code": -32603, "message": f"Tool execution failed: {str(e)}"}}
    
    def run_stdio(self):
        """Run the MCP server in stdio mode with robust error handling."""
        logger.info("Starting AIDE ML MCP server in stdio mode")
        
        while True:
            try:
                # Read JSON-RPC request from stdin
                line = sys.stdin.readline()
                if not line:
                    logger.info("EOF received, shutting down server")
                    break
                
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                
                try:
                    request = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON request: {e}")
                    self._send_error_response(None, -32700, "Parse error")
                    continue
                
                # Handle the request
                response_data = self.handle_request(request)
                
                # Send JSON-RPC response
                self._send_response(request.get("id"), response_data)
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down")
                break
            except Exception as e:
                logger.error(f"Unexpected error in stdio loop: {e}")
                self._send_error_response(None, -32603, f"Internal error: {str(e)}")
    
    def _send_response(self, request_id: Optional[Any], response_data: Dict[str, Any]):
        """Send a JSON-RPC response."""
        if "error" in response_data:
            # This is an error response
            json_response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": response_data["error"]
            }
        else:
            # This is a success response
            json_response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": response_data
            }
        
        try:
            print(json.dumps(json_response))
            sys.stdout.flush()
        except Exception as e:
            logger.error(f"Error sending response: {e}")
    
    def _send_error_response(self, request_id: Optional[Any], code: int, message: str):
        """Send a JSON-RPC error response."""
        json_response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }
        
        try:
            print(json.dumps(json_response))
            sys.stdout.flush()
        except Exception as e:
            logger.error(f"Error sending error response: {e}")


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