#!/usr/bin/env python3
"""
Advanced MCP (Model Context Protocol) Server for AIDE ML.

This enhanced server provides comprehensive MCP support with:
- Multiple function registration and management
- HTTP mode support (in addition to stdio)
- Function parameter validation
- Async handler support
- Enhanced error handling and logging
- Function metadata and versioning
"""

import argparse
import asyncio
import json
import logging
import sys
from typing import Any, Dict, Optional, List, Callable, Union
from pathlib import Path
import jsonschema
from aiohttp import web
import uvloop  # Optional: high-performance event loop

logger = logging.getLogger("aide.mcp.advanced")


class AdvancedMCPServer:
    """Enhanced MCP server with advanced features."""
    
    def __init__(self, name: str = "aide-ml-mcp", version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.functions = {}
        self.function_metadata = {}
        self.middleware = []
        self.stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "function_calls": {}
        }
    
    def register_function(
        self,
        name: str,
        schema: dict,
        handler: Optional[Callable] = None,
        description: str = "",
        version: str = "1.0.0",
        tags: List[str] = None,
        validate_params: bool = True
    ):
        """Register a function with enhanced metadata and validation."""
        self.functions[name] = {
            "schema": schema,
            "handler": handler or self._default_handler,
            "validate_params": validate_params
        }
        
        self.function_metadata[name] = {
            "description": description,
            "version": version,
            "tags": tags or [],
            "call_count": 0,
            "last_called": None
        }
        
        logger.info(f"Registered function: {name} (version: {version})")
    
    def register_middleware(self, middleware: Callable):
        """Register middleware for request/response processing."""
        self.middleware.append(middleware)
    
    async def _default_handler(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Default async handler that echoes parameters."""
        return {
            "status": "success",
            "result": params,
            "metadata": {
                "handler": "default",
                "timestamp": asyncio.get_event_loop().time()
            }
        }
    
    def _validate_params(self, params: Dict[str, Any], schema: dict) -> Optional[str]:
        """Validate parameters against JSON schema."""
        try:
            jsonschema.validate(params, schema)
            return None
        except jsonschema.ValidationError as e:
            return str(e)
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an incoming MCP request with enhanced processing."""
        method = request.get("method")
        params = request.get("params", {})
        
        # Apply middleware
        for mw in self.middleware:
            request = await mw(request) if asyncio.iscoroutinefunction(mw) else mw(request)
        
        # Update stats
        self.stats["total_calls"] += 1
        
        try:
            if method == "tools/list":
                result = await self._handle_tools_list(params)
            elif method == "tools/call":
                result = await self._handle_tools_call(params)
            elif method == "tools/info":
                result = await self._handle_tools_info(params)
            elif method == "server/info":
                result = await self._handle_server_info(params)
            elif method == "server/stats":
                result = await self._handle_server_stats(params)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            self.stats["successful_calls"] += 1
            return result
            
        except Exception as e:
            self.stats["failed_calls"] += 1
            logger.error(f"Error handling request: {e}")
            return {
                "error": {
                    "code": -32603,
                    "message": str(e),
                    "data": {
                        "method": method,
                        "params": params
                    }
                }
            }
    
    async def _handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List available tools with enhanced metadata."""
        tools = []
        
        for name, func_info in self.functions.items():
            metadata = self.function_metadata.get(name, {})
            tool = {
                "name": f"mcp__aide__call_{name}",
                "description": metadata.get("description", f"AIDE ML function: {name}"),
                "inputSchema": func_info["schema"],
                "metadata": {
                    "version": metadata.get("version", "1.0.0"),
                    "tags": metadata.get("tags", []),
                    "call_count": metadata.get("call_count", 0)
                }
            }
            tools.append(tool)
        
        return {
            "tools": tools,
            "metadata": {
                "server": self.name,
                "version": self.version,
                "function_count": len(tools)
            }
        }
    
    async def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool with enhanced validation and error handling."""
        tool_name = params.get("name", "")
        tool_params = params.get("arguments", {})
        
        # Extract function name from MCP tool name
        if tool_name.startswith("mcp__aide__call_"):
            func_name = tool_name[len("mcp__aide__call_"):]
        else:
            raise ValueError(f"Invalid tool name format: {tool_name}")
        
        if func_name not in self.functions:
            raise ValueError(f"Unknown function: {func_name}")
        
        func_info = self.functions[func_name]
        
        # Validate parameters if enabled
        if func_info["validate_params"]:
            error = self._validate_params(tool_params, func_info["schema"])
            if error:
                raise ValueError(f"Parameter validation failed: {error}")
        
        # Update function metadata
        import datetime
        metadata = self.function_metadata.get(func_name, {})
        metadata["call_count"] = metadata.get("call_count", 0) + 1
        metadata["last_called"] = datetime.datetime.now().isoformat()
        
        # Update stats
        self.stats["function_calls"][func_name] = self.stats["function_calls"].get(func_name, 0) + 1
        
        # Call the handler (support both sync and async)
        handler = func_info["handler"]
        if asyncio.iscoroutinefunction(handler):
            result = await handler(tool_params)
        else:
            result = handler(tool_params)
        
        return result
    
    async def _handle_tools_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed information about a specific tool."""
        tool_name = params.get("name", "")
        
        # Extract function name
        if tool_name.startswith("mcp__aide__call_"):
            func_name = tool_name[len("mcp__aide__call_"):]
        else:
            raise ValueError(f"Invalid tool name format: {tool_name}")
        
        if func_name not in self.functions:
            raise ValueError(f"Unknown function: {func_name}")
        
        func_info = self.functions[func_name]
        metadata = self.function_metadata.get(func_name, {})
        
        return {
            "name": tool_name,
            "function_name": func_name,
            "schema": func_info["schema"],
            "metadata": metadata,
            "validation_enabled": func_info["validate_params"]
        }
    
    async def _handle_server_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get server information."""
        return {
            "name": self.name,
            "version": self.version,
            "capabilities": [
                "tools/list",
                "tools/call",
                "tools/info",
                "server/info",
                "server/stats"
            ],
            "modes": ["stdio", "http"],
            "function_count": len(self.functions),
            "middleware_count": len(self.middleware)
        }
    
    async def _handle_server_stats(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            "stats": self.stats,
            "uptime": asyncio.get_event_loop().time(),
            "function_metadata": self.function_metadata
        }
    
    async def run_stdio(self):
        """Run the server in stdio mode with async support."""
        logger.info(f"Starting {self.name} MCP server (v{self.version}) in stdio mode")
        
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)
        
        while True:
            try:
                # Read JSON-RPC request from stdin
                line = await reader.readline()
                if not line:
                    logger.info("EOF received, shutting down server")
                    break
                
                line = line.decode().strip()
                if not line:
                    continue
                
                try:
                    request = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON request: {e}")
                    await self._send_error_response(None, -32700, "Parse error")
                    continue
                
                # Handle the request
                response_data = await self.handle_request(request)
                
                # Send JSON-RPC response
                await self._send_response(request.get("id"), response_data)
                
            except asyncio.CancelledError:
                logger.info("Server cancelled, shutting down")
                break
            except Exception as e:
                logger.error(f"Unexpected error in stdio loop: {e}")
                await self._send_error_response(None, -32603, f"Internal error: {str(e)}")
    
    async def _send_response(self, request_id: Optional[Any], response_data: Dict[str, Any]):
        """Send an async JSON-RPC response."""
        if "error" in response_data:
            json_response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": response_data["error"]
            }
        else:
            json_response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": response_data
            }
        
        print(json.dumps(json_response))
        sys.stdout.flush()
    
    async def _send_error_response(self, request_id: Optional[Any], code: int, message: str):
        """Send an error response."""
        await self._send_response(request_id, {
            "error": {
                "code": code,
                "message": message
            }
        })
    
    def create_http_app(self) -> web.Application:
        """Create aiohttp application for HTTP mode."""
        app = web.Application()
        app.router.add_post('/mcp', self._handle_http_request)
        app.router.add_get('/health', self._handle_health_check)
        app.router.add_get('/stats', self._handle_stats_endpoint)
        return app
    
    async def _handle_http_request(self, request: web.Request) -> web.Response:
        """Handle HTTP MCP requests."""
        try:
            data = await request.json()
            response_data = await self.handle_request(data)
            
            if "error" in response_data:
                return web.json_response({
                    "jsonrpc": "2.0",
                    "id": data.get("id"),
                    "error": response_data["error"]
                }, status=400)
            else:
                return web.json_response({
                    "jsonrpc": "2.0",
                    "id": data.get("id"),
                    "result": response_data
                })
        except Exception as e:
            logger.error(f"HTTP request error: {e}")
            return web.json_response({
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32603,
                    "message": str(e)
                }
            }, status=500)
    
    async def _handle_health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({
            "status": "healthy",
            "server": self.name,
            "version": self.version,
            "functions": len(self.functions)
        })
    
    async def _handle_stats_endpoint(self, request: web.Request) -> web.Response:
        """Statistics endpoint."""
        stats_data = await self._handle_server_stats({})
        return web.json_response(stats_data)
    
    async def run_http(self, host: str = "0.0.0.0", port: int = 8080):
        """Run the server in HTTP mode."""
        logger.info(f"Starting {self.name} MCP server (v{self.version}) in HTTP mode on {host}:{port}")
        
        app = self.create_http_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        
        logger.info(f"HTTP server started on http://{host}:{port}")
        
        # Keep the server running
        try:
            await asyncio.Future()  # Run forever
        except asyncio.CancelledError:
            logger.info("HTTP server cancelled, shutting down")
            await runner.cleanup()


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stderr)
        ]
    )


async def main():
    """Main entry point with example usage."""
    parser = argparse.ArgumentParser(description="Advanced AIDE ML MCP Server")
    parser.add_argument("--mode", choices=["stdio", "http"], default="stdio",
                        help="Server mode (default: stdio)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="HTTP host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080,
                        help="HTTP port (default: 8080)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--use-uvloop", action="store_true",
                        help="Use uvloop for better performance")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Use uvloop if requested
    if args.use_uvloop:
        try:
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            logger.info("Using uvloop for better performance")
        except ImportError:
            logger.warning("uvloop not available, using default event loop")
    
    # Create server
    server = AdvancedMCPServer(
        name="aide-ml-advanced",
        version="2.0.0"
    )
    
    # Register example functions
    server.register_function(
        name="analyze_data",
        schema={
            "type": "object",
            "properties": {
                "data": {"type": "array", "items": {"type": "number"}},
                "method": {"type": "string", "enum": ["mean", "median", "std", "min", "max"]}
            },
            "required": ["data", "method"]
        },
        description="Analyze numerical data with various statistical methods",
        version="1.0.0",
        tags=["statistics", "analysis"]
    )
    
    server.register_function(
        name="transform_text",
        schema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "operation": {"type": "string", "enum": ["uppercase", "lowercase", "reverse", "capitalize"]}
            },
            "required": ["text", "operation"]
        },
        description="Transform text with various operations",
        version="1.0.0",
        tags=["text", "transformation"]
    )
    
    # Example middleware
    async def logging_middleware(request):
        logger.debug(f"Request: {request}")
        return request
    
    server.register_middleware(logging_middleware)
    
    # Run server
    if args.mode == "stdio":
        await server.run_stdio()
    else:
        await server.run_http(args.host, args.port)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)