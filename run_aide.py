#!/usr/bin/env python3
"""
Runner script for AIDE without installing via setup.py
This script adds the current directory to Python path and runs the aide module
"""

import sys
import os
import argparse

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the main function
from aide.run import run

def parse_backend_args():
    """
    Parse --backend and --backend-opt arguments and convert them to OmegaConf format.
    Other arguments are passed through unchanged.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--backend', help='Backend to use (e.g., claude_code, openai, anthropic)')
    parser.add_argument('--backend-opt', action='append', help='Backend options as key=value')
    
    # Parse known args
    args, remaining = parser.parse_known_args()
    
    # Convert to OmegaConf format
    converted_args = []
    
    if args.backend:
        converted_args.append(f'agent.backend={args.backend}')
    
    if args.backend_opt:
        for opt in args.backend_opt:
            if '=' in opt:
                key, value = opt.split('=', 1)
                converted_args.append(f'agent.backend_options.{key}={value}')
    
    # Add remaining args and converted args
    return remaining + converted_args

if __name__ == "__main__":
    # Convert backend args to OmegaConf format
    sys.argv[1:] = parse_backend_args()
    run()