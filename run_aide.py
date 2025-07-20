#!/usr/bin/env python3
"""
Runner script for AIDE without installing via setup.py
This script adds the current directory to Python path and runs the aide module
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the main function
from aide.run import run

if __name__ == "__main__":
    run()