#!/usr/bin/env python3
"""
Runner script for AIDE WebUI without installing via setup.py
This script adds the current directory to Python path and runs the webui
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the webui
from aide.webui.app import WebUI

if __name__ == "__main__":
    app = WebUI()
    app.run()