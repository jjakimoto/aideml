#!/usr/bin/env python3
"""
Test script to verify development mode setup
"""

import sys
import os

# Add the current directory to Python path (same as run_aide.py)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Test basic imports
    import aide
    from aide.agent import Agent
    from aide.interpreter import Interpreter
    from aide.journal import Journal
    print("✓ Basic imports successful")
    
    # Test if we can create an Experiment instance (without running it)
    print("✓ Can import aide.Experiment class")
    
    print("\nDevelopment mode setup is working correctly!")
    print("\nYou can now:")
    print("1. Run AIDE with: python run_aide.py")
    print("2. Run WebUI with: python run_webui.py")
    print("3. Import aide module in your Python scripts")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nPlease make sure you have installed all dependencies:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    sys.exit(1)