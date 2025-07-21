#!/usr/bin/env python3
"""
AIDE ML API Key Configuration Helper
This script helps set up API keys for both Claude Code and Anthropic backends
"""

import os
import sys
import subprocess
from pathlib import Path
import json


def check_claude_code_auth():
    """Check if Claude Code SDK is authenticated"""
    try:
        # Try to run a simple claude command to check auth
        result = subprocess.run(
            ["which", "claude"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            return False, "Claude Code CLI not found"
        
        # Check if authenticated (this is a placeholder - actual check may vary)
        # Claude Code SDK should handle auth automatically
        return True, "Claude Code SDK is installed"
    except Exception as e:
        return False, str(e)


def setup_anthropic_key():
    """Set up Anthropic API key from ANTHROPIC_API_KEY_TILDE or prompt user"""
    # Check if ANTHROPIC_API_KEY is already set
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("✅ ANTHROPIC_API_KEY is already set")
        return True
    
    # Check for ANTHROPIC_API_KEY_TILDE
    tilde_key = os.environ.get("ANTHROPIC_API_KEY_TILDE")
    if tilde_key:
        print("📋 Found ANTHROPIC_API_KEY_TILDE, converting to ANTHROPIC_API_KEY")
        os.environ["ANTHROPIC_API_KEY"] = tilde_key
        
        # Also save to .env file for persistence
        env_file = Path.home() / ".aide_ml" / ".env"
        env_file.parent.mkdir(exist_ok=True)
        
        # Read existing .env if it exists
        existing_vars = {}
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.strip().split('=', 1)
                        existing_vars[key] = value
        
        # Update with new key
        existing_vars["ANTHROPIC_API_KEY"] = tilde_key
        
        # Write back
        with open(env_file, 'w') as f:
            for key, value in existing_vars.items():
                f.write(f"{key}={value}\n")
        
        print(f"✅ Saved ANTHROPIC_API_KEY to {env_file}")
        return True
    
    # Prompt user for key
    print("\n🔑 Anthropic API Key Setup")
    print("=" * 40)
    print("To use the Anthropic backend, you need an API key from:")
    print("https://console.anthropic.com/settings/keys")
    print("\nYou can set it in one of these ways:")
    print("1. Export ANTHROPIC_API_KEY environment variable")
    print("2. Export ANTHROPIC_API_KEY_TILDE environment variable")
    print("3. Add to ~/.aide_ml/.env file")
    
    key = input("\nEnter your Anthropic API key (or press Enter to skip): ").strip()
    
    if key:
        # Save to .env file
        env_file = Path.home() / ".aide_ml" / ".env"
        env_file.parent.mkdir(exist_ok=True)
        
        with open(env_file, 'a') as f:
            f.write(f"\nANTHROPIC_API_KEY={key}\n")
        
        os.environ["ANTHROPIC_API_KEY"] = key
        print(f"✅ Saved ANTHROPIC_API_KEY to {env_file}")
        return True
    else:
        print("⚠️  Skipped Anthropic API key setup")
        return False


def create_example_env_file():
    """Create an example .env file with all possible configurations"""
    example_env = Path.home() / ".aide_ml" / ".env.example"
    example_env.parent.mkdir(exist_ok=True)
    
    content = """# AIDE ML Environment Variables Example
# Copy this file to .env and fill in your values

# Anthropic API Key (required for Anthropic backend)
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Alternative: Use ANTHROPIC_API_KEY_TILDE if you prefer
# ANTHROPIC_API_KEY_TILDE=your-anthropic-api-key-here

# OpenAI API Key (optional, for OpenAI backend)
# OPENAI_API_KEY=your-openai-api-key-here

# Google API Key (optional, for Gemini backend)
# GOOGLE_API_KEY=your-google-api-key-here

# OpenRouter API Key (optional, for OpenRouter backend)
# OPENROUTER_API_KEY=your-openrouter-api-key-here

# Default backend to use (claude_code, anthropic, openai, gemini, hybrid)
# DEFAULT_BACKEND=claude_code

# Default model for each backend
# DEFAULT_CLAUDE_MODEL=claude-opus-4
# DEFAULT_ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
# DEFAULT_OPENAI_MODEL=gpt-4o
# DEFAULT_GEMINI_MODEL=gemini-2.0-flash-thinking-exp-1219

# Performance monitoring settings
# ENABLE_PERFORMANCE_MONITORING=true
# PERFORMANCE_LOG_DIR=~/.aide_ml/performance_logs

# MCP (Model Context Protocol) settings
# ENABLE_MCP=true
# MCP_HTTP_MODE=false
# MCP_HTTP_PORT=8080
"""
    
    with open(example_env, 'w') as f:
        f.write(content)
    
    print(f"📝 Created example environment file at: {example_env}")


def check_environment():
    """Check if conda environment is activated"""
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env != "aideml":
        print("⚠️  Warning: 'aideml' conda environment is not activated")
        print("   Run: conda activate aideml")
        return False
    return True


def main():
    print("🚀 AIDE ML API Key Configuration")
    print("=" * 40)
    
    # Check conda environment
    if not check_environment():
        print("\n❌ Please activate the aideml conda environment first")
        sys.exit(1)
    
    # Create example .env file
    create_example_env_file()
    
    # Check Claude Code authentication
    print("\n1️⃣  Checking Claude Code Backend...")
    claude_ok, claude_msg = check_claude_code_auth()
    if claude_ok:
        print(f"✅ {claude_msg}")
        print("   Claude Code should handle authentication automatically")
    else:
        print(f"⚠️  {claude_msg}")
        print("   Install Claude Code CLI from: https://github.com/anthropics/claude-code")
    
    # Set up Anthropic API key
    print("\n2️⃣  Setting up Anthropic Backend...")
    anthropic_ok = setup_anthropic_key()
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Configuration Summary")
    print("=" * 40)
    
    backends_available = []
    if claude_ok:
        backends_available.append("claude_code")
    if anthropic_ok:
        backends_available.append("anthropic")
    
    if backends_available:
        print(f"✅ Available backends: {', '.join(backends_available)}")
        print("\nYou can now run examples with:")
        for backend in backends_available:
            print(f"  python run_bitcoin.py --backend {backend}")
    else:
        print("❌ No backends configured")
        print("   Please set up at least one backend to run examples")
    
    # Additional tips
    print("\n💡 Tips:")
    print("- For production use, add API keys to ~/.aide_ml/.env")
    print("- Use different models with --backend-opt model=<model-name>")
    print("- Enable MCP with --backend-opt use_mcp=true (Claude Code only)")
    print("- Compare backends with: python compare_backends.py")


if __name__ == "__main__":
    main()