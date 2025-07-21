#!/bin/bash
# AIDE ML Environment Setup Script
# This script sets up the conda environment and installs all required dependencies

set -e  # Exit on error

echo "🚀 AIDE ML Environment Setup"
echo "============================"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Error: Conda is not installed. Please install Anaconda or Miniconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Get the script directory (examples/scripts)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo "📁 Project root: $PROJECT_ROOT"

# Check if environment.yml exists
if [ ! -f "$PROJECT_ROOT/environment.yml" ]; then
    echo "❌ Error: environment.yml not found in project root"
    exit 1
fi

# Check if aideml environment already exists
if conda env list | grep -q "^aideml "; then
    echo "✅ Conda environment 'aideml' already exists"
    echo "   To update it, run: conda env update -f environment.yml"
else
    echo "📦 Creating conda environment from environment.yml..."
    conda env create -f "$PROJECT_ROOT/environment.yml"
    echo "✅ Conda environment created successfully"
fi

echo ""
echo "🔧 Activating the environment..."
echo "   Run: conda activate aideml"

# Create necessary directories
echo ""
echo "📁 Creating required directories..."
mkdir -p ~/.aide_ml/{logs,benchmarks,e2e_test_results,performance_logs}
echo "✅ Directories created"

# Check Python version in the environment
echo ""
echo "🐍 Checking Python version..."
conda run -n aideml python --version

# Install additional dependencies if requirements.txt exists
if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
    echo ""
    echo "📦 Installing Python dependencies..."
    conda run -n aideml pip install -r "$PROJECT_ROOT/requirements.txt"
    echo "✅ Python dependencies installed"
fi

echo ""
echo "✨ Environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the environment: conda activate aideml"
echo "2. Configure API keys: python setup_api_keys.py"
echo "3. Run an example: cd ../bitcoin_price && python run_bitcoin.py"