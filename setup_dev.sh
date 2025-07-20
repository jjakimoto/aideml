#!/bin/bash
# Quick setup script for development mode

echo "Setting up AIDE ML for development mode..."

# Check if we're in the right directory
if [ ! -f "setup.py" ] || [ ! -d "aide" ]; then
    echo "Error: Please run this script from the root of the aideml repository"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Setup complete! You can now run AIDE ML using:"
echo "  python run_aide.py data_dir=<path> goal=<goal> eval=<metric>"
echo ""
echo "Or run the WebUI with:"
echo "  python run_webui.py"
echo ""
echo "Don't forget to set your API key:"
echo "  export OPENAI_API_KEY=<your-key>"