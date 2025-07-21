#!/bin/bash
# AIDE ML Examples Quick Start Script
# This script demonstrates the complete workflow from setup to running examples

set -e  # Exit on error

echo "🚀 AIDE ML Examples Quick Start"
echo "=============================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Environment Setup
echo -e "${YELLOW}Step 1: Setting up environment...${NC}"
cd scripts
bash setup_env.sh
cd ..

echo ""
echo -e "${GREEN}✅ Environment setup complete${NC}"
echo ""

# Step 2: API Key Configuration
echo -e "${YELLOW}Step 2: Configuring API keys...${NC}"
echo "This will set up API keys for both Claude Code and Anthropic backends"
python scripts/setup_api_keys.py

echo ""
echo -e "${GREEN}✅ API key configuration complete${NC}"
echo ""

# Step 3: Activate conda environment reminder
echo -e "${YELLOW}Step 3: Activate conda environment${NC}"
echo "Please run the following command if not already activated:"
echo -e "${GREEN}conda activate aideml${NC}"
echo ""
read -p "Press Enter to continue once the environment is activated..."

# Step 4: Run Bitcoin example
echo ""
echo -e "${YELLOW}Step 4: Running Bitcoin price prediction example...${NC}"
echo "This will run a quick test with reduced iterations"
cd bitcoin_price
python run_bitcoin.py --backend claude_code --iterations 3 --workers 2
cd ..

echo ""
echo -e "${GREEN}✅ Bitcoin example complete${NC}"
echo ""

# Step 5: Optional - Run comparison
echo -e "${YELLOW}Step 5: Backend comparison (optional)${NC}"
echo "Would you like to run a quick comparison between backends?"
read -p "This will take about 10-15 minutes. Continue? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd scripts
    python compare_backends.py --quick --task bitcoin_price
    cd ..
    echo ""
    echo -e "${GREEN}✅ Comparison complete${NC}"
fi

# Summary
echo ""
echo "=============================="
echo -e "${GREEN}🎉 Quick start complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Check the outputs/ directory for generated solutions"
echo "2. Read the tutorials in bitcoin_price/README.md and house_prices/README.md"
echo "3. Try running with different parameters:"
echo "   - Use --backend anthropic for Anthropic backend"
echo "   - Use --use-mcp for enhanced Claude Code features"
echo "   - Use --iterations to control search depth"
echo "4. Run the house prices example for a more complex task:"
echo "   cd house_prices && python run_house_prices.py"
echo ""
echo "Happy experimenting with AIDE ML! 🚀"