#!/bin/bash

# Model Generalization Framework - Quick Execution Script
# Author: Manvanth Sai
# This script automates environment setup and framework execution

set -e  # Exit immediately if a command fails

# Color codes for enhanced output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Display header
echo -e "${BLUE}${BOLD}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     Model Generalization Framework - Quick Start              ║"
echo "║     Author: Manvanth Sai                                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check Python version
echo -e "${CYAN}🔍 Checking Python environment...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Error: Python 3 is not installed${NC}"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python version: $PYTHON_VERSION${NC}"

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}❌ Error: requirements.txt not found${NC}"
    exit 1
fi

# Create visualizations directory
if [ ! -d "visualizations" ]; then
    echo -e "${YELLOW}📁 Creating visualizations directory...${NC}"
    mkdir -p visualizations
fi

# Check for virtual environment
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}🔧 Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment found${NC}"
fi

# Activate virtual environment
echo -e "${CYAN}🔌 Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${CYAN}📦 Upgrading pip...${NC}"
pip install --upgrade pip -q

# Install/check dependencies
echo -e "${CYAN}📦 Installing dependencies...${NC}"
pip install -r requirements.txt -q

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All dependencies installed successfully${NC}"
else
    echo -e "${RED}❌ Failed to install dependencies${NC}"
    exit 1
fi

# Run the framework
echo ""
echo -e "${BLUE}${BOLD}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}${BOLD}  🚀 Starting Model Generalization Framework Analysis${NC}"
echo -e "${BLUE}${BOLD}════════════════════════════════════════════════════════════════${NC}"
echo ""

python3 main.py

# Check execution status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}${BOLD}  ✅ Analysis Completed Successfully!${NC}"
    echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${CYAN}📊 Generated Visualizations:${NC}"
    echo ""
    
    # List generated files with size
    for file in visualizations/*.png visualizations/*.html visualizations/*.log; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            filesize=$(du -h "$file" | cut -f1)
            echo -e "   ${GREEN}✓${NC} $filename ${YELLOW}($filesize)${NC}"
        fi
    done
    
    echo ""
    echo -e "${YELLOW}💡 Tips:${NC}"
    echo -e "   • View PNG files with any image viewer"
    echo -e "   • Open HTML files in your web browser for interactive exploration"
    echo -e "   • Check analysis.log for detailed execution information"
    echo ""
    echo -e "${BLUE}📂 Output location: $(pwd)/visualizations/${NC}"
    
else
    echo ""
    echo -e "${RED}${BOLD}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}${BOLD}  ❌ Analysis Failed${NC}"
    echo -e "${RED}${BOLD}════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${YELLOW}Please check visualizations/analysis.log for error details${NC}"
    exit 1
fi

# Deactivate virtual environment
deactivate

echo ""
echo -e "${CYAN}Thank you for using Model Generalization Framework!${NC}"
echo ""
