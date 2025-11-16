#!/bin/bash

# Setup Python Virtual Environment for MediScan AI Backend

echo "🐍 Setting up Python Virtual Environment..."
echo "==========================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

echo ""
echo "📥 Activating virtual environment and installing dependencies..."

# Activate virtual environment and install dependencies
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "==========================================="
echo "✅ Setup Complete!"
echo "==========================================="
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source backend/venv/bin/activate"
echo ""
echo "To start the backend server:"
echo "  source backend/venv/bin/activate"
echo "  python backend/app.py"
echo ""
echo "Or use the main start script:"
echo "  ./start.sh"
echo ""
