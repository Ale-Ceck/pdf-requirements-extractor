#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Determine Python command to use
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "Python is not installed. Please install Python 3 and try again."
    exit 1
fi

# Check if Python virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    $PYTHON_CMD -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate || {
    echo "Failed to activate virtual environment. Make sure Python venv module is installed."
    exit 1
}

# Verify Python is now from the virtual environment
PYTHON_PATH=$(which python)
echo "Using Python: $PYTHON_PATH"

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Node.js is required for the frontend but not installed."
    echo "Please install Node.js and npm, then run this script again."
    exit 1
fi

echo "Using Node.js version: $(node -v)"
echo "Using npm version: $(npm -v)"

# Check if frontend exists
if [ ! -d "frontend" ]; then
    echo "Error: frontend directory not found!"
    exit 1
fi

# Check if frontend node_modules exists, if not, install dependencies
if [ ! -d "frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    (cd frontend && npm install) || {
        echo "Failed to install npm dependencies. Please check npm installation."
        exit 1
    }
fi

# Check if frontend build exists, if not, build it
if [ ! -d "frontend/build" ]; then
    echo "Building frontend..."
    (cd frontend && npm run build) || {
        echo "Failed to build frontend. Please check for errors."
        exit 1
    }
fi

# Check if API server exists
if [ ! -f "api_server.py" ]; then
    echo "Error: api_server.py not found!"
    exit 1
fi

# Create .env.local for frontend if it doesn't exist
if [ ! -f "frontend/.env.local" ]; then
    echo "Creating frontend environment configuration..."
    echo "REACT_APP_API_URL=http://localhost:5001/api" > frontend/.env.local
fi

# Start the API server and frontend development server
echo "Starting PDF Requirements Extractor API server and frontend..."
echo "The frontend will be available at http://localhost:3000"
echo "The API server will be available at http://localhost:5001"
echo "Press Ctrl+C to stop both servers"

# Start API server in the background
python api_server.py &
API_PID=$!

# Start frontend development server
(cd frontend && npm start)

# Cleanup API server when script is terminated
trap "kill $API_PID" EXIT