@echo off
echo PDF Requirements Extractor Launcher

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python is not installed. Please install Python 3 and try again.
    pause
    exit /b 1
)

echo Using Python version:
python --version

REM Check if Python virtual environment exists
if not exist venv (
    echo Creating Python virtual environment...
    python -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo Failed to create virtual environment. Make sure Python venv module is installed.
        pause
        exit /b 1
    )
)

REM Activate the virtual environment
echo Activating virtual environment...
call venv\Scripts\activate
if %ERRORLEVEL% neq 0 (
    echo Failed to activate the virtual environment.
    pause
    exit /b 1
)

REM Install Python dependencies
echo Installing Python dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo Failed to install Python dependencies.
    pause
    exit /b 1
)

REM Check if Node.js is installed
where node >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Node.js is required for the frontend but not installed.
    echo Please install Node.js and npm, then run this script again.
    pause
    exit /b 1
)

echo Using Node.js version:
node --version
echo Using npm version:
npm --version

REM Check if frontend directory exists
if not exist frontend (
    echo Error: frontend directory not found!
    pause
    exit /b 1
)

REM Check if frontend node_modules exists, if not, install dependencies
if not exist frontend\node_modules (
    echo Installing frontend dependencies...
    cd frontend
    npm install
    if %ERRORLEVEL% neq 0 (
        echo Failed to install npm dependencies. Please check npm installation.
        cd ..
        pause
        exit /b 1
    )
    cd ..
)

REM Check if frontend build exists, if not, build it
if not exist frontend\build (
    echo Building frontend...
    cd frontend
    npm run build
    if %ERRORLEVEL% neq 0 (
        echo Failed to build frontend. Please check for errors.
        cd ..
        pause
        exit /b 1
    )
    cd ..
)

REM Check if API server exists
if not exist api_server.py (
    echo Error: api_server.py not found!
    pause
    exit /b 1
)

REM Start the API server
echo Starting PDF Requirements Extractor API server...
echo Open your browser and navigate to http://localhost:5001
echo Press Ctrl+C to stop the server
python api_server.py