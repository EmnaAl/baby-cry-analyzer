@echo off
echo Baby Cry Analyzer - Windows Startup Script
echo ==========================================

echo.
echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ from https://python.org
    pause
    exit /b 1
)

echo.
echo Running setup...
python setup.py

echo.
echo Setup complete! 
echo.
echo What would you like to do?
echo 1. Generate sample data for testing
echo 2. Train the model
echo 3. Start the API server
echo 4. Run tests
echo 5. Exit
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo.
    echo Generating sample data...
    python generate_samples.py --samples 10
    echo.
    echo Sample data generated! You can now train the model.
    pause
) else if "%choice%"=="2" (
    echo.
    echo Training model...
    python train_model.py
    echo.
    echo Training complete! You can now start the API server.
    pause
) else if "%choice%"=="3" (
    echo.
    echo Starting API server...
    echo Open http://localhost:5000 in your browser
    echo Press Ctrl+C to stop the server
    python app.py
) else if "%choice%"=="4" (
    echo.
    echo Running tests...
    python test_api.py
    pause
) else if "%choice%"=="5" (
    echo Goodbye!
    exit /b 0
) else (
    echo Invalid choice. Please run the script again.
    pause
)

echo.
echo Press any key to exit...
pause > nul
