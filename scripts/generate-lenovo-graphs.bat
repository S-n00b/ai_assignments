@echo off
REM Lenovo Graph Generation Batch Script
REM Simple batch file to run the Python graph generation script

echo 🎯 Lenovo Graph Generation Script
echo ========================================

REM Check if virtual environment is activated
if not defined VIRTUAL_ENV (
    echo ⚠️  Virtual environment not detected.
    echo Please activate it first:
    echo    C:\Users\samne\PycharmProjects\ai_assignments\venv\Scripts\Activate.bat
    echo.
    echo Then run this script again.
    pause
    exit /b 1
)

echo ✅ Virtual environment activated: %VIRTUAL_ENV%

REM Change to project directory
cd /d "C:\Users\samne\PycharmProjects\ai_assignments"

REM Run the Python script
echo.
echo 🚀 Running graph generation...
python scripts\generate_lenovo_graphs_simple.py

REM Check if successful
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Graph generation completed successfully!
    echo 📁 Check the 'neo4j_data' directory for generated files
    echo 🔗 Access Neo4j Browser at: http://localhost:7474
) else (
    echo.
    echo ❌ Graph generation failed
    echo Check the error messages above for details
)

echo.
pause
