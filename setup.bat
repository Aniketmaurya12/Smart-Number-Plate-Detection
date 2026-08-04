@echo off
echo ============================================
echo   FastALPR - Setup
echo ============================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install it from https://www.python.org/downloads/
    echo IMPORTANT: check "Add python.exe to PATH" during install.
    pause
    exit /b 1
)

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies (this may take a few minutes)...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ============================================
echo   Setup complete!
echo ============================================
echo.
echo To run on an image:
echo     venv\Scripts\activate.bat
echo     python run_image.py --input assets\test_image.png
echo.
echo To run on a video:
echo     venv\Scripts\activate.bat
echo     python run_video.py --input traffic.mp4
echo.
pause
