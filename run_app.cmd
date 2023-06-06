@echo off

set VENV_DIR=venv

REM Check if virtual environment already exists
if not exist %VENV_DIR% (
    REM Create a new virtual environment
    python -m venv %VENV_DIR%
)

REM Activate virtual environment
call %VENV_DIR%\Scripts\activate.bat

REM Install requirements
pip install -r requirements.txt

REM Start Flask server in a new command prompt window
start cmd /k "call %VENV_DIR%\Scripts\activate.bat && python src/server.py"

REM Wait for the server to start
timeout /t 5

REM Start Streamlit app in the current command prompt window
call %VENV_DIR%\Scripts\activate.bat && streamlit run src/app.py

REM Deactivate virtual environment
deactivate
