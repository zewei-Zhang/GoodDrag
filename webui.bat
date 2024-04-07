@echo off

REM Temporary file for modified requirements
set TEMP_REQ_FILE=temp_requirements.txt

REM Detect CUDA version using nvcc
for /f "delims=" %%i in ('nvcc --version ^| findstr /i "release"') do set "CUDA_VER_FULL=%%i"

REM Extract the version number, assuming it's in the format "Cuda compilation tools, release 11.8, V11.8.89"
for /f "tokens=5 delims=, " %%a in ("%CUDA_VER_FULL%") do set "CUDA_VER=%%a"
for /f "tokens=1 delims=vV" %%b in ("%CUDA_VER%") do set "CUDA_VER=%%b"
for /f "tokens=1-2 delims=." %%c in ("%CUDA_VER%") do set "CUDA_MAJOR=%%c" & set "CUDA_MINOR=%%d"

REM Concatenate major and minor version numbers to form the CUDA tag
set "CUDA_TAG=cu%CUDA_MAJOR%%CUDA_MINOR%"

echo Detected CUDA Tag: %CUDA_TAG%

REM Modify the torch and torchvision lines in requirements.txt to include the CUDA version
for /F "tokens=*" %%A in (requirements.txt) do (
    echo %%A | findstr /I "torch==" >nul
    if errorlevel 1 (
        echo %%A | findstr /I "torchvision==" >nul
        if errorlevel 1 (
            echo %%A >> "%TEMP_REQ_FILE%"
        ) else (
            echo torchvision==0.15.2+%CUDA_TAG% >> "%TEMP_REQ_FILE%"
        )
    ) else (
        echo torch==2.0.1+%CUDA_TAG%>> "%TEMP_REQ_FILE%"
    )
)

REM Replace the original requirements file with the modified one
move /Y "%TEMP_REQ_FILE%" requirements.txt


REM Define the virtual environment directory
set VENV_DIR=GoodDrag

REM Check if Python is installed
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install it first.
    pause
    exit /b
)



REM Create a virtual environment if it doesn't exist
if not exist "%VENV_DIR%" (
    echo Creating virtual environment...
    python -m venv %VENV_DIR%
)

REM Activate the virtual environment
call %VENV_DIR%\Scripts\activate.bat

REM Install dependencies (uncomment and modify the next line if you have any dependencies)
pip install -r requirements.txt

REM Run the Python script
echo Starting gooddrag_ui.py...
python gooddrag_ui.py

REM Deactivate the virtual environment on script exit
call %VENV_DIR%\Scripts\deactivate.bat

echo Script finished. Press any key to exit.
pause > nul
