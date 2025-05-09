@echo off

REM Clean up any existing processes
if exist .api_pid (
    for /f %%i in (.api_pid) do (
        if not "%%i"=="" (
            taskkill /F /PID %%i >nul 2>&1
            if %errorlevel% equ 0 (
                echo Successfully killed API process with PID %%i
            )
        )
    )
    del .api_pid
)

if exist .streamlit_pid (
    for /f %%i in (.streamlit_pid) do (
        if not "%%i"=="" (
            taskkill /F /PID %%i >nul 2>&1
            if %errorlevel% equ 0 (
                echo Successfully killed Streamlit process with PID %%i
            )
        )
    )
    del .streamlit_pid
)

REM Create necessary directories
if not exist "data\research_output" mkdir "data\research_output"
if not exist "data\paper_cache" mkdir "data\paper_cache"
if not exist "data\zotero_downloads" mkdir "data\zotero_downloads"
if not exist "data\arxiv_downloads" mkdir "data\arxiv_downloads"
if not exist "data\json_output" mkdir "data\json_output"

REM Start the backend API
echo Starting backend API...
start /B python web_api.py
echo %ERRORLEVEL% > .api_pid
echo Backend API started

REM Wait for API to initialize
echo Waiting for API to initialize...
timeout /t 3 /nobreak >nul

REM Start the Streamlit frontend
echo Starting Streamlit frontend...
start /B streamlit run streamlit_app.py
echo %ERRORLEVEL% > .streamlit_pid
echo Streamlit frontend started

echo.
echo Research Agent is now running!
echo Backend API: http://localhost:8000
echo Frontend UI: http://localhost:8501
echo To stop the application, run: stop_app.bat
echo.
echo Press Ctrl+C to exit
echo.

REM On Windows, we don't use the tail -f command, instead we pause
pause