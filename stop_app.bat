@echo off

echo Stopping Research Agent...

rem Stop the API process
if exist ".api_pid" (
    set /p API_PID=<.api_pid
    echo Stopping backend API...
    taskkill /PID %API_PID% /F >nul 2>&1
    if errorlevel 1 (
        echo API process was not running
    ) else (
        echo API process stopped
    )
    del .api_pid >nul 2>&1
) else (
    echo API PID file not found
)

rem Stop the Streamlit process
if exist ".streamlit_pid" (
    set /p STREAMLIT_PID=<.streamlit_pid
    echo Stopping Streamlit frontend...
    taskkill /PID %STREAMLIT_PID% /F >nul 2>&1
    if errorlevel 1 (
        echo Streamlit process was not running
    ) else (
        echo Streamlit process stopped
    )
    del .streamlit_pid >nul 2>&1
) else (
    echo Streamlit PID file not found
)

echo Research Agent stopped successfully
pause