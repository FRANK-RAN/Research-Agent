@echo off

rem Create necessary directories
if not exist "research_output" mkdir research_output
if not exist "paper_cache" mkdir paper_cache
if not exist "zotero_downloads" mkdir zotero_downloads
if not exist "arxiv_downloads" mkdir arxiv_downloads
if not exist "json_output" mkdir json_output

rem Start the backend API
echo Starting backend API...
start /B python web_api.py
echo %ERRORLEVEL% > .api_pid
echo Backend API started

rem Wait for API to initialize
echo Waiting for API to initialize...
timeout /t 3 /nobreak >nul

rem Start the Streamlit frontend
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

rem On Windows, we don't use the tail -f command, instead we pause
pause