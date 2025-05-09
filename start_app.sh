#!/bin/bash

# Function to check if a port is in use
check_port() {
    lsof -i :$1 >/dev/null 2>&1
    return $?
}

# Function to kill process on a port
kill_port() {
    if check_port $1; then
        echo "Port $1 is in use. Attempting to kill the process..."
        lsof -ti :$1 | xargs kill -9 2>/dev/null
        sleep 1
    fi
}

# Clean up any existing processes
if [ -f .api_pid ]; then
    kill $(cat .api_pid) 2>/dev/null
    rm .api_pid
fi

if [ -f .streamlit_pid ]; then
    kill $(cat .streamlit_pid) 2>/dev/null
    rm .streamlit_pid
fi

# Kill any processes using our ports
kill_port 8000  # FastAPI port
kill_port 8501  # Streamlit port

# Create necessary directories
mkdir -p data/research_output data/paper_cache data/zotero_downloads data/arxiv_downloads data/json_output

# Start the backend API
echo "Starting backend API..."
python web_api.py &
API_PID=$!
echo $API_PID > .api_pid
echo "Backend API started with PID: $API_PID"

# Wait for API to initialize and check if it's running
echo "Waiting for API to initialize..."
sleep 3
if ! check_port 8000; then
    echo "Error: Failed to start backend API. Port 8000 is not available."
    exit 1
fi

# Start the Streamlit frontend
echo "Starting Streamlit frontend..."
streamlit run streamlit_app.py &
STREAMLIT_PID=$!
echo $STREAMLIT_PID > .streamlit_pid
echo "Streamlit frontend started with PID: $STREAMLIT_PID"

echo ""
echo "Research Agent is now running!"
echo "Backend API: http://localhost:8000"
echo "Frontend UI: http://localhost:8501"
echo "To stop the application, run: ./stop_app.sh"
echo ""
echo "Press Ctrl+C to stop watching logs (the application will continue running)"
echo "Showing logs..."
echo ""

# Show logs (optional - press Ctrl+C to stop watching logs)
tail -f nohup.out 2>/dev/null || echo "Log file not available"