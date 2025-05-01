#!/bin/bash

# Create necessary directories
mkdir -p research_output paper_cache zotero_downloads arxiv_downloads json_output

# Start the backend API
echo "Starting backend API..."
python web_api.py &
API_PID=$!
echo $API_PID > .api_pid
echo "Backend API started with PID: $API_PID"

# Wait for API to initialize
echo "Waiting for API to initialize..."
sleep 3

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