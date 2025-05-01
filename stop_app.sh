#!/bin/bash

echo "Stopping Research Agent..."

# Stop the API process
if [ -f ".api_pid" ]; then
    API_PID=$(cat .api_pid)
    echo "Stopping backend API (PID: $API_PID)..."
    kill $API_PID 2>/dev/null || echo "API process was not running"
    rm .api_pid
else
    echo "API PID file not found"
fi

# Stop the Streamlit process
if [ -f ".streamlit_pid" ]; then
    STREAMLIT_PID=$(cat .streamlit_pid)
    echo "Stopping Streamlit frontend (PID: $STREAMLIT_PID)..."
    kill $STREAMLIT_PID 2>/dev/null || echo "Streamlit process was not running"
    rm .streamlit_pid
else
    echo "Streamlit PID file not found"
fi

echo "Research Agent stopped successfully"