#!/bin/bash
# Script to stop Data Filtering Pipeline Visualizer

cd /cephfs/liuxinyu/data-filtering-pipeline-visualizer

echo "Stopping Data Filtering Pipeline Visualizer..."

# Try to stop gunicorn
if [ -f gunicorn.pid ]; then
    PID=$(cat gunicorn.pid)
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        echo "Stopped gunicorn process (PID: $PID)"
        rm gunicorn.pid
    else
        echo "Gunicorn process not running"
        rm gunicorn.pid
    fi
fi

# Try to stop python app
if [ -f app.pid ]; then
    PID=$(cat app.pid)
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        echo "Stopped app process (PID: $PID)"
        rm app.pid
    else
        echo "App process not running"
        rm app.pid
    fi
fi

# Fallback: kill any remaining processes
pkill -f "gunicorn.*app:app" 2>/dev/null && echo "Killed remaining gunicorn processes"
pkill -f "python3 app.py" 2>/dev/null && echo "Killed remaining python app processes"

echo "Service stopped."
