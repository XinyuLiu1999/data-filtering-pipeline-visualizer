#!/bin/bash
# Script to start Data Filtering Pipeline Visualizer
# This works on systems without systemd

cd /cephfs/liuxinyu/data-filtering-pipeline-visualizer

PORT=7014
HOST="0.0.0.0"

echo "Starting Data Filtering Pipeline Visualizer..."
echo "Access at: http://$(curl -s ifconfig.me):$PORT"
echo ""

# Use conda environment if available
if command -v conda &> /dev/null && conda env list | grep -q "data-vis"; then
    echo "Using conda environment: data-vis"

    # Check if gunicorn is available in the conda env
    if conda run -n data-vis which gunicorn &> /dev/null; then
        echo "Using Gunicorn (production server)..."
        nohup conda run -n data-vis gunicorn --workers 4 --bind $HOST:$PORT --timeout 300 app:app > logs/gunicorn.log 2>&1 &
        PID=$!
        echo $PID > gunicorn.pid
        echo "Started with PID: $PID"
        echo "Logs: logs/gunicorn.log"
    else
        echo "Using Python directly..."
        nohup conda run -n data-vis python3 app.py --host $HOST --port $PORT > logs/app.log 2>&1 &
        PID=$!
        echo $PID > app.pid
        echo "Started with PID: $PID"
        echo "Logs: logs/app.log"
    fi
else
    # Fallback to system python
    echo "Using system Python..."
    nohup python3 app.py --host $HOST --port $PORT > logs/app.log 2>&1 &
    PID=$!
    echo $PID > app.pid
    echo "Started with PID: $PID"
    echo "Logs: logs/app.log"
fi

echo ""
echo "To stop the service, run: ./stop_service.sh"
