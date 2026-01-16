#!/bin/bash
# Script to check status of Data Filtering Pipeline Visualizer

cd /cephfs/liuxinyu/data-filtering-pipeline-visualizer

echo "=== Data Filtering Pipeline Visualizer Status ==="
echo ""

RUNNING=false

# Check gunicorn
if [ -f gunicorn.pid ]; then
    PID=$(cat gunicorn.pid)
    if kill -0 $PID 2>/dev/null; then
        echo "✓ Gunicorn is running (PID: $PID)"
        RUNNING=true
    else
        echo "✗ Gunicorn PID file exists but process is not running"
        rm gunicorn.pid
    fi
fi

# Check python app
if [ -f app.pid ]; then
    PID=$(cat app.pid)
    if kill -0 $PID 2>/dev/null; then
        echo "✓ Python app is running (PID: $PID)"
        RUNNING=true
    else
        echo "✗ App PID file exists but process is not running"
        rm app.pid
    fi
fi

# Check for any running processes
echo ""
echo "Running processes:"
ps aux | grep -E "(gunicorn|app.py)" | grep -v grep

echo ""
if [ "$RUNNING" = true ]; then
    echo "Status: RUNNING ✓"
    PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "Unable to detect")
    echo "Access at: http://$PUBLIC_IP:7014"
else
    echo "Status: STOPPED ✗"
    echo "Start with: ./start_service.sh"
fi

# Check if port is in use
echo ""
if command -v netstat &> /dev/null; then
    echo "Port 7014 status:"
    netstat -tuln | grep :7014 || echo "Port 7014 is not in use"
elif command -v ss &> /dev/null; then
    echo "Port 7014 status:"
    ss -tuln | grep :7014 || echo "Port 7014 is not in use"
fi
