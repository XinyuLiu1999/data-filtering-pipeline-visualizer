#!/bin/bash

# Data Filtering Pipeline Visualization Tool
# Usage: ./run.sh [--port PORT] [--host HOST] [--debug]

PORT=${PORT:-5000}
HOST=${HOST:-0.0.0.0}
DEBUG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Change to script directory
cd "$(dirname "$0")"

# Check if dependencies are installed
if ! python3 -c "import flask, pandas, numpy" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

echo "Starting Data Filtering Pipeline Visualizer..."
echo "Access the tool at: http://${HOST}:${PORT}"
echo ""

python3 app.py --host "$HOST" --port "$PORT" $DEBUG
