#!/bin/bash

# Sign Language Interpreter - Flask App
# Simple startup script

echo "🤟 Starting Sign Language Interpreter..."
echo ""

# Install dependencies if needed
if ! python -c "import flask" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install flask flask-socketio python-socketio
fi

# Run the Flask app
cd src
python app.py

