#!/bin/bash

# Start the server in the background
echo "Starting the financial server..."
uvicorn financial_server:app --host 0.0.0.0 --port 8000 --reload &
SERVER_PID=$!

# Wait for the server to start
echo "Waiting for server to start..."
sleep 5

# Run the tests
echo "Running tests..."
python test_server.py

# Kill the server process
echo "Stopping the server..."
kill $SERVER_PID 