#!/bin/bash

# DVC Testing Script
# This script guides you through the process of tracking data with DVC and RustFS

echo "Step 1: Checking environment..."
if [ -d "dvc/.venv" ]; then
    echo "Activating local modular environment..."
    source dvc/.venv/bin/activate
fi

if ! curl -s http://localhost:9000/health > /dev/null; then
    echo "ERROR: RustFS is not running. Please run: docker compose -f docker/docker-compose.yml up rustfs -d"
    exit 1
fi
echo "RustFS is up!"

echo "Step 2: Configuring a DEDICATED test remote..."
# We do NOT use -d here so it doesn't become the default project remote
dvc remote add rustfs-test s3://pcb-test-bucket --force
dvc remote modify rustfs-test endpointurl http://localhost:9000
dvc remote modify rustfs-test access_key_id rustfsadmin
dvc remote modify rustfs-test secret_access_key rustfsadmin
echo "Test remote configured!"

echo "Step 3: Tracking test data..."
dvc add dvc/tests/test_data.txt
echo "Data tracked locally in .dvc/cache"

echo "Step 4: Pushing data to the TEST bucket..."
echo "Note: If this fails, ensure you have created the 'pcb-test-bucket' in the RustFS console at http://localhost:9001"
dvc push -r rustfs-test

if [ $? -eq 0 ]; then
    echo "Success! Your test data is now in the separate test bucket."
    
    echo "Step 5: Testing 'dvc pull' from the test bucket..."
    rm dvc/tests/test_data.txt
    echo "File removed. Restoring with dvc pull..."
    dvc pull dvc/tests/test_data.txt.dvc -r rustfs-test
    
    if [ -f dvc/tests/test_data.txt ]; then
        echo "COMPLETE SUCCESS: Data was recovered from the test bucket!"
        cat dvc/tests/test_data.txt
    else
        echo "Error: Pull failed."
    fi
else
    echo "Error: Push failed. Did you create the 'pcb-test-bucket'?"
fi
