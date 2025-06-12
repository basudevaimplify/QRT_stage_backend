import requests
import json
import os
from datetime import datetime

# Server URL
BASE_URL = "http://localhost:8000"
CSV_DIR = "csv_files"

def test_endpoints():
    # Test POST endpoints with CSV files
    post_endpoints = {
        "/process_journal_entries": "journal_entries.csv",
        "/process_interco_entries": "interco_reconciliation.csv",
        "/process_consolidated_entries": "consolidation.csv",
        "/process_gst_entries": "gst_validation.csv",
        "/process_tds_entries": "tds_validation.csv",
        "/process_provision_entries": "provision_planning.csv"
    }

    for endpoint, csv_file in post_endpoints.items():
        print(f"\nTesting {endpoint} endpoint...")
        file_path = os.path.join(CSV_DIR, csv_file)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                files = {'file': (csv_file, f, 'text/csv')}
                response = requests.post(f"{BASE_URL}{endpoint}", files=files)
                print(f"Status Code: {response.status_code}")
                print(f"Response: {response.json()}")
        else:
            print(f"File not found: {file_path}")

    # Test GET endpoints
    get_endpoints = [
        "/api/journal-entries",
        "/api/provision-planning",
        "/api/consolidation",
        "/api/financial-report",
        "/api/audit-log",
        "/api/interco-reconciliation",
        "/api/gst-validation",
        "/api/tds-validation",
        "/api/traces-data"
    ]

    for endpoint in get_endpoints:
        print(f"\nTesting {endpoint} endpoint...")
        response = requests.get(f"{BASE_URL}{endpoint}")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")

if __name__ == "__main__":
    print("Starting server tests with actual CSV files...")
    test_endpoints() 