import os
from sample_data import (
    SAMPLE_PROVISION_CSV,
    SAMPLE_CONSOLIDATION_CSV,
    SAMPLE_FINANCIAL_REPORT_CSV,
    SAMPLE_AUDIT_LOG_CSV,
    SAMPLE_JOURNAL_CSV,
    SAMPLE_INTERCO_CSV,
    SAMPLE_GST_CSV,
    SAMPLE_TDS_CSV,
    SAMPLE_TRACES_CSV,
    SAMPLE_TRIAL_BALANCE_CSV,
    SAMPLE_IC_MAPPING_CSV,
    SAMPLE_EXPENSE_LEDGER_CSV,
    SAMPLE_PREVIOUS_PROVISIONS_CSV,
    SAMPLE_VENDOR_INFO_CSV
)

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def write_csv_file(filename, content):
    with open(filename, 'w') as f:
        f.write(content.strip())

def main():
    # Create a directory for CSV files
    csv_dir = "csv_files"
    ensure_directory(csv_dir)

    # Define file mappings
    file_mappings = {
        # Provision Planning Agent files
        "provision_planning.csv": SAMPLE_PROVISION_CSV,
        "expense_ledger.csv": SAMPLE_EXPENSE_LEDGER_CSV,
        "previous_provisions.csv": SAMPLE_PREVIOUS_PROVISIONS_CSV,
        "vendor_info.csv": SAMPLE_VENDOR_INFO_CSV,

        # Financial Consolidation Agent files
        "consolidation.csv": SAMPLE_CONSOLIDATION_CSV,
        "trial_balance.csv": SAMPLE_TRIAL_BALANCE_CSV,
        "ic_mapping.csv": SAMPLE_IC_MAPPING_CSV,

        # Financial Reporting Agent file
        "financial_report.csv": SAMPLE_FINANCIAL_REPORT_CSV,

        # Audit Support Agent file
        "audit_log.csv": SAMPLE_AUDIT_LOG_CSV,

        # Journal Validation Agent file
        "journal_entries.csv": SAMPLE_JOURNAL_CSV,

        # Intercompany Reconciliation Agent file
        "interco_reconciliation.csv": SAMPLE_INTERCO_CSV,

        # GST Validation Agent file
        "gst_validation.csv": SAMPLE_GST_CSV,

        # TDS Validation Agent files
        "tds_validation.csv": SAMPLE_TDS_CSV,
        "traces_data.csv": SAMPLE_TRACES_CSV
    }

    # Write each CSV file
    for filename, content in file_mappings.items():
        filepath = os.path.join(csv_dir, filename)
        write_csv_file(filepath, content)
        print(f"Created: {filepath}")

if __name__ == "__main__":
    main()