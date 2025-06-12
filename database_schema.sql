-- Database schema for financial processing system

-- Provision Planning Tables
CREATE TABLE IF NOT EXISTS provision_planning (
    id SERIAL PRIMARY KEY,
    expense_category VARCHAR(100),
    vendor_name VARCHAR(100),
    amount DECIMAL(15,2),
    frequency VARCHAR(20),
    last_provision_date DATE,
    expected_date DATE,
    confidence_score DECIMAL(4,2)
);

CREATE TABLE IF NOT EXISTS expense_ledger (
    expense_id VARCHAR(20) PRIMARY KEY,
    expense_category VARCHAR(100),
    vendor_name VARCHAR(100),
    amount DECIMAL(15,2),
    transaction_date DATE,
    payment_status VARCHAR(20),
    recurring_flag CHAR(1)
);

CREATE TABLE IF NOT EXISTS previous_provisions (
    provision_id VARCHAR(20) PRIMARY KEY,
    expense_category VARCHAR(100),
    vendor_name VARCHAR(100),
    amount DECIMAL(15,2),
    provision_date DATE,
    actual_amount DECIMAL(15,2),
    difference DECIMAL(15,2),
    status VARCHAR(20)
);

CREATE TABLE IF NOT EXISTS vendor_info (
    vendor_id VARCHAR(20) PRIMARY KEY,
    vendor_name VARCHAR(100),
    contract_start_date DATE,
    contract_end_date DATE,
    payment_terms VARCHAR(50),
    recurring_amount DECIMAL(15,2),
    frequency VARCHAR(20)
);

-- Financial Consolidation Tables
CREATE TABLE IF NOT EXISTS consolidation (
    id SERIAL PRIMARY KEY,
    entity_code VARCHAR(20),
    account_code VARCHAR(20),
    account_name VARCHAR(100),
    debit_amount DECIMAL(15,2),
    credit_amount DECIMAL(15,2),
    transaction_date DATE,
    interco_flag CHAR(1)
);

CREATE TABLE IF NOT EXISTS trial_balance (
    entity_code VARCHAR(20),
    account_code VARCHAR(20),
    account_name VARCHAR(100),
    opening_debit DECIMAL(15,2),
    opening_credit DECIMAL(15,2),
    current_debit DECIMAL(15,2),
    current_credit DECIMAL(15,2),
    closing_debit DECIMAL(15,2),
    closing_credit DECIMAL(15,2),
    PRIMARY KEY (entity_code, account_code)
);

CREATE TABLE IF NOT EXISTS ic_mapping (
    entity_from VARCHAR(20),
    entity_to VARCHAR(20),
    account_from VARCHAR(20),
    account_to VARCHAR(20),
    elimination_account VARCHAR(20),
    description TEXT,
    PRIMARY KEY (entity_from, entity_to, account_from)
);

-- Financial Reporting Table
CREATE TABLE IF NOT EXISTS financial_report (
    id SERIAL PRIMARY KEY,
    report_type VARCHAR(10),
    period VARCHAR(20),
    entity VARCHAR(20),
    metric VARCHAR(100),
    value DECIMAL(15,2),
    previous_value DECIMAL(15,2),
    variance DECIMAL(15,2),
    variance_percentage DECIMAL(10,2)
);

-- Audit Support Table
CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    journal_id VARCHAR(20),
    transaction_date DATE,
    account_code VARCHAR(20),
    amount DECIMAL(15,2),
    user_id VARCHAR(20),
    action_type VARCHAR(20),
    change_details TEXT,
    audit_status VARCHAR(20)
);

-- Journal Validation Table
CREATE TABLE IF NOT EXISTS journal_entries (
    id SERIAL PRIMARY KEY,
    journal_id VARCHAR(20),
    transaction_date DATE,
    account_code VARCHAR(20),
    account_name VARCHAR(100),
    debit_amount DECIMAL(15,2),
    credit_amount DECIMAL(15,2),
    entity VARCHAR(20),
    description TEXT
);

-- Intercompany Reconciliation Table
CREATE TABLE IF NOT EXISTS interco_reconciliation (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(20),
    entity_from VARCHAR(20),
    entity_to VARCHAR(20),
    account_code VARCHAR(20),
    amount DECIMAL(15,2),
    transaction_date DATE,
    status VARCHAR(20),
    reconciliation_id VARCHAR(20)
);

-- GST Validation Table
CREATE TABLE IF NOT EXISTS gst_validation (
    id SERIAL PRIMARY KEY,
    invoice_number VARCHAR(20),
    gstin VARCHAR(20),
    vendor_name VARCHAR(100),
    invoice_date DATE,
    taxable_amount DECIMAL(15,2),
    cgst_amount DECIMAL(15,2),
    sgst_amount DECIMAL(15,2),
    igst_amount DECIMAL(15,2),
    total_amount DECIMAL(15,2),
    itc_eligible CHAR(1)
);

-- TDS Validation Tables
CREATE TABLE IF NOT EXISTS tds_validation (
    id SERIAL PRIMARY KEY,
    challan_number VARCHAR(20),
    pan VARCHAR(20),
    vendor_name VARCHAR(100),
    section_code VARCHAR(10),
    tds_amount DECIMAL(15,2),
    transaction_date DATE,
    payment_amount DECIMAL(15,2),
    status VARCHAR(20)
);

CREATE TABLE IF NOT EXISTS traces_data (
    id SERIAL PRIMARY KEY,
    challan_number VARCHAR(20),
    pan VARCHAR(20),
    vendor_name VARCHAR(100),
    section_code VARCHAR(10),
    tds_amount DECIMAL(15,2),
    transaction_date DATE,
    status VARCHAR(20),
    acknowledgement_number VARCHAR(20)
); 