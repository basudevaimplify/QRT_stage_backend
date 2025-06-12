import os
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_dates(start_date, end_date, n):
    date_range = (end_date - start_date).days
    return [start_date + timedelta(days=random.randint(0, date_range)) for _ in range(n)]

def generate_provision_planning_data(n=100):
    expense_categories = ['Rent', 'Utilities', 'Maintenance', 'Insurance', 'Salaries', 'Marketing', 'IT Services', 'Legal', 'Consulting', 'Travel']
    vendors = [f'Vendor_{i}' for i in range(1, 21)]
    frequencies = ['Monthly', 'Quarterly', 'Annual']
    
    data = []
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    for _ in range(n):
        category = random.choice(expense_categories)
        vendor = random.choice(vendors)
        amount = round(random.uniform(1000, 100000), 2)
        frequency = random.choice(frequencies)
        last_date = random.choice(generate_dates(start_date, end_date, 1))
        expected_date = last_date + timedelta(days=30 if frequency == 'Monthly' else 90 if frequency == 'Quarterly' else 365)
        confidence = round(random.uniform(0.7, 1.0), 2)
        
        data.append({
            'expense_category': category,
            'vendor_name': vendor,
            'amount': amount,
            'frequency': frequency,
            'last_provision_date': last_date.strftime('%Y-%m-%d'),
            'expected_date': expected_date.strftime('%Y-%m-%d'),
            'confidence_score': confidence
        })
    
    return pd.DataFrame(data)

def generate_consolidation_data(n=100):
    entities = [f'ENT{i:03d}' for i in range(1, 11)]
    accounts = [f'ACC{i:04d}' for i in range(1, 21)]
    account_names = ['Cash', 'Accounts Receivable', 'Inventory', 'Fixed Assets', 'Accounts Payable', 'Loans', 'Equity', 'Revenue', 'Expenses']
    
    data = []
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    for _ in range(n):
        entity = random.choice(entities)
        account = random.choice(accounts)
        account_name = random.choice(account_names)
        debit = round(random.uniform(0, 1000000), 2) if random.random() > 0.5 else 0
        credit = round(random.uniform(0, 1000000), 2) if debit == 0 else 0
        date = random.choice(generate_dates(start_date, end_date, 1))
        interco = 'Y' if random.random() > 0.7 else 'N'
        
        data.append({
            'entity_code': entity,
            'account_code': account,
            'account_name': account_name,
            'debit_amount': debit,
            'credit_amount': credit,
            'transaction_date': date.strftime('%Y-%m-%d'),
            'interco_flag': interco
        })
    
    return pd.DataFrame(data)

def generate_financial_report_data(n=100):
    report_types = ['BS', 'PL', 'CF']
    periods = [f'2024-Q{i}' for i in range(1, 5)]
    entities = [f'ENT{i:03d}' for i in range(1, 11)]
    metrics = ['Total Assets', 'Total Liabilities', 'Revenue', 'Expenses', 'Net Income', 'Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow']
    
    data = []
    for _ in range(n):
        report_type = random.choice(report_types)
        period = random.choice(periods)
        entity = random.choice(entities)
        metric = random.choice(metrics)
        value = round(random.uniform(100000, 10000000), 2)
        prev_value = round(value * random.uniform(0.8, 1.2), 2)
        variance = value - prev_value
        variance_pct = round((variance / prev_value) * 100, 2)
        
        data.append({
            'report_type': report_type,
            'period': period,
            'entity': entity,
            'metric': metric,
            'value': value,
            'previous_value': prev_value,
            'variance': variance,
            'variance_percentage': variance_pct
        })
    
    return pd.DataFrame(data)

def generate_audit_log_data(n=100):
    journal_ids = [f'J{i:03d}' for i in range(1, 51)]
    accounts = [f'ACC{i:04d}' for i in range(1, 21)]
    users = [f'USER{i}' for i in range(1, 11)]
    actions = ['CREATE', 'UPDATE', 'DELETE', 'VERIFY', 'APPROVE']
    statuses = ['PENDING', 'REVIEWED', 'APPROVED', 'REJECTED']
    
    data = []
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    for _ in range(n):
        journal_id = random.choice(journal_ids)
        date = random.choice(generate_dates(start_date, end_date, 1))
        account = random.choice(accounts)
        amount = round(random.uniform(1000, 100000), 2)
        user = random.choice(users)
        action = random.choice(actions)
        change = f"{action} by {user}"
        status = random.choice(statuses)
        
        data.append({
            'journal_id': journal_id,
            'transaction_date': date.strftime('%Y-%m-%d'),
            'account_code': account,
            'amount': amount,
            'user_id': user,
            'action_type': action,
            'change_details': change,
            'audit_status': status
        })
    
    return pd.DataFrame(data)

def generate_journal_entries_data(n=100):
    journal_ids = [f'J{i:03d}' for i in range(1, 51)]
    accounts = [f'ACC{i:04d}' for i in range(1, 21)]
    account_names = ['Cash', 'Accounts Receivable', 'Inventory', 'Fixed Assets', 'Accounts Payable', 'Loans', 'Equity', 'Revenue', 'Expenses']
    entities = [f'ENT{i:03d}' for i in range(1, 11)]
    
    data = []
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    for _ in range(n):
        journal_id = random.choice(journal_ids)
        date = random.choice(generate_dates(start_date, end_date, 1))
        account = random.choice(accounts)
        account_name = random.choice(account_names)
        debit = round(random.uniform(1000, 100000), 2) if random.random() > 0.5 else 0
        credit = round(random.uniform(1000, 100000), 2) if debit == 0 else 0
        entity = random.choice(entities)
        description = f"Transaction for {account_name}"
        
        data.append({
            'journal_id': journal_id,
            'transaction_date': date.strftime('%Y-%m-%d'),
            'account_code': account,
            'account_name': account_name,
            'debit_amount': debit,
            'credit_amount': credit,
            'entity': entity,
            'description': description
        })
    
    return pd.DataFrame(data)

def generate_interco_reconciliation_data(n=100):
    entities = [f'ENT{i:03d}' for i in range(1, 11)]
    accounts = [f'ACC{i:04d}' for i in range(1, 21)]
    statuses = ['PENDING', 'MATCHED', 'UNMATCHED']
    
    data = []
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    for _ in range(n):
        transaction_id = f'IC{random.randint(1, 999):03d}'
        entity_from = random.choice(entities)
        entity_to = random.choice([e for e in entities if e != entity_from])
        account = random.choice(accounts)
        amount = round(random.uniform(1000, 100000), 2)
        date = random.choice(generate_dates(start_date, end_date, 1))
        status = random.choice(statuses)
        reconciliation_id = f'RC{random.randint(1, 999):03d}'
        
        data.append({
            'transaction_id': transaction_id,
            'entity_from': entity_from,
            'entity_to': entity_to,
            'account_code': account,
            'amount': amount,
            'transaction_date': date.strftime('%Y-%m-%d'),
            'status': status,
            'reconciliation_id': reconciliation_id
        })
    
    return pd.DataFrame(data)

def generate_gst_validation_data(n=100):
    gstins = [f'{random.randint(10, 99)}AAAAA{random.randint(1000, 9999)}A{random.randint(1, 9)}Z{random.randint(1, 9)}' for _ in range(10)]
    vendors = [f'Vendor_{i}' for i in range(1, 21)]
    
    data = []
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    for _ in range(n):
        invoice_number = f'INV{random.randint(1, 9999):04d}'
        gstin = random.choice(gstins)
        vendor = random.choice(vendors)
        date = random.choice(generate_dates(start_date, end_date, 1))
        taxable = round(random.uniform(10000, 1000000), 2)
        cgst = round(taxable * 0.09, 2)
        sgst = round(taxable * 0.09, 2)
        igst = 0 if random.random() > 0.3 else round(taxable * 0.18, 2)
        total = taxable + cgst + sgst + igst
        itc = 'Y' if random.random() > 0.2 else 'N'
        
        data.append({
            'invoice_number': invoice_number,
            'gstin': gstin,
            'vendor_name': vendor,
            'invoice_date': date.strftime('%Y-%m-%d'),
            'taxable_amount': taxable,
            'cgst_amount': cgst,
            'sgst_amount': sgst,
            'igst_amount': igst,
            'total_amount': total,
            'itc_eligible': itc
        })
    
    return pd.DataFrame(data)

def generate_tds_validation_data(n=100):
    pans = [f'{chr(random.randint(65, 90))}{chr(random.randint(65, 90))}{chr(random.randint(65, 90))}{chr(random.randint(65, 90))}{chr(random.randint(65, 90))}{random.randint(1000, 9999)}{chr(random.randint(65, 90))}' for _ in range(20)]
    vendors = [f'Vendor_{i}' for i in range(1, 21)]
    sections = ['194C', '194J', '194H', '194I', '194D']
    
    data = []
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    for _ in range(n):
        challan = f'CH{random.randint(1, 9999):04d}'
        pan = random.choice(pans)
        vendor = random.choice(vendors)
        section = random.choice(sections)
        payment = round(random.uniform(10000, 1000000), 2)
        tds = round(payment * random.uniform(0.01, 0.1), 2)
        date = random.choice(generate_dates(start_date, end_date, 1))
        status = random.choice(['PAID', 'PENDING', 'REJECTED'])
        
        data.append({
            'challan_number': challan,
            'pan': pan,
            'vendor_name': vendor,
            'section_code': section,
            'tds_amount': tds,
            'transaction_date': date.strftime('%Y-%m-%d'),
            'payment_amount': payment,
            'status': status
        })
    
    return pd.DataFrame(data)

def generate_traces_data(n=100):
    pans = [f'{chr(random.randint(65, 90))}{chr(random.randint(65, 90))}{chr(random.randint(65, 90))}{chr(random.randint(65, 90))}{chr(random.randint(65, 90))}{random.randint(1000, 9999)}{chr(random.randint(65, 90))}' for _ in range(20)]
    vendors = [f'Vendor_{i}' for i in range(1, 21)]
    sections = ['194C', '194J', '194H', '194I', '194D']
    
    data = []
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    for _ in range(n):
        challan = f'CH{random.randint(1, 9999):04d}'
        pan = random.choice(pans)
        vendor = random.choice(vendors)
        section = random.choice(sections)
        tds = round(random.uniform(1000, 100000), 2)
        date = random.choice(generate_dates(start_date, end_date, 1))
        status = random.choice(['ACCEPTED', 'PENDING', 'REJECTED'])
        ack = f'ACK{random.randint(1, 9999):04d}' if status == 'ACCEPTED' else ''
        
        data.append({
            'challan_number': challan,
            'pan': pan,
            'vendor_name': vendor,
            'section_code': section,
            'tds_amount': tds,
            'transaction_date': date.strftime('%Y-%m-%d'),
            'status': status,
            'acknowledgement_number': ack
        })
    
    return pd.DataFrame(data)

def main():
    # Create a directory for CSV files
    csv_dir = "csv_files"
    ensure_directory(csv_dir)

    # Generate and save each dataset
    datasets = {
        'provision_planning.csv': generate_provision_planning_data,
        'consolidation.csv': generate_consolidation_data,
        'financial_report.csv': generate_financial_report_data,
        'audit_log.csv': generate_audit_log_data,
        'journal_entries.csv': generate_journal_entries_data,
        'interco_reconciliation.csv': generate_interco_reconciliation_data,
        'gst_validation.csv': generate_gst_validation_data,
        'tds_validation.csv': generate_tds_validation_data,
        'traces_data.csv': generate_traces_data
    }

    for filename, generator in datasets.items():
        df = generator(100)  # Generate 100 rows
        filepath = os.path.join(csv_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Created: {filepath} with {len(df)} rows")

if __name__ == "__main__":
    main() 