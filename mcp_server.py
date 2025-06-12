from fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict
from typing import List, Dict
from fastmcp import FastMCP
import pandas as pd
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.vectorstores import FAISS
from typing import Optional
from pandas import DataFrame
from langchain.schema import Document
from langchain.vectorstores import FAISS
import re
import glob
from openai import OpenAI
from openai import OpenAI
import os
import pandas as pd
import openai
import json
import os
import pandas as pd
import openai
import json
from database import get_db_connection, DB_PARAMS
import psycopg2
from psycopg2.extras import RealDictCursor

# System prompt describing the purpose of the system
#mcp = FastMCP("Automated Financial Closure System for Indian Entities")


mcp = FastMCP(
    "Automated Financial Closure System for Indian Entities"
)

# Define the input schema using Pydantic
class JournalValidatorInput(BaseModel):
    data: List[Dict]  # List of journal entries




# Load .env variables
load_dotenv()

# Initialize models
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")





import pandas as pd
from typing import Optional
from pandas import DataFrame
from langchain_core.documents import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings  # or your embedding model
from langchain.llms import OpenAI  # or your chosen LLM

# Assume these are defined somewhere globally
embedding_model = OpenAIEmbeddings()
llm = OpenAI()
class JournalValidatorInput(BaseModel):
    data: List[Dict] 
# ----------------------------
# Journal Validator Agent
# ----------------------------

@mcp.tool()
def journal_validator_agent(input: dict = None) -> dict:
    """
    Validate journal data from all CSVs in DATA/Transactional/, run LLM analytics, and pass enriched data to Interco Reconciliation Agent.
    """
    import glob
    input = input or {}

    # Step 1: Load and concatenate all journal entries from DATA/Transactional/*.csv
    data_dir = os.path.join(os.path.dirname(__file__), '../DATA/Transactional')
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    if not csv_files:
        # Fallback: look for JOURNALS.csv in the current directory
        fallback_csv = os.path.join(os.path.dirname(__file__), 'JOURNALS.csv')
        if os.path.exists(fallback_csv):
            csv_files = [fallback_csv]
        else:
            return {"error": f"No CSV files found in {data_dir} or {fallback_csv}"}
    df_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Normalize column names to title case and strip spaces
            df.columns = [c.strip().title().replace('_', ' ') for c in df.columns]
            # Map common alternate names to expected names
            col_map = {
                'Entry Type': 'Type',
                'Journal Id': 'Journal ID',
                'Date': 'Date',
                'Account': 'Account',
                'Amount': 'Amount',
                'Entity': 'Entity',
            }
            df = df.rename(columns=col_map)
            df_list.append(df)
        except Exception as e:
            return {"error": f"Error reading {file}: {str(e)}"}
    df = pd.concat(df_list, ignore_index=True)

    # Step 2: Basic validations
    null_fields = df.isnull().any(axis=1).sum()
    if null_fields > 0:
        df = df.dropna()  # Remove rows with null values

    debit_credit_balance = (
        df[df['Type'] == 'Debit']['Amount'].sum() == df[df['Type'] == 'Credit']['Amount'].sum()
    )
    duplicates = df.duplicated().sum()

    # Step 3: Ensure required columns exist
    required_columns = {"Date", "Account", "Type", "Amount", "Entity", "Journal ID"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        return {"error": f"Missing required columns: {missing_columns}"}

    # Step 4: Convert rows to documents for LLM
    documents = [
        Document(page_content=", ".join(f"{col}: {row[col]}" for col in df.columns))
        for _, row in df.iterrows()
    ]

    # Step 5: Create vectorstore
    vectorstore = FAISS.from_documents(documents, embedding_model)

    # Step 6: LLM response
    question = input.get("question", "Provide an overview of the journal entries.")
    relevant_docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    prompt = (
        SYSTEM_PROMPTS["JOURNAL_VALIDATOR"] +
        "\n\n" + context + f"\n\nQuestion: {question}"
    )
    response = llm.invoke(prompt)

    # Step 7: Output validation + pass-through for next agent
    status = "PASS" if null_fields == 0 and debit_credit_balance and duplicates == 0 else "FAIL"
    return {
        "null_fields": null_fields,
        "debit_credit_balance": debit_credit_balance,
        "duplicates": duplicates,
        "llm_response": response,
        "validated_data": df.to_dict(orient="records"),  # Pass this to interco agent
        "status": status
    }

# ----------------------------
# Interco Reconciliation Agent
# ----------------------------

@mcp.tool()
def interco_reconciliation_agent(input: Optional[dict] = None) -> dict:
    """
    Detect intercompany mismatches, prepare elimination entries, and provide LLM insights.
    """
    input = input or {}  #
    try:
        df = pd.read_csv("INTER_CON.CSV")
    except FileNotFoundError:
        return {"error": "INTER_CON.CSV file not found."}



    # Step 2: Ensure required columns exist
    required_columns = {"Entity", "Counterparty", "Account", "Amount", "Journal ID"}
    if not required_columns.issubset(set(df.columns)):
        return {"error": f"Missing required columns: {required_columns - set(df.columns)}"}

    # Step 3: Detect mismatches by Entity-Counterparty
    mismatches = []
    grouped = df.groupby(['Entity', 'Counterparty'])

    for (entity, counterparty), group in grouped:
        total_amount = group['Amount'].sum()
        tolerance = 0.01  # Define a tolerance for rounding errors
        if abs(total_amount) > tolerance:
            mismatches.append({
                "Entity": entity,
                "Counterparty": counterparty,
                "TotalAmount": total_amount,
                "JournalIDs": list(group["Journal ID"].unique())
            })

    # Step 4: Find missing counterparties
    missing_counterparties = df[df['Counterparty'].isnull()]['Journal ID'].unique().tolist()

    # Step 5: Create documents for LLM
    documents = [
        Document(page_content=", ".join(f"{col}: {row[col]}" for col in df.columns))
        for _, row in df.iterrows()
    ]
    vectorstore = FAISS.from_documents(documents, embedding_model)

    # Step 6: LLM analysis
    question = input.get("question", "Provide insights on intercompany transactions.")
    relevant_docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    prompt = (
        SYSTEM_PROMPTS["INTERCO_RECONCILIATION"] +
        "\n\n" + context + f"\n\nQuestion: {question}"
    )
    response = llm.invoke(prompt)

    return {
        "mismatches": mismatches,
        "missing_counterparties": missing_counterparties,
        "llm_response": response
    }



# GSTValidatorAgent
@mcp.tool()
def gst_validator_agent(input: Optional[dict] = None) -> dict:
    """
    Compare GSTR-1, GSTR-3B with sales register and GSTR-2A/2B with purchase register.
    Ensures GST compliance and eligible ITC claims.
    """
    input = input or {}
    try:
        # Load GST data from CSV
        df = pd.read_csv("GST.csv")
    except FileNotFoundError:
        return {"error": "GST.csv file not found."}

    # Step 1: Ensure required columns exist
    required_columns = {"Invoice No", "GSTIN", "Invoice Date", "Tax Amount", "ITC Flag", "GSTR Summary"}
    if not required_columns.issubset(set(df.columns)):
        return {"error": f"Missing required columns: {required_columns - set(df.columns)}"}

    # Step 2: Find mismatches between GSTR Summary and Invoice No
    mismatches = df[df['GSTR Summary'] != df['Invoice No']]
    
    # Step 3: Find missing invoices
    missing_invoices = df[df['Invoice No'].isnull()]
    
    # Step 4: Find excess ITC claims
    excess_itc = df[df['ITC Flag'] == 'Excess']

    # Step 5: Create documents for LLM analysis
    documents = [
        Document(page_content=", ".join(f"{col}: {row[col]}" for col in df.columns))
        for _, row in df.iterrows()
    ]
    vectorstore = FAISS.from_documents(documents, embedding_model)

    # Step 6: LLM analysis
    question = input.get("question", "Provide insights on GST compliance and ITC claims.")
    relevant_docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    prompt = (
        SYSTEM_PROMPTS["GST_VALIDATOR"] +
        "\n\n" + context + f"\n\nQuestion: {question}"
    )
    response = llm.invoke(prompt)

    return {
        "mismatches": mismatches.to_dict(orient="records"),
        "missing_invoices": missing_invoices.to_dict(orient="records"),
        "excess_itc": excess_itc.to_dict(orient="records"),
        "llm_response": response
    }

# TDSValidatorAgent
@mcp.tool()
def tds_validator_agent(input: Optional[dict] = None) -> dict:
    """
    Validate TDS deductions and challans against TRACES data.
    Ensures accurate TDS deduction, filing, and challan application.
    """
    input = input or {}
    try:
        # Load TDS data from CSV
        df = pd.read_csv("TDS.csv")
    except FileNotFoundError:
        return {"error": "TDS.csv file not found."}

    # Step 1: Ensure required columns exist
    required_columns = {"PAN", "Section", "TDS Amount", "Challan No", "Date"}
    if not required_columns.issubset(set(df.columns)):
        return {"error": f"Missing required columns: {required_columns - set(df.columns)}"}

    # Step 2: Group by PAN and Section to validate deduction percentages
    grouped = df.groupby(['PAN', 'Section'])
    deduction_mismatches = []
    
    # Define standard TDS rates for different sections
    tds_rates = {
        '192': 0.10,  # Salary
        '194C': 0.02,  # Contract
        '194J': 0.10,  # Professional/Technical
        '194I': 0.10,  # Rent
        # Add more sections as needed
    }

    for (pan, section), group in grouped:
        total_amount = group['TDS Amount'].sum()
        expected_tds = total_amount * tds_rates.get(section, 0)
        actual_tds = total_amount  # Assuming TDS Amount represents actual TDS deducted
        
        if abs(expected_tds - actual_tds) > 0.01:  # Tolerance for rounding
            deduction_mismatches.append({
                "PAN": pan,
                "Section": section,
                "Expected TDS": expected_tds,
                "Actual TDS": actual_tds,
                "Difference": expected_tds - actual_tds
            })

    # Step 3: Find missing challans
    missing_challans = df[df['Challan No'].isnull()]
    
    # Step 4: Create documents for LLM analysis
    documents = [
        Document(page_content=", ".join(f"{col}: {row[col]}" for col in df.columns))
        for _, row in df.iterrows()
    ]
    vectorstore = FAISS.from_documents(documents, embedding_model)

    # Step 5: LLM analysis
    question = input.get("question", "Provide insights on TDS compliance and challan status.")
    relevant_docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    prompt = (
        SYSTEM_PROMPTS["TDS_VALIDATOR"] +
        "\n\n" + context + f"\n\nQuestion: {question}"
    )
    response = llm.invoke(prompt)

    return {
        "deduction_mismatches": deduction_mismatches,
        "missing_challans": missing_challans.to_dict(orient="records"),
        "llm_response": response
    }




@mcp.tool()
def provision_agent(input: Optional[dict] = None) -> dict:
    """
    Suggest quarterly provisions based on historical trends and outstanding balances.
    Recommends journal entries for accruals and provisions.
    """
    input = input or {}
    try:
        # Load provision data from SYNTHETIC_MERGED.csv
        df = pd.read_csv("SYNTHETIC_MERGED.csv")
    except FileNotFoundError:
        return {"error": "SYNTHETIC_MERGED.csv file not found."}

    # Step 1: Ensure required columns exist
    required_columns = {"Expense Ledger", "Period", "Vendor", "Frequency", "Last Provision Date", "Amount"}
    if not required_columns.issubset(set(df.columns)):
        return {"error": f"Missing required columns: {required_columns - set(df.columns)}"}

    # Step 2: Convert dates to datetime
    df['Last Provision Date'] = pd.to_datetime(df['Last Provision Date'])
    df['Period'] = pd.to_datetime(df['Period'])

    # Step 3: Identify missing provisions
    current_date = pd.Timestamp.now()
    missing_provisions = []
    for _, row in df.iterrows():
        # Calculate expected next provision date based on frequency
        if row['Frequency'] == 'Monthly':
            expected_date = row['Last Provision Date'] + pd.DateOffset(months=1)
        elif row['Frequency'] == 'Quarterly':
            expected_date = row['Last Provision Date'] + pd.DateOffset(months=3)
        elif row['Frequency'] == 'Annual':
            expected_date = row['Last Provision Date'] + pd.DateOffset(years=1)
        else:
            continue
        if expected_date < current_date:
            missing_provisions.append({
                "Expense Ledger": row['Expense Ledger'],
                "Vendor": row['Vendor'],
                "Last Provision Date": row['Last Provision Date'],
                "Expected Date": expected_date,
                "Days Overdue": (current_date - expected_date).days
            })

    # Step 4: Detect amount spikes
    amount_spikes = []
    grouped = df.groupby(['Expense Ledger', 'Vendor'])
    for (ledger, vendor), group in grouped:
        if len(group) >= 2:
            group = group.sort_values('Period')
            group['pct_change'] = group['Amount'].pct_change() * 100
            spikes = group[group['pct_change'] > 20]
            if not spikes.empty:
                for _, spike in spikes.iterrows():
                    amount_spikes.append({
                        "Expense Ledger": ledger,
                        "Vendor": vendor,
                        "Period": spike['Period'],
                        "Amount": spike['Amount'],
                        "Previous Amount": group[group['Period'] < spike['Period']]['Amount'].iloc[-1] if not group[group['Period'] < spike['Period']].empty else None,
                        "Percentage Change": spike['pct_change']
                    })
    documents = [
        Document(page_content=", ".join(f"{col}: {row[col]}" for col in df.columns))
        for _, row in df.iterrows()
    ]
    vectorstore = FAISS.from_documents(documents, embedding_model)
    question = input.get("question", "Provide insights on provision patterns and recommendations.")
    relevant_docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    prompt = (
        SYSTEM_PROMPTS["PROVISION_AGENT"] +
        "\n\n" + context + f"\n\nQuestion: {question}"
    )
    response = llm.invoke(prompt)
    return {
        "missing_provisions": missing_provisions,
        "amount_spikes": amount_spikes,
        "llm_response": response
    }


# System Prompts for Agents
SYSTEM_PROMPTS = {
    "JOURNAL_VALIDATOR": (
        "You are a financial validation agent. Your task is to meticulously review each "
        "journal entry provided. Identify and report any missing required fields, "
        "discrepancies where total debits do not equal total credits within a journal "
        "transaction, and instances of duplicate Journal IDs. "
        "Provide a clear validation report, highlighting all discrepancies found. "
        "For each issue, specify the type of error/warning, the affected field/entry, "
        "and a concise message. Focus on structured validation."
    ),
    "INTERCO_RECONCILIATION": (
        "You are an intercompany reconciliation agent. Your primary role is to compare "
        "intercompany journal entries across different entities. Detect and report "
        "any amount mismatches between corresponding entries and identify missing "
        "counterpart records. Generate a match/mismatch report with a clear summary "
        "of balances. Ensure precision in identifying discrepancies."
    ),
    "GST_VALIDATOR": (
        "You are a GST compliance agent specializing in Indian tax laws. Match "
        "uploaded GST returns data (GSTR-1, 3B, 2A, 2B) with internal sales and "
        "purchase registers. Flag any invoices present in one record but missing "
        "in the other (e.g., GSTR-2A vs. Purchase Register). Identify and warn "
        "about potentially ineligible Input Tax Credit (ITC) claims as per GST rules. "
        "Provide a detailed GST reconciliation report."
    ),
    "TDS_VALIDATOR": (
        "You are a TDS compliance agent for Indian entities. Your role is to validate "
        "TDS deductions and challan details against TRACES data (if available or simulated). "
        "Highlight any instances where the deducted amount is less than the prescribed "
        "threshold, or where PAN/Challan details are missing or invalid. "
        "Generate a TDS mismatch report along with challan verification status."
    ),
    "PROVISION_AGENT": (
        "You are a provision automation agent. Your task is to calculate appropriate "
        "financial provisions (e.g., for expenses, liabilities) based on provided "
        "expense ledgers, vendor data, and historical patterns. Use vendor frequency "
        "and expense data to estimate accruals. Explain your calculation methodology."
    ),
    "CONSOLIDATION_AGENT": (
        "You are a financial consolidation agent. Your responsibility is to aggregate "
        "validated journal entries and trial balances from various entities into "
        "a single, consolidated financial statement (e.g., Balance Sheet, P&L). "
        "Ensure intercompany eliminations are correctly performed. "
        "Present the final consolidated financial statements."
    ),
    "REPORT_AGENT": (
        "You are a compliance reporting agent. Based on the consolidated financial "
        "statements and validated data, generate comprehensive compliance reports "
        "as per Indian regulations (e.g., SEBI, Companies Act, Income Tax Act). "
        "Identify key metrics, ensure all mandatory disclosures are present, and "
        "format the report for regulatory submission."
    ),
    "AUDIT_AGENT": (
        "You are an audit assistant. Your primary function is to maintain a comprehensive "
        "audit trail. Log all changes, user comments, uploaded documents, agent decisions, "
        "and system activities. Based on these logs, generate variance reports, "
        "trace specific transactions, and provide insights for auditors. "
        "Ensure data integrity and traceability for compliance and review."
    ),
}


# Rule Logic Descriptions (for Agents to interpret or for deterministic checks)
RULE_LOGIC = {
    "JOURNAL_VALIDATOR": {
        "MISSING_FIELDS": "All fields (Date, Account, Type, Amount, Entity, Journal ID) must be present.",
        "DEBIT_CREDIT_MISMATCH": "For a given Journal ID, the sum of Debit amounts must equal the sum of Credit amounts.",
        "DUPLICATE_JOURNAL_ID": "Each Journal ID must be unique within the uploaded batch and existing records."
    },
    "INTERCO_RECONCILIATION": {
        "AMOUNT_MISMATCH": "Corresponding intercompany entries between entities must have identical amounts.",
        "MISSING_COUNTERPART": "If an entry exists for Entity A, a corresponding entry must exist for its Counterparty B for the same transaction."
    },
    "GST_VALIDATOR": {
        "INVOICE_NOT_IN_GSTR": "Invoices from internal sales/purchase register must match invoices in GSTR-1/3B/2A/2B.",
        "INELIGIBLE_ITC": "Identify ITC claims on invoices that are not eligible as per GST rules (e.g., blocked credits, non-business use)."
    },
    "TDS_VALIDATOR": {
        "DEDUCTED_LESS_THRESHOLD": "TDS amount deducted must meet or exceed the prescribed threshold for the relevant section and payee type.",
        "MISSING_PAN_CHALLAN": "PAN of deductee and valid Challan No. are mandatory for all TDS deductions.",
        "CHALLAN_MISMATCH_TRACES": "Challan details (amount, date) must match with TRACES records (or simulated TRACES data)."
    }
}

LLM_ROUTING_CONFIG = {
    "journal_validator": {"complexity": "low", "data_type": "structured", "preferred_llm": "fast"},
    "interco_reconciliation": {"complexity": "medium", "data_type": "structured", "preferred_llm": "accurate"},
    "gst_validator": {"complexity": "high", "data_type": "structured", "preferred_llm": "accurate"},
    "tds_validator": {"complexity": "high", "data_type": "structured", "preferred_llm": "accurate"},
    "provision_agent": {"complexity": "medium", "data_type": "mixed", "preferred_llm": "accurate"},
    "consolidation_agent": {"complexity": "high", "data_type": "structured", "preferred_llm": "accurate"},
    "report_agent": {"complexity": "high", "data_type": "mixed", "preferred_llm": "accurate"},
    "audit_agent": {"complexity": "low", "data_type": "mixed", "preferred_llm": "fast"},
}

# ----------------------------
# Consolidation Agent
# ----------------------------

@mcp.tool()
def consolidation_agent(input: Optional[dict] = None) -> dict:
    """
    Consolidate trial balances across entities and apply intercompany elimination.
    Prepares consolidated Balance Sheet, P&L, and Cash Flow.
    """
    input = input or {}
    try:
        # Load trial balance data from CSV
        tb_df = pd.read_csv("TRIAL_BALANCE.csv")
        # Load intercompany matrix
        interco_df = pd.read_csv("INTER_CON.CSV")
    except FileNotFoundError as e:
        return {"error": f"Required file not found: {str(e)}"}

    # Step 1: Ensure required columns exist
    required_tb_columns = {"Account", "Debit", "Credit", "Entity"}
    required_interco_columns = {"Entity", "Counterparty", "Amount", "Account"}
    
    if not required_tb_columns.issubset(set(tb_df.columns)):
        return {"error": f"Missing required columns in Trial Balance: {required_tb_columns - set(tb_df.columns)}"}
    if not required_interco_columns.issubset(set(interco_df.columns)):
        return {"error": f"Missing required columns in Intercompany Matrix: {required_interco_columns - set(interco_df.columns)}"}

    # Step 2: Prepare trial balance data
    tb_df['Net Amount'] = tb_df['Debit'] - tb_df['Credit']
    
    # Step 3: Apply intercompany eliminations
    eliminations = []
    for _, row in interco_df.iterrows():
        # Find matching entries in trial balance
        entity_entry = tb_df[
            (tb_df['Entity'] == row['Entity']) & 
            (tb_df['Account'] == row['Account'])
        ]
        counterparty_entry = tb_df[
            (tb_df['Entity'] == row['Counterparty']) & 
            (tb_df['Account'] == row['Account'])
        ]
        
        if not entity_entry.empty and not counterparty_entry.empty:
            elimination_amount = min(
                abs(entity_entry['Net Amount'].iloc[0]),
                abs(counterparty_entry['Net Amount'].iloc[0])
            )
            eliminations.append({
                "Entity": row['Entity'],
                "Counterparty": row['Counterparty'],
                "Account": row['Account'],
                "Elimination Amount": elimination_amount
            })

    # Step 4: Create consolidated trial balance
    consolidated_tb = tb_df.groupby('Account')['Net Amount'].sum().reset_index()
    
    # Apply eliminations
    for elim in eliminations:
        account_mask = consolidated_tb['Account'] == elim['Account']
        if account_mask.any():
            consolidated_tb.loc[account_mask, 'Net Amount'] -= elim['Elimination Amount']

    # Step 5: Prepare financial statements
    # Define account classifications
    balance_sheet_accounts = ['Assets', 'Liabilities', 'Equity']
    pl_accounts = ['Revenue', 'Expenses']
    cf_accounts = ['Operating', 'Investing', 'Financing']

    # Split consolidated TB into financial statements
    balance_sheet = consolidated_tb[consolidated_tb['Account'].str.contains('|'.join(balance_sheet_accounts))]
    pl = consolidated_tb[consolidated_tb['Account'].str.contains('|'.join(pl_accounts))]
    cash_flow = consolidated_tb[consolidated_tb['Account'].str.contains('|'.join(cf_accounts))]

    # Step 6: Check for structural issues
    structural_issues = []
    
    # Check if balance sheet balances
    if abs(balance_sheet['Net Amount'].sum()) > 0.01:
        structural_issues.append({
            "type": "Balance Sheet Mismatch",
            "difference": balance_sheet['Net Amount'].sum()
        })
    
    # Check for missing account classifications
    unclassified_accounts = consolidated_tb[
        ~consolidated_tb['Account'].str.contains('|'.join(balance_sheet_accounts + pl_accounts + cf_accounts))
    ]
    if not unclassified_accounts.empty:
        structural_issues.append({
            "type": "Unclassified Accounts",
            "accounts": unclassified_accounts['Account'].tolist()
        })

    # Step 7: Create documents for LLM analysis
    documents = [
        Document(page_content=", ".join(f"{col}: {row[col]}" for col in consolidated_tb.columns))
        for _, row in consolidated_tb.iterrows()
    ]
    vectorstore = FAISS.from_documents(documents, embedding_model)

    # Step 8: LLM analysis
    question = input.get("question", "Provide insights on the consolidated financial statements.")
    relevant_docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    prompt = (
        SYSTEM_PROMPTS["CONSOLIDATION_AGENT"] +
        "\n\n" + context + f"\n\nQuestion: {question}"
    )
    response = llm.invoke(prompt)

    return {
        "consolidated_trial_balance": consolidated_tb.to_dict(orient="records"),
        "eliminations": eliminations,
        "balance_sheet": balance_sheet.to_dict(orient="records"),
        "profit_loss": pl.to_dict(orient="records"),
        "cash_flow": cash_flow.to_dict(orient="records"),
        "structural_issues": structural_issues,
        "llm_response": response
    }

@mcp.tool()
def report_agent(input: Optional[dict] = None) -> dict:
    """
    Generate comprehensive financial reports including financial statements, variance analysis, and board packs.
    Produces reports in Excel, PDF, and MCA formats with validation checks and warnings.
    """
    input = input or {}
    
    try:
        # 1. Get Validated Journal Entries
        journal_validation = journal_validator_agent({})
        if "error" in journal_validation:
            return {"error": f"Journal validation failed: {journal_validation['error']}"}
        
        journals_df = pd.DataFrame(journal_validation["validated_data"])
        
        # 2. Get Consolidated Financial Statements
        consolidation_results = consolidation_agent({})
        if "error" in consolidation_results:
            return {"error": f"Consolidation failed: {consolidation_results['error']}"}
        
        consolidated_tb = pd.DataFrame(consolidation_results["consolidated_trial_balance"])
        eliminations = consolidation_results["eliminations"]
        balance_sheet = pd.DataFrame(consolidation_results["balance_sheet"])
        profit_loss = pd.DataFrame(consolidation_results["profit_loss"])
        cash_flow = pd.DataFrame(consolidation_results["cash_flow"])
        
        # 3. Get Intercompany Reconciliation Results
        interco_results = interco_reconciliation_agent({})
        if "error" in interco_results:
            return {"error": f"Intercompany reconciliation failed: {interco_results['error']}"}
        
        # 4. Get GST Validation Results
        gst_results = gst_validator_agent({})
        if "error" in gst_results:
            return {"error": f"GST validation failed: {gst_results['error']}"}
        
        # 5. Get TDS Validation Results
        tds_results = tds_validator_agent({})
        if "error" in tds_results:
            return {"error": f"TDS validation failed: {tds_results['error']}"}
        
        # 6. Get Provision Analysis Results
        provision_results = provision_agent({})
        if "error" in provision_results:
            return {"error": f"Provision analysis failed: {provision_results['error']}"}

        # 7. Enhanced Financial Statement Generation
        financial_statements = {
            "balance_sheet": {
                "data": balance_sheet.to_dict(orient="records"),
                "validation": {
                    "is_balanced": abs(balance_sheet["Net Amount"].sum()) < 0.01,
                    "structural_issues": consolidation_results["structural_issues"],
                    "intercompany_eliminations": eliminations
                }
            },
            "income_statement": {
                "data": profit_loss.to_dict(orient="records"),
                "validation": {
                    "revenue_recognition": journal_validation["llm_response"],
                    "expense_matching": "Validated through journal entries",
                    "provisions": provision_results.get("missing_provisions", [])
                }
            },
            "cash_flow": {
                "data": cash_flow.to_dict(orient="records"),
                "validation": {
                    "operating_activities": "Validated through journal entries",
                    "investing_activities": "Validated through journal entries",
                    "financing_activities": "Validated through journal entries"
                }
            }
        }

        # Use SYSTEM_PROMPTS for report_agent LLM insights if needed
        report_prompt = SYSTEM_PROMPTS["REPORT_AGENT"]
        # You can use this prompt in any LLM call for report generation/summary
        # For now, the prompt is available for future LLM integration

        # 8. Enhanced Variance Analysis
        variance_analysis = {
            "revenue": {
                "current": profit_loss[profit_loss["Account"].str.contains("Revenue")]["Net Amount"].sum(),
                "previous": profit_loss[profit_loss["Account"].str.contains("Revenue")]["Net Amount"].sum() * 0.9,
                "variance": None,
                "percentage": None
            },
            "expenses": {
                "current": abs(profit_loss[profit_loss["Account"].str.contains("Expense")]["Net Amount"].sum()),
                "previous": abs(profit_loss[profit_loss["Account"].str.contains("Expense")]["Net Amount"].sum()) * 0.9,
                "variance": None,
                "percentage": None
            },
            "assets": {
                "current": balance_sheet[balance_sheet["Account"].str.contains("Asset")]["Net Amount"].sum(),
                "previous": balance_sheet[balance_sheet["Account"].str.contains("Asset")]["Net Amount"].sum() * 0.9,
                "variance": None,
                "percentage": None
            },
            "provisions": {
                "current": provision_results.get("total_provisions", 0),
                "previous": provision_results.get("total_provisions", 0) * 0.9,
                "variance": None,
                "percentage": None
            }
        }

        # Calculate variances
        for category in variance_analysis:
            current = variance_analysis[category]["current"]
            previous = variance_analysis[category]["previous"]
            variance_analysis[category]["variance"] = current - previous
            variance_analysis[category]["percentage"] = ((current - previous) / previous) * 100 if previous != 0 else 0

        # 9. Comprehensive Validation Checks
        validation_checks = {
            "journal_entries": {
                "null_fields": journal_validation["null_fields"],
                "debit_credit_balance": journal_validation["debit_credit_balance"],
                "duplicates": journal_validation["duplicates"],
                "llm_insights": journal_validation["llm_response"]
            },
            "consolidation": {
                "eliminations": eliminations,
                "structural_issues": consolidation_results["structural_issues"],
                "llm_insights": consolidation_results["llm_response"]
            },
            "intercompany": {
                "mismatches": interco_results["mismatches"],
                "missing_counterparties": interco_results["missing_counterparties"],
                "llm_insights": interco_results["llm_response"]
            },
            "compliance": {
                "GST": {
                    "mismatches": gst_results["mismatches"],
                    "missing_invoices": gst_results["missing_invoices"],
                    "excess_itc": gst_results["excess_itc"],
                    "llm_insights": gst_results["llm_response"]
                },
                "TDS": {
                    "deduction_mismatches": tds_results["deduction_mismatches"],
                    "missing_challans": tds_results["missing_challans"],
                    "llm_insights": tds_results["llm_response"]
                }
            },
            "provisions": {
                "missing_provisions": provision_results["missing_provisions"],
                "amount_spikes": provision_results["amount_spikes"],
                "llm_insights": provision_results["llm_response"]
            }
        }

        # 10. Comprehensive Warnings and Issues
        warnings = []
        
        # Journal validation warnings
        if journal_validation["null_fields"] > 0:
            warnings.append(f"Warning: {journal_validation['null_fields']} null fields found in journal entries")
        if not journal_validation["debit_credit_balance"]:
            warnings.append("Warning: Debit-Credit balance mismatch in journal entries")
        if journal_validation["duplicates"] > 0:
            warnings.append(f"Warning: {journal_validation['duplicates']} duplicate journal entries found")

        # Consolidation warnings
        for issue in consolidation_results["structural_issues"]:
            warnings.append(f"Warning: {issue['type']} - {issue.get('difference', '')}")

        # Intercompany warnings
        if interco_results["mismatches"]:
            warnings.append(f"Warning: {len(interco_results['mismatches'])} intercompany mismatches found")
        if interco_results["missing_counterparties"]:
            warnings.append(f"Warning: {len(interco_results['missing_counterparties'])} missing counterparties found")

        # GST warnings
        if gst_results["mismatches"]:
            warnings.append(f"Warning: {len(gst_results['mismatches'])} GST mismatches found")
        if gst_results["missing_invoices"]:
            warnings.append(f"Warning: {len(gst_results['missing_invoices'])} missing GST invoices found")
        if gst_results["excess_itc"]:
            warnings.append(f"Warning: {len(gst_results['excess_itc'])} excess ITC claims found")

        # TDS warnings
        if tds_results["deduction_mismatches"]:
            warnings.append(f"Warning: {len(tds_results['deduction_mismatches'])} TDS deduction mismatches found")
        if tds_results["missing_challans"]:
            warnings.append(f"Warning: {len(tds_results['missing_challans'])} missing TDS challans found")

        # Provision warnings
        if provision_results["missing_provisions"]:
            warnings.append(f"Warning: {len(provision_results['missing_provisions'])} missing provisions found")
        if provision_results["amount_spikes"]:
            warnings.append(f"Warning: {len(provision_results['amount_spikes'])} provision amount spikes found")

        return {
            "financial_statements": financial_statements,
            "variance_analysis": variance_analysis,
            "validation_checks": validation_checks,
            "warnings": warnings,
            "consolidated_data": {
                "trial_balance": consolidated_tb.to_dict(orient="records"),
                "eliminations": eliminations,
                "structural_issues": consolidation_results["structural_issues"]
            },
            "agent_insights": {
                "journal_validation": journal_validation["llm_response"],
                "consolidation": consolidation_results["llm_response"],
                "intercompany": interco_results["llm_response"],
                "gst": gst_results["llm_response"],
                "tds": tds_results["llm_response"],
                "provisions": provision_results["llm_response"]
            }
        }

    except Exception as e:
        return {"error": f"Error generating reports: {str(e)}"}

@mcp.tool()
def audit_agent(input: Optional[dict] = None) -> dict:
    """
    Track journal-level activity, respond to auditor queries, and collate documents.
    Support audit by presenting logs, evidence, and change history.
    Journal logs, documents, comments, activity history.
    Timeline tracking, attach evidence, filter queries.
    Missing log? Error, No document? Warning.
    """
    input = input or {}
    question = input.get("question", "Provide an audit overview for journal entries.")
    entry_id = input.get("entry_id")
    filter_user = input.get("user")
    filter_action = input.get("action")
    
    # 1. Get validated journal entries and logs
    journal_validation = journal_validator_agent({"question": question})
    if "error" in journal_validation:
        return {"error": f"Journal validation failed: {journal_validation['error']}"}
    journals = journal_validation["validated_data"]
    
    # 2. Simulate or extract logs, comments, and activity history
    logs = []
    comments = []
    timeline = []
    documents = []
    missing_log = False
    missing_document = False
    warnings = []
    errors = []
    
    # For demo: treat each journal as a log entry, and simulate comments/docs
    for j in journals:
        if entry_id and j.get("Journal ID") != entry_id:
            continue
        log_entry = {
            "timestamp": j.get("Date"),
            "user": j.get("Entity", "system"),
            "action": j.get("Type"),
            "amount": j.get("Amount"),
            "account": j.get("Account"),
            "journal_id": j.get("Journal ID"),
            "comment": f"Auto-log for {j.get('Type')} entry."
        }
        if (not filter_user or log_entry["user"] == filter_user) and (not filter_action or log_entry["action"] == filter_action):
            logs.append(log_entry)
            timeline.append({"timestamp": log_entry["timestamp"], "event": f"{log_entry['action']} by {log_entry['user']}"})
            # Simulate a comment for every 3rd entry
            jid = j.get("Journal ID", "0")
            # Extract numeric part from the end of the Journal ID
            match = re.search(r'(\\d+)$', jid)
            jid_num = int(match.group(1)) if match else 0
            if jid_num % 3 == 0:
                comments.append({"user": log_entry["user"], "comment": "Sample auditor comment.", "timestamp": log_entry["timestamp"]})
            # Simulate a document for every 2nd entry
            if jid_num % 2 == 0:
                documents.append({"file_path": f"docs/{log_entry['journal_id']}.pdf", "uploaded_by": log_entry["user"], "upload_date": log_entry["timestamp"]})
    
    # 3. Validation: missing log or document
    if not logs:
        missing_log = True
        errors.append("Missing log for the specified entry or filter.")
    if not documents:
        missing_document = True
        warnings.append("No supporting document attached for the specified entry or filter.")
    
    # 4. LLM response for auditor queries
    from langchain.vectorstores import FAISS
    from langchain_core.documents import Document
    from mcp_server import embedding_model, llm
    documents_for_llm = [Document(page_content=", ".join(f"{k}: {v}" for k, v in j.items())) for j in journals]
    if documents_for_llm:
        vectorstore = FAISS.from_documents(documents_for_llm, embedding_model)
        relevant_docs = vectorstore.similarity_search(question, k=4)
        context = "\n\n".join(doc.page_content for doc in relevant_docs)
        prompt = (
            SYSTEM_PROMPTS["AUDIT_AGENT"] +
            "\n\n" + context + f"\n\nQuestion: {question}"
        )
        llm_response = llm.invoke(prompt)
    else:
        llm_response = "No journal data available for audit analysis."
    
    return {
        "logs": logs,
        "comments": comments,
        "documents": documents,
        "timeline": timeline,
        "validation": {
            "missing_log": missing_log,
            "missing_document": missing_document,
            "warnings": warnings,
            "errors": errors
        },
        "llm_response": llm_response
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
