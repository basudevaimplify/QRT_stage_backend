import pandas as pd
from sqlalchemy import create_engine, text
from io import StringIO
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from typing_extensions import TypedDict
from openai import OpenAI
from langgraph.graph import StateGraph, END, START
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import uvicorn
import re
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

client = OpenAI(api_key=OPENAI_API_KEY)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Agent-specific Prompts ---

PROVISION_AGENT_PROMPT = """
You are a Provision Planning Agent. Your role is to:
1. Analyze expense patterns and suggest accrual entries
2. Identify repeating expense patterns
3. Suggest quarterly provisions based on trends
4. Flag missing regular provisions
5. Alert on abnormal jumps (>20%)
6. Provide confidence scores for suggestions
"""

CONSO_AGENT_PROMPT = """
You are a Financial Consolidation Agent. Your role is to:
1. Consolidate trial balances across entities
2. Eliminate intercompany transactions
3. Generate group-level financial statements
4. Flag IC elimination mismatches
5. Alert on unlinked entities
6. Ensure compliance with Ind AS 110
"""

FINSTAT_AGENT_PROMPT = """
You are a Financial Reporting Agent. Your role is to:
1. Generate board packs and statutory reports
2. Create MCA-compliant XMLs
3. Prepare presentation-ready financial reports
4. Flag empty FS fields
5. Validate MCA tags
6. Include variance analysis
"""

AUDIT_AGENT_PROMPT = """
You are an Audit Support Agent. Your role is to:
1. Track journal-level changes
2. Document audit queries
3. Support evidence collation
4. Flag missing documents
5. Alert on unresolved queries (>7 days)
6. Maintain audit trail
"""

JOURNAL_VALIDATOR_PROMPT = """
You are a Financial Validation Agent. Your role is to:
1. Validate journal entries for missing data
2. Check debit-credit matches
3. Detect duplicates
4. Flag missing values
5. Alert on duplicate Journal IDs
6. Generate validation reports
"""

INTERCO_RECON_PROMPT = """
You are an Intercompany Reconciliation Agent. Your role is to:
1. Reconcile intercompany entries
2. Match payables and receivables
3. Flag mismatched amounts (>¥1)
4. Alert on missing counterpart entries
5. Generate reconciliation reports
6. Ensure group financial consistency
"""

GST_VALIDATOR_PROMPT = """
You are a GST Compliance Agent. Your role is to:
1. Match GST returns with internal records
2. Verify GSTR filings alignment
3. Flag missing invoices in GSTR
4. Alert on ineligible ITC claims
5. Generate reconciliation reports
6. Ensure GST compliance
"""

TDS_VALIDATOR_PROMPT = """
You are a TDS Compliance Agent. Your role is to:
1. Validate TDS deductions and challans
2. Match with TRACES data
3. Flag TDS below threshold
4. Alert on missing PAN/Challan
5. Generate verification reports
6. Ensure TDS compliance
"""

# --- FastAPI setup ---

app = FastAPI(title="Financial Processing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---

class ProcessRequest(BaseModel):
    journal_csv: Optional[str] = None
    interco_csv: Optional[str] = None
    consolidation_csv: Optional[str] = None
    gst_csv: Optional[str] = None
    tds_csv: Optional[str] = None
    provision_csv: Optional[str] = None
    expense_ledger: Optional[str] = None
    previous_provisions: Optional[str] = None
    vendor_info: Optional[str] = None
    trial_balance: Optional[str] = None
    ic_mapping: Optional[str] = None
    audit_logs: Optional[str] = None
    supporting_docs: Optional[List[str]] = None
    traces_data: Optional[str] = None

class ProcessResponse(BaseModel):
    status: str
    message: str
    data: Optional[Any] = None
    error: Optional[str] = None
    llm_feedback: Optional[str] = None
    confidence_score: Optional[float] = None
    validation_report: Optional[Dict[str, Any]] = None
    reconciliation_report: Optional[Dict[str, Any]] = None
    audit_trail: Optional[List[Dict[str, Any]]] = None

# --- LangGraph States ---

class ProvisionState(TypedDict):
    expense_ledger: str
    previous_provisions: str
    vendor_info: str
    llm_feedback: str
    status: str
    confidence_score: float

class ConsoState(TypedDict):
    trial_balance: str
    ic_mapping: str
    llm_feedback: str
    status: str
    elimination_logs: List[Dict[str, Any]]

class FinStatState(TypedDict):
    validated_journals: str
    consolidated_fs: str
    variance_data: str
    llm_feedback: str
    status: str
    report_urls: List[str]

class AuditState(TypedDict):
    journal_logs: str
    agent_logs: str
    documents: List[str]
    comments: str
    llm_feedback: str
    status: str
    audit_trail: List[Dict[str, Any]]

class JournalValidatorState(TypedDict):
    journal_entries: str
    llm_feedback: str
    status: str
    validation_report: Dict[str, Any]

class IntercoReconState(TypedDict):
    entity_entries: str
    llm_feedback: str
    status: str
    reconciliation_report: Dict[str, Any]

class GSTValidatorState(TypedDict):
    gst_returns: str
    internal_records: str
    llm_feedback: str
    status: str
    reconciliation_report: Dict[str, Any]

class TDSValidatorState(TypedDict):
    tds_entries: str
    traces_data: str
    llm_feedback: str
    status: str
    verification_report: Dict[str, Any]

# --- OpenAI Client ---


# --- LLM-powered validation/analyze functions ---

def llm_validate(prompt: str, csv_data: str) -> str:
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": csv_data},
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0,
        max_tokens=900
    )
    return response.choices[0].message.content

def validate_journal_entries(state: JournalValidatorState) -> JournalValidatorState:
    feedback = llm_validate(JOURNAL_VALIDATOR_PROMPT, state["journal_entries"])
    validation_report = {
        "missing_data": [],
        "debit_credit_mismatches": [],
        "duplicates": [],
        "warnings": [],
        "summary": ""
    }
    return {**state, "llm_feedback": feedback, "status": "validated", "validation_report": validation_report}

def validate_interco_entries(state: IntercoReconState) -> IntercoReconState:
    feedback = llm_validate(INTERCO_RECON_PROMPT, state["entity_entries"])
    reconciliation_report = {
        "mismatched_amounts": [],
        "missing_counterparts": [],
        "reconciliation_summary": "",
        "suggested_corrections": []
    }
    return {**state, "llm_feedback": feedback, "status": "validated", "reconciliation_report": reconciliation_report}

def validate_consolidation_entries(state: ConsoState) -> ConsoState:
    feedback = llm_validate(CONSO_AGENT_PROMPT, state["trial_balance"])
    elimination_logs = []
    return {**state, "llm_feedback": feedback, "status": "validated", "elimination_logs": elimination_logs}

def validate_gst_entries(state: GSTValidatorState) -> GSTValidatorState:
    feedback = llm_validate(GST_VALIDATOR_PROMPT, state["gst_returns"])
    reconciliation_report = {
        "missing_invoices": [],
        "ineligible_itc": [],
        "reconciliation_summary": "",
        "suggested_corrections": []
    }
    return {**state, "llm_feedback": feedback, "status": "validated", "reconciliation_report": reconciliation_report}

def validate_tds_entries(state: TDSValidatorState) -> TDSValidatorState:
    feedback = llm_validate(TDS_VALIDATOR_PROMPT, state["tds_entries"])
    verification_report = {
        "below_threshold": [],
        "missing_documents": [],
        "verification_summary": "",
        "suggested_corrections": []
    }
    return {**state, "llm_feedback": feedback, "status": "validated", "verification_report": verification_report}

def analyze_provisions(state: ProvisionState) -> ProvisionState:
    feedback = llm_validate(PROVISION_AGENT_PROMPT, state["expense_ledger"])
    confidence_score = 0.0  # Calculate based on data quality and patterns
    return {**state, "llm_feedback": feedback, "status": "analyzed", "confidence_score": confidence_score}

# --- LangGraph workflows ---

def make_workflow(StateType, validate_fn):
    workflow = StateGraph(StateType)
    workflow.add_node("validate", validate_fn)
    workflow.add_edge(START, "validate")
    workflow.add_edge("validate", END)
    return workflow.compile()

journal_workflow = make_workflow(JournalValidatorState, validate_journal_entries)
interco_workflow = make_workflow(IntercoReconState, validate_interco_entries)
consolidation_workflow = make_workflow(ConsoState, validate_consolidation_entries)
gst_workflow = make_workflow(GSTValidatorState, validate_gst_entries)
tds_workflow = make_workflow(TDSValidatorState, validate_tds_entries)
provision_workflow = make_workflow(ProvisionState, analyze_provisions)

# --- New Agent-specific Workflows ---

def validate_finstat_entries(state: FinStatState) -> FinStatState:
    feedback = llm_validate(FINSTAT_AGENT_PROMPT, state["validated_journals"])
    return {**state, "llm_feedback": feedback, "status": "validated"}

def validate_audit_entries(state: AuditState) -> AuditState:
    feedback = llm_validate(AUDIT_AGENT_PROMPT, state["journal_logs"])
    return {**state, "llm_feedback": feedback, "status": "validated"}

finstat_workflow = make_workflow(FinStatState, validate_finstat_entries)
audit_workflow = make_workflow(AuditState, validate_audit_entries)

# --- Database operations ---

def get_provision_planning_data():
    with get_db() as db:
        try:
            result = db.execute(text("SELECT * FROM provision_planning"))
            columns = result.keys()
            return [dict(zip(columns, row)) for row in result.fetchall()]
        except Exception as e:
            print(f"Error fetching provision planning data: {str(e)}")
            return []

def get_consolidation_data():
    with get_db() as db:
        try:
            result = db.execute(text("SELECT * FROM consolidation"))
            columns = result.keys()
            return [dict(zip(columns, row)) for row in result.fetchall()]
        except Exception as e:
            print(f"Error fetching consolidation data: {str(e)}")
            return []

def get_financial_report_data():
    with get_db() as db:
        try:
            # Get data from all financial tables
            financial_data = {
                "provisions": [],
                "consolidation": [],
                "journal_entries": [],
                "interco_reconciliation": [],
                "gst_validation": [],
                "tds_validation": []
            }
            
            # Get provision planning data
            result = db.execute(text("SELECT * FROM provision_planning"))
            columns = result.keys()
            financial_data["provisions"] = [dict(zip(columns, row)) for row in result.fetchall()]
            
            # Get consolidation data
            result = db.execute(text("SELECT * FROM consolidation"))
            columns = result.keys()
            financial_data["consolidation"] = [dict(zip(columns, row)) for row in result.fetchall()]
            
            # Get journal entries
            result = db.execute(text("SELECT * FROM journal_entries"))
            columns = result.keys()
            financial_data["journal_entries"] = [dict(zip(columns, row)) for row in result.fetchall()]
            
            # Get intercompany reconciliation data
            result = db.execute(text("SELECT * FROM interco_reconciliation"))
            columns = result.keys()
            financial_data["interco_reconciliation"] = [dict(zip(columns, row)) for row in result.fetchall()]
            
            # Get GST validation data
            result = db.execute(text("SELECT * FROM gst_validation"))
            columns = result.keys()
            financial_data["gst_validation"] = [dict(zip(columns, row)) for row in result.fetchall()]
            
            # Get TDS validation data
            result = db.execute(text("SELECT * FROM tds_validation"))
            columns = result.keys()
            financial_data["tds_validation"] = [dict(zip(columns, row)) for row in result.fetchall()]
            
            # Check if any data exists
            has_data = any(len(data) > 0 for data in financial_data.values())
            if not has_data:
                return []
            
            # Create summary statistics
            summary = {
                "total_provisions": len(financial_data["provisions"]),
                "total_consolidated_entries": len(financial_data["consolidation"]),
                "total_journal_entries": len(financial_data["journal_entries"]),
                "total_interco_transactions": len(financial_data["interco_reconciliation"]),
                "total_gst_entries": len(financial_data["gst_validation"]),
                "total_tds_entries": len(financial_data["tds_validation"]),
                "data_sources": {
                    "provisions": bool(financial_data["provisions"]),
                    "consolidation": bool(financial_data["consolidation"]),
                    "journal_entries": bool(financial_data["journal_entries"]),
                    "interco_reconciliation": bool(financial_data["interco_reconciliation"]),
                    "gst_validation": bool(financial_data["gst_validation"]),
                    "tds_validation": bool(financial_data["tds_validation"])
                }
            }
            
            # Prepare data for LLM analysis
            llm_prompt = f"""
            Analyze the following financial data and provide a concise summary with bullet points:

            1. Provision Planning:
            {financial_data['provisions']}

            2. Consolidation Data:
            {financial_data['consolidation']}

            3. Journal Entries:
            {financial_data['journal_entries']}

            4. Intercompany Reconciliation:
            {financial_data['interco_reconciliation']}

            5. GST Validation:
            {financial_data['gst_validation']}

            6. TDS Validation:
            {financial_data['tds_validation']}

            Provide a brief summary in this format:

            • Financial Health: [1-2 sentences]
            • Key Metrics: [3-4 bullet points]
            • Critical Issues: [2-3 bullet points]
            • Action Items: [2-3 bullet points]
            • Compliance Status: [1-2 sentences]
            • Data Quality: [1-2 sentences]

            Keep each point brief and focused on the most important information.
            """
            
            # Get LLM analysis
            llm_analysis = llm_validate(FINSTAT_AGENT_PROMPT, llm_prompt)
            
            return {
                "financial_data": financial_data,
                "summary": summary,
                "llm_analysis": llm_analysis
            }
            
        except Exception as e:
            print(f"Error fetching financial report data: {str(e)}")
            return []

def get_audit_log_data():
    with get_db() as db:
        try:
            # Get data from all audit-related tables
            audit_data = {
                "journal_reconciliation": [],
                "interco_reconciliation": [],
                "tax_compliance": {
                    "gst": [],
                    "tds": []
                },
                "provisions": [],
                "consolidation": [],
                "audit_logs": []
            }
            
            # Get journal reconciliation data
            result = db.execute(text("""
                SELECT 
                    j.id,
                    j.journal_id,
                    j.transaction_date,
                    j.account_code,
                    j.account_name,
                    j.debit_amount,
                    j.credit_amount,
                    j.entity,
                    j.description
                FROM journal_entries j
                ORDER BY j.transaction_date DESC
            """))
            columns = result.keys()
            audit_data["journal_reconciliation"] = [dict(zip(columns, row)) for row in result.fetchall()]
            
            # Get intercompany reconciliation data
            result = db.execute(text("""
                SELECT 
                    i.id,
                    i.transaction_id,
                    i.entity_from,
                    i.entity_to,
                    i.account_code,
                    i.amount,
                    i.transaction_date,
                    i.status,
                    i.reconciliation_id
                FROM interco_reconciliation i
                ORDER BY i.transaction_date DESC
            """))
            columns = result.keys()
            audit_data["interco_reconciliation"] = [dict(zip(columns, row)) for row in result.fetchall()]
            
            # Get GST validation data
            result = db.execute(text("""
                SELECT 
                    g.id,
                    g.invoice_number,
                    g.gstin,
                    g.vendor_name,
                    g.invoice_date,
                    g.taxable_amount,
                    g.cgst_amount,
                    g.sgst_amount,
                    g.igst_amount,
                    g.total_amount,
                    g.itc_eligible
                FROM gst_validation g
                ORDER BY g.invoice_date DESC
            """))
            columns = result.keys()
            audit_data["tax_compliance"]["gst"] = [dict(zip(columns, row)) for row in result.fetchall()]
            
            # Get TDS validation data
            result = db.execute(text("""
                SELECT 
                    t.id,
                    t.challan_number,
                    t.pan,
                    t.vendor_name,
                    t.section_code,
                    t.tds_amount,
                    t.transaction_date,
                    t.payment_amount,
                    t.status
                FROM tds_validation t
                ORDER BY t.transaction_date DESC
            """))
            columns = result.keys()
            audit_data["tax_compliance"]["tds"] = [dict(zip(columns, row)) for row in result.fetchall()]
            
            # Get provision planning data
            result = db.execute(text("""
                SELECT 
                    p.id,
                    p.expense_category,
                    p.vendor_name,
                    p.amount,
                    p.frequency,
                    p.last_provision_date,
                    p.expected_date,
                    p.confidence_score
                FROM provision_planning p
                ORDER BY p.expected_date DESC
            """))
            columns = result.keys()
            audit_data["provisions"] = [dict(zip(columns, row)) for row in result.fetchall()]
            
            # Get consolidation data
            result = db.execute(text("""
                SELECT 
                    c.id,
                    c.entity_code,
                    c.account_code,
                    c.account_name,
                    c.debit_amount,
                    c.credit_amount,
                    c.transaction_date,
                    c.interco_flag
                FROM consolidation c
                ORDER BY c.entity_code, c.account_code
            """))
            columns = result.keys()
            audit_data["consolidation"] = [dict(zip(columns, row)) for row in result.fetchall()]
            
            # Get audit logs
            result = db.execute(text("""
                SELECT 
                    a.id,
                    a.journal_id,
                    a.transaction_date,
                    a.account_code,
                    a.amount,
                    a.user_id,
                    a.action_type,
                    a.change_details,
                    a.audit_status
                FROM audit_log a
                ORDER BY a.transaction_date DESC
            """))
            columns = result.keys()
            audit_data["audit_logs"] = [dict(zip(columns, row)) for row in result.fetchall()]
            
            # Check if any data exists
            has_data = (
                len(audit_data["journal_reconciliation"]) > 0 or
                len(audit_data["interco_reconciliation"]) > 0 or
                len(audit_data["tax_compliance"]["gst"]) > 0 or
                len(audit_data["tax_compliance"]["tds"]) > 0 or
                len(audit_data["provisions"]) > 0 or
                len(audit_data["consolidation"]) > 0 or
                len(audit_data["audit_logs"]) > 0
            )
            
            if not has_data:
                return []
            
            # Create summary statistics
            summary = {
                "total_journal_entries": len(audit_data["journal_reconciliation"]),
                "total_interco_transactions": len(audit_data["interco_reconciliation"]),
                "total_gst_entries": len(audit_data["tax_compliance"]["gst"]),
                "total_tds_entries": len(audit_data["tax_compliance"]["tds"]),
                "total_provisions": len(audit_data["provisions"]),
                "total_consolidation_entries": len(audit_data["consolidation"]),
                "total_audit_logs": len(audit_data["audit_logs"]),
                "data_sources": {
                    "journal_reconciliation": len(audit_data["journal_reconciliation"]) > 0,
                    "interco_reconciliation": len(audit_data["interco_reconciliation"]) > 0,
                    "gst_validation": len(audit_data["tax_compliance"]["gst"]) > 0,
                    "tds_validation": len(audit_data["tax_compliance"]["tds"]) > 0,
                    "provisions": len(audit_data["provisions"]) > 0,
                    "consolidation": len(audit_data["consolidation"]) > 0,
                    "audit_logs": len(audit_data["audit_logs"]) > 0
                }
            }
            
            # Prepare data for LLM analysis
            llm_prompt = f"""
            Analyze the following audit data and provide a concise summary with bullet points:

            1. Journal Reconciliation:
            {audit_data['journal_reconciliation']}

            2. Intercompany Reconciliation:
            {audit_data['interco_reconciliation']}

            3. Tax Compliance (GST & TDS):
            {audit_data['tax_compliance']}

            4. Provisions:
            {audit_data['provisions']}

            5. Consolidation:
            {audit_data['consolidation']}

            6. Audit Logs:
            {audit_data['audit_logs']}

            Provide a brief audit summary in this format:

            • Journal Trail: [Key findings about journal entries]
            • Intercompany Status: [Summary of intercompany reconciliations]
            • Tax Compliance: [GST and TDS compliance status]
            • Provisions Review: [Assessment of provisions]
            • Consolidation Check: [Consolidation process review]
            • Audit Trail: [Key audit activities and findings]
            • Critical Issues: [2-3 most important audit findings]
            • Action Items: [2-3 immediate actions needed]

            Keep each point brief and focused on audit-relevant information.
            """
            
            # Get LLM analysis
            llm_analysis = llm_validate(AUDIT_AGENT_PROMPT, llm_prompt)
            
            return {
                "audit_data": audit_data,
                "summary": summary,
                "llm_analysis": llm_analysis
            }
            
        except Exception as e:
            print(f"Error fetching audit log data: {str(e)}")
            return []

def get_journal_entries_data():
    with get_db() as db:
        try:
            result = db.execute(text("SELECT * FROM journal_entries"))
            columns = result.keys()
            return [dict(zip(columns, row)) for row in result.fetchall()]
        except Exception as e:
            print(f"Error fetching journal entries data: {str(e)}")
            return []

def get_interco_reconciliation_data():
    with get_db() as db:
        try:
            result = db.execute(text("SELECT * FROM interco_reconciliation"))
            columns = result.keys()
            return [dict(zip(columns, row)) for row in result.fetchall()]
        except Exception as e:
            print(f"Error fetching intercompany reconciliation data: {str(e)}")
            return []

def get_gst_validation_data():
    with get_db() as db:
        try:
            result = db.execute(text("SELECT * FROM gst_validation"))
            columns = result.keys()
            return [dict(zip(columns, row)) for row in result.fetchall()]
        except Exception as e:
            print(f"Error fetching GST validation data: {str(e)}")
            return []

def get_tds_validation_data():
    with get_db() as db:
        try:
            result = db.execute(text("SELECT * FROM tds_validation"))
            columns = result.keys()
            return [dict(zip(columns, row)) for row in result.fetchall()]
        except Exception as e:
            print(f"Error fetching TDS validation data: {str(e)}")
            return []

def get_traces_data():
    with get_db() as db:
        try:
            result = db.execute(text("SELECT * FROM traces_data"))
            columns = result.keys()
            return [dict(zip(columns, row)) for row in result.fetchall()]
        except Exception as e:
            print(f"Error fetching TRACES data: {str(e)}")
            return []

# --- Endpoints ---

@app.post("/process_journal_entries", response_model=ProcessResponse)
async def process_journal_entries(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")
    try:
        content = await file.read()
        journal_csv = content.decode('utf-8')
        initial_state = JournalValidatorState(journal_entries=journal_csv, llm_feedback="", status="pending")
        result = journal_workflow.invoke(initial_state)
        llm_feedback = result['llm_feedback']
        
        # Check for validation success indicators in the feedback
        validation_success = any(phrase in llm_feedback.lower() for phrase in [
            "no errors",
            "all entries are valid",
            "validated successfully",
            "no duplicate",
            "no missing data"
        ])
        
        if validation_success:
            df = pd.read_csv(StringIO(journal_csv))
            engine = create_engine(DATABASE_URL)
            try:
                journal_ids = tuple(df["journal_id"].unique())
                if journal_ids:
                    query = text("SELECT journal_id FROM journal_entries WHERE journal_id IN :journal_ids")
                    with engine.connect() as conn:
                        result_db = conn.execute(query, {"journal_ids": journal_ids})
                        existing = {row[0] for row in result_db}
                        if existing:
                            return ProcessResponse(
                                status="error",
                                message="",
                                error=f"Duplicate journal_id(s) found in DB: {', '.join(existing)}",
                                llm_feedback=llm_feedback
                            )
                df.to_sql("journal_entries", engine, if_exists="append", index=False)
                query = text("SELECT * FROM journal_entries WHERE journal_id IN :journal_ids ORDER BY journal_id, transaction_date")
                with engine.connect() as conn:
                    result_db = conn.execute(query, {"journal_ids": journal_ids})
                    stored_df = pd.DataFrame(result_db.fetchall(), columns=result_db.keys())
                return ProcessResponse(
                    status="success",
                    message="Data verified and stored successfully",
                    data=stored_df.to_dict(orient='records'),
                    llm_feedback=llm_feedback
                )
            except Exception as e:
                return ProcessResponse(
                    status="error",
                    message="",
                    error=f"Unexpected error during DB insertion: {str(e)}",
                    llm_feedback=llm_feedback
                )
        else:
            return ProcessResponse(
                status="error",
                message="",
                error="Validation issues found.",
                llm_feedback=llm_feedback
            )
    except Exception as e:
        return ProcessResponse(
            status="error",
            message="",
            error=f"Error processing file: {str(e)}"
        )

@app.post("/process_interco_entries", response_model=ProcessResponse)
async def process_interco_entries(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")
    
    try:
        content = await file.read()
        interco_csv = content.decode('utf-8')
        initial_state = IntercoReconState(entity_entries=interco_csv, llm_feedback="", status="pending")
        result = interco_workflow.invoke(initial_state)
        llm_feedback = result['llm_feedback']

        # For intercompany reconciliation, we consider it successful if we get detailed reconciliation findings
        validation_success = any(phrase in llm_feedback.lower() for phrase in [
            "no errors",
            "all entries are valid",
            "validated successfully",
            "reconciliation process",
            "mismatched transactions",
            "missing counterpart entries",
            "reconciliation report"
        ])
        
        if not validation_success:
            return ProcessResponse(
                status="error",
                message="",
                error="Validation issues found.",
                llm_feedback=llm_feedback
            )

        df = pd.read_csv(StringIO(interco_csv))
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
        
        try:
            transaction_ids = tuple(df["transaction_id"].unique())
            if transaction_ids:
                query = text("SELECT transaction_id FROM interco_reconciliation WHERE transaction_id IN :transaction_ids")
                with engine.connect() as conn:
                    result_db = conn.execute(query, {"transaction_ids": transaction_ids})
                    existing = {row[0] for row in result_db}
                    if existing:
                        return ProcessResponse(
                            status="error",
                            message="",
                            error=f"Duplicate transaction_id(s) found in DB: {', '.join(existing)}",
                            llm_feedback=llm_feedback
                        )
            
            df.to_sql("interco_reconciliation", engine, if_exists="append", index=False)
            query = text("SELECT * FROM interco_reconciliation WHERE transaction_id IN :transaction_ids ORDER BY transaction_id, transaction_date")
            with engine.connect() as conn:
                result_db = conn.execute(query, {"transaction_ids": transaction_ids})
                stored_df = pd.DataFrame(result_db.fetchall(), columns=result_db.keys())
            
            # Extract reconciliation report from LLM feedback
            reconciliation_report = {
                "mismatched_transactions": [],
                "missing_counterparts": [],
                "pending_transactions": [],
                "action_items": []
            }
            
            # Parse LLM feedback to extract reconciliation details
            if "Mismatched Transactions" in llm_feedback:
                mismatched_section = llm_feedback.split("Mismatched Transactions")[1].split("###")[0]
                for line in mismatched_section.split("\n"):
                    if "IC" in line and "UNMATCHED" in line:
                        transaction = line.strip().split(" - ")[0]
                        reconciliation_report["mismatched_transactions"].append(transaction)
            
            if "Missing Counterpart Entries" in llm_feedback:
                missing_section = llm_feedback.split("Missing Counterpart Entries")[1].split("###")[0]
                for line in missing_section.split("\n"):
                    if "UNMATCHED" in line:
                        transaction = line.strip().split(" - ")[0]
                        reconciliation_report["missing_counterparts"].append(transaction)
            
            if "Action Items" in llm_feedback:
                action_section = llm_feedback.split("Action Items")[1]
                for line in action_section.split("\n"):
                    if line.strip().startswith(("1.", "2.", "3.")):
                        reconciliation_report["action_items"].append(line.strip())
            
            return ProcessResponse(
                status="success",
                message="Data verified and stored successfully",
                data=stored_df.to_dict(orient='records'),
                llm_feedback=llm_feedback,
                reconciliation_report=reconciliation_report
            )
        except Exception as e:
            return ProcessResponse(
                status="error",
                message="",
                error=f"Unexpected error during DB insertion: {str(e)}",
                llm_feedback=llm_feedback
            )
    except Exception as e:
        return ProcessResponse(
            status="error",
            message="",
            error=f"Error processing file: {str(e)}"
        )

@app.post("/process_consolidated_entries", response_model=ProcessResponse)
async def process_consolidated_entries(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")
    try:
        content = await file.read()
        consolidation_csv = content.decode('utf-8')
        initial_state = ConsoState(trial_balance=consolidation_csv, llm_feedback="", status="pending")
        result = consolidation_workflow.invoke(initial_state)
        llm_feedback = result['llm_feedback']

        # For consolidation, we consider it successful if we get consolidation analysis
        validation_success = any(phrase in llm_feedback.lower() for phrase in [
            "no errors",
            "all entries are valid",
            "validated successfully",
            "consolidation process",
            "trial balance",
            "intercompany elimination",
            "consolidation report"
        ])
        
        if validation_success:
            df = pd.read_csv(StringIO(consolidation_csv))
            df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
            
            # Analyze the consolidation data
            consolidation_report = {
                "entity_summary": {},
                "intercompany_transactions": [],
                "account_summary": {},
                "elimination_entries": [],
                "consolidation_issues": []
            }
            
            # Calculate entity-wise totals
            for entity in df['entity_code'].unique():
                entity_data = df[df['entity_code'] == entity]
                consolidation_report["entity_summary"][entity] = {
                    "total_debit": entity_data['debit_amount'].sum(),
                    "total_credit": entity_data['credit_amount'].sum(),
                    "net_position": entity_data['debit_amount'].sum() - entity_data['credit_amount'].sum()
                }
            
            # Identify intercompany transactions
            interco_transactions = df[df['interco_flag'] == 'Y']
            for _, row in interco_transactions.iterrows():
                consolidation_report["intercompany_transactions"].append({
                    "entity": row['entity_code'],
                    "account": row['account_name'],
                    "debit": row['debit_amount'],
                    "credit": row['credit_amount'],
                    "date": row['transaction_date']
                })
            
            # Analyze account-wise totals
            for account in df['account_name'].unique():
                account_data = df[df['account_name'] == account]
                consolidation_report["account_summary"][account] = {
                    "total_debit": account_data['debit_amount'].sum(),
                    "total_credit": account_data['credit_amount'].sum(),
                    "net_position": account_data['debit_amount'].sum() - account_data['credit_amount'].sum()
                }
            
            # Check for potential elimination entries
            for _, row in interco_transactions.iterrows():
                if row['debit_amount'] > 0:
                    matching_credit = interco_transactions[
                        (interco_transactions['account_name'] == row['account_name']) &
                        (interco_transactions['credit_amount'] == row['debit_amount'])
                    ]
                    if not matching_credit.empty:
                        consolidation_report["elimination_entries"].append({
                            "debit_entity": row['entity_code'],
                            "credit_entity": matching_credit.iloc[0]['entity_code'],
                            "account": row['account_name'],
                            "amount": row['debit_amount']
                        })
            
            # Check for consolidation issues
            for entity, summary in consolidation_report["entity_summary"].items():
                if abs(summary["net_position"]) > 0.01:  # Allow for small rounding differences
                    consolidation_report["consolidation_issues"].append({
                        "entity": entity,
                        "issue": "Unbalanced position",
                        "net_position": summary["net_position"]
                    })
            
            try:
                df.to_sql("consolidation", engine, if_exists="append", index=False)
                query = text("SELECT * FROM consolidation ORDER BY entity_code, transaction_date")
                with engine.connect() as conn:
                    result_db = conn.execute(query)
                stored_df = pd.DataFrame(result_db.fetchall(), columns=result_db.keys())
                
                return ProcessResponse(
                    status="success",
                    message="Data verified and stored successfully",
                    data=stored_df.to_dict(orient='records'),
                    llm_feedback=llm_feedback,
                    reconciliation_report=consolidation_report
                )
            except Exception as e:
                return ProcessResponse(
                    status="error",
                    message="",
                    error=f"Unexpected error during DB insertion: {str(e)}",
                    llm_feedback=llm_feedback
                )
        else:
            return ProcessResponse(
                status="error",
                message="",
                error="Validation issues found.",
                llm_feedback=llm_feedback
            )
    except Exception as e:
        return ProcessResponse(
            status="error",
            message="",
            error=f"Error processing file: {str(e)}"
        )

@app.post("/process_gst_entries", response_model=ProcessResponse)
async def process_gst_entries(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")
    try:
        content = await file.read()
        gst_csv = content.decode('utf-8')
        initial_state = GSTValidatorState(gst_returns=gst_csv, internal_records="", llm_feedback="", status="pending")
        result = gst_workflow.invoke(initial_state)
        llm_feedback = result['llm_feedback']
        
        # For GST validation, we consider it successful if we get GST analysis
        validation_success = any(phrase in llm_feedback.lower() for phrase in [
            "no errors",
            "all entries are valid",
            "validated successfully",
            "gst validation",
            "itc analysis",
            "tax calculation",
            "gst compliance"
        ])
        
        if validation_success:
            df = pd.read_csv(StringIO(gst_csv))
            df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
            
            # Analyze the GST data
            gst_report = {
                "vendor_summary": {},
                "itc_summary": {
                    "eligible": 0.0,
                    "ineligible": 0.0,
                    "total": 0.0
                },
                "tax_summary": {
                    "cgst": 0.0,
                    "sgst": 0.0,
                    "igst": 0.0,
                    "total_tax": 0.0
                },
                "validation_issues": [],
                "monthly_summary": {}
            }
            
            # Calculate vendor-wise summary
            for vendor in df['vendor_name'].unique():
                vendor_data = df[df['vendor_name'] == vendor]
                gst_report["vendor_summary"][vendor] = {
                    "total_invoices": len(vendor_data),
                    "total_amount": vendor_data['total_amount'].sum(),
                    "total_tax": (vendor_data['cgst_amount'].sum() + 
                                vendor_data['sgst_amount'].sum() + 
                                vendor_data['igst_amount'].sum()),
                    "eligible_itc": vendor_data[vendor_data['itc_eligible'] == 'Y']['total_amount'].sum(),
                    "ineligible_itc": vendor_data[vendor_data['itc_eligible'] == 'N']['total_amount'].sum()
                }
            
            # Calculate ITC summary
            gst_report["itc_summary"]["eligible"] = df[df['itc_eligible'] == 'Y']['total_amount'].sum()
            gst_report["itc_summary"]["ineligible"] = df[df['itc_eligible'] == 'N']['total_amount'].sum()
            gst_report["itc_summary"]["total"] = df['total_amount'].sum()
            
            # Calculate tax summary
            gst_report["tax_summary"]["cgst"] = df['cgst_amount'].sum()
            gst_report["tax_summary"]["sgst"] = df['sgst_amount'].sum()
            gst_report["tax_summary"]["igst"] = df['igst_amount'].sum()
            gst_report["tax_summary"]["total_tax"] = (df['cgst_amount'].sum() + 
                                                     df['sgst_amount'].sum() + 
                                                     df['igst_amount'].sum())
            
            # Generate monthly summary
            df['invoice_date'] = pd.to_datetime(df['invoice_date'])
            for month in df['invoice_date'].dt.to_period('M').unique():
                month_data = df[df['invoice_date'].dt.to_period('M') == month]
                gst_report["monthly_summary"][str(month)] = {
                    "total_invoices": len(month_data),
                    "total_amount": month_data['total_amount'].sum(),
                    "total_tax": (month_data['cgst_amount'].sum() + 
                                month_data['sgst_amount'].sum() + 
                                month_data['igst_amount'].sum()),
                    "eligible_itc": month_data[month_data['itc_eligible'] == 'Y']['total_amount'].sum(),
                    "ineligible_itc": month_data[month_data['itc_eligible'] == 'N']['total_amount'].sum()
                }
            
            # Check for validation issues
            for _, row in df.iterrows():
                # Check tax calculation
                calculated_total = (row['taxable_amount'] + 
                                 row['cgst_amount'] + 
                                 row['sgst_amount'] + 
                                 row['igst_amount'])
                if abs(calculated_total - row['total_amount']) > 0.01:  # Allow for small rounding differences
                    gst_report["validation_issues"].append({
                        "invoice": row['invoice_number'],
                        "issue": "Tax calculation mismatch",
                        "calculated": calculated_total,
                        "actual": row['total_amount']
                    })
                
                # Check GSTIN format
                if not re.match(r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$', row['gstin']):
                    gst_report["validation_issues"].append({
                        "invoice": row['invoice_number'],
                        "issue": "Invalid GSTIN format",
                        "gstin": row['gstin']
                    })
            
            try:
                df.to_sql("gst_validation", engine, if_exists="append", index=False)
                query = text("SELECT * FROM gst_validation ORDER BY invoice_date, invoice_number")
                with engine.connect() as conn:
                    result_db = conn.execute(query)
                stored_df = pd.DataFrame(result_db.fetchall(), columns=result_db.keys())
                
                return ProcessResponse(
                    status="success",
                    message="Data verified and stored successfully",
                    data=stored_df.to_dict(orient='records'),
                    llm_feedback=llm_feedback,
                    reconciliation_report=gst_report
                )
            except Exception as e:
                return ProcessResponse(
                    status="error",
                    message="",
                    error=f"Unexpected error during DB insertion: {str(e)}",
                    llm_feedback=llm_feedback
                )
        else:
            return ProcessResponse(
                status="error",
                message="",
                error="Validation issues found.",
                llm_feedback=llm_feedback
            )
    except Exception as e:
        return ProcessResponse(
            status="error",
            message="",
            error=f"Error processing file: {str(e)}"
        )

@app.post("/process_tds_entries", response_model=ProcessResponse)
async def process_tds_entries(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")
    try:
        content = await file.read()
        tds_csv = content.decode('utf-8')
        initial_state = TDSValidatorState(tds_entries=tds_csv, traces_data="", llm_feedback="", status="pending")
        result = tds_workflow.invoke(initial_state)
        llm_feedback = result['llm_feedback']
        
        # For TDS validation, we consider it successful if we get TDS analysis
        validation_success = any(phrase in llm_feedback.lower() for phrase in [
            "no errors",
            "all entries are valid",
            "validated successfully",
            "tds validation",
            "tds compliance",
            "tds verification",
            "tds analysis"
        ])
        
        if validation_success:
            df = pd.read_csv(StringIO(tds_csv))
            df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
            
            # Analyze the TDS data
            tds_report = {
                "vendor_summary": {},
                "section_summary": {},
                "status_summary": {
                    "paid": 0,
                    "pending": 0,
                    "rejected": 0
                },
                "monthly_summary": {},
                "validation_issues": [],
                "threshold_analysis": {
                    "below_threshold": [],
                    "above_threshold": []
                }
            }
            
            # Calculate vendor-wise summary
            for vendor in df['vendor_name'].unique():
                vendor_data = df[df['vendor_name'] == vendor]
                tds_report["vendor_summary"][vendor] = {
                    "total_challans": len(vendor_data),
                    "total_tds": vendor_data['tds_amount'].sum(),
                    "total_payment": vendor_data['payment_amount'].sum(),
                    "status_breakdown": {
                        "paid": len(vendor_data[vendor_data['status'] == 'PAID']),
                        "pending": len(vendor_data[vendor_data['status'] == 'PENDING']),
                        "rejected": len(vendor_data[vendor_data['status'] == 'REJECTED'])
                    }
                }
            
            # Calculate section-wise summary
            for section in df['section_code'].unique():
                section_data = df[df['section_code'] == section]
                tds_report["section_summary"][section] = {
                    "total_challans": len(section_data),
                    "total_tds": section_data['tds_amount'].sum(),
                    "total_payment": section_data['payment_amount'].sum(),
                    "status_breakdown": {
                        "paid": len(section_data[section_data['status'] == 'PAID']),
                        "pending": len(section_data[section_data['status'] == 'PENDING']),
                        "rejected": len(section_data[section_data['status'] == 'REJECTED'])
                    }
                }
            
            # Calculate status summary
            tds_report["status_summary"]["paid"] = len(df[df['status'] == 'PAID'])
            tds_report["status_summary"]["pending"] = len(df[df['status'] == 'PENDING'])
            tds_report["status_summary"]["rejected"] = len(df[df['status'] == 'REJECTED'])
            
            # Generate monthly summary
            df['transaction_date'] = pd.to_datetime(df['transaction_date'])
            for month in df['transaction_date'].dt.to_period('M').unique():
                month_data = df[df['transaction_date'].dt.to_period('M') == month]
                tds_report["monthly_summary"][str(month)] = {
                    "total_challans": len(month_data),
                    "total_tds": month_data['tds_amount'].sum(),
                    "total_payment": month_data['payment_amount'].sum(),
                    "status_breakdown": {
                        "paid": len(month_data[month_data['status'] == 'PAID']),
                        "pending": len(month_data[month_data['status'] == 'PENDING']),
                        "rejected": len(month_data[month_data['status'] == 'REJECTED'])
                    }
                }
            
            # Check for validation issues
            for _, row in df.iterrows():
                # Check TDS threshold based on section
                threshold = {
                    '194C': 30000,  # Contract payments
                    '194D': 15000,  # Insurance commission
                    '194H': 15000,  # Commission or brokerage
                    '194I': 240000, # Rent
                    '194J': 30000   # Professional/technical services
                }
                
                if row['section_code'] in threshold:
                    if row['payment_amount'] < threshold[row['section_code']]:
                        tds_report["threshold_analysis"]["below_threshold"].append({
                            "challan": row['challan_number'],
                            "section": row['section_code'],
                            "payment": row['payment_amount'],
                            "threshold": threshold[row['section_code']]
                        })
                    else:
                        tds_report["threshold_analysis"]["above_threshold"].append({
                            "challan": row['challan_number'],
                            "section": row['section_code'],
                            "payment": row['payment_amount'],
                            "threshold": threshold[row['section_code']]
                        })
                
                # Check PAN format
                if not re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$', row['pan']):
                    tds_report["validation_issues"].append({
                        "challan": row['challan_number'],
                        "issue": "Invalid PAN format",
                        "pan": row['pan']
                    })
                
                # Check TDS calculation
                expected_tds = 0
                if row['section_code'] == '194C':
                    expected_tds = row['payment_amount'] * 0.01  # 1%
                elif row['section_code'] == '194D':
                    expected_tds = row['payment_amount'] * 0.05  # 5%
                elif row['section_code'] == '194H':
                    expected_tds = row['payment_amount'] * 0.05  # 5%
                elif row['section_code'] == '194I':
                    expected_tds = row['payment_amount'] * 0.10  # 10%
                elif row['section_code'] == '194J':
                    expected_tds = row['payment_amount'] * 0.10  # 10%
                
                if abs(expected_tds - row['tds_amount']) > 0.01:  # Allow for small rounding differences
                    tds_report["validation_issues"].append({
                        "challan": row['challan_number'],
                        "issue": "TDS calculation mismatch",
                        "calculated": expected_tds,
                        "actual": row['tds_amount']
                    })
            
            try:
                df.to_sql("tds_validation", engine, if_exists="append", index=False)
                query = text("SELECT * FROM tds_validation ORDER BY transaction_date, challan_number")
                with engine.connect() as conn:
                    result_db = conn.execute(query)
                    stored_df = pd.DataFrame(result_db.fetchall(), columns=result_db.keys())
                
                return ProcessResponse(
                    status="success",
                    message="Data verified and stored successfully",
                    data=stored_df.to_dict(orient='records'),
                    llm_feedback=llm_feedback,
                    reconciliation_report=tds_report
                )
            except Exception as e:
                return ProcessResponse(
                    status="error",
                    message="",
                    error=f"Unexpected error during DB insertion: {str(e)}",
                    llm_feedback=llm_feedback
                )
        else:
            return ProcessResponse(
                status="error",
                message="",
                error="Validation issues found.",
                llm_feedback=llm_feedback
            )
    except Exception as e:
        return ProcessResponse(
            status="error",
            message="",
            error=f"Error processing file: {str(e)}"
        )

@app.post("/process_provision_entries", response_model=ProcessResponse)
async def process_provision_entries(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")
    try:
        content = await file.read()
        provision_csv = content.decode('utf-8')
        initial_state = ProvisionState(
            expense_ledger=provision_csv,
            previous_provisions="",
            vendor_info="",
            llm_feedback="",
            status="pending",
            confidence_score=0.0
        )
        result = provision_workflow.invoke(initial_state)
        llm_feedback = result['llm_feedback']
        
        # For provision planning, we consider it successful if we get provision analysis
        validation_success = any(phrase in llm_feedback.lower() for phrase in [
            "no errors",
            "all entries are valid",
            "validated successfully",
            "provision analysis",
            "expense patterns",
            "provision planning",
            "accrual entries"
        ])
        
        if validation_success:
            df = pd.read_csv(StringIO(provision_csv))
            df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
            
            # Analyze the provision data
            provision_report = {
                "expense_category_summary": {},
                "vendor_summary": {},
                "frequency_summary": {
                    "monthly": {"count": 0, "total_amount": 0.0},
                    "quarterly": {"count": 0, "total_amount": 0.0},
                    "annual": {"count": 0, "total_amount": 0.0}
                },
                "confidence_analysis": {
                    "high": {"count": 0, "total_amount": 0.0},  # > 0.9
                    "medium": {"count": 0, "total_amount": 0.0},  # 0.7-0.9
                    "low": {"count": 0, "total_amount": 0.0}  # < 0.7
                },
                "upcoming_provisions": [],
                "validation_issues": []
            }
            
            # Calculate expense category summary
            for category in df['expense_category'].unique():
                category_data = df[df['expense_category'] == category]
                provision_report["expense_category_summary"][category] = {
                    "total_provisions": len(category_data),
                    "total_amount": category_data['amount'].sum(),
                    "average_amount": category_data['amount'].mean(),
                    "frequency_breakdown": {
                        "monthly": len(category_data[category_data['frequency'] == 'Monthly']),
                        "quarterly": len(category_data[category_data['frequency'] == 'Quarterly']),
                        "annual": len(category_data[category_data['frequency'] == 'Annual'])
                    }
                }
            
            # Calculate vendor-wise summary
            for vendor in df['vendor_name'].unique():
                vendor_data = df[df['vendor_name'] == vendor]
                provision_report["vendor_summary"][vendor] = {
                    "total_provisions": len(vendor_data),
                    "total_amount": vendor_data['amount'].sum(),
                    "categories": list(vendor_data['expense_category'].unique()),
                    "frequency_breakdown": {
                        "monthly": len(vendor_data[vendor_data['frequency'] == 'Monthly']),
                        "quarterly": len(vendor_data[vendor_data['frequency'] == 'Quarterly']),
                        "annual": len(vendor_data[vendor_data['frequency'] == 'Annual'])
                    }
                }
            
            # Calculate frequency summary
            for freq in ['Monthly', 'Quarterly', 'Annual']:
                freq_data = df[df['frequency'] == freq]
                provision_report["frequency_summary"][freq.lower()] = {
                    "count": len(freq_data),
                    "total_amount": freq_data['amount'].sum()
                }
            
            # Analyze confidence scores
            for _, row in df.iterrows():
                if row['confidence_score'] > 0.9:
                    provision_report["confidence_analysis"]["high"]["count"] += 1
                    provision_report["confidence_analysis"]["high"]["total_amount"] += row['amount']
                elif row['confidence_score'] >= 0.7:
                    provision_report["confidence_analysis"]["medium"]["count"] += 1
                    provision_report["confidence_analysis"]["medium"]["total_amount"] += row['amount']
                else:
                    provision_report["confidence_analysis"]["low"]["count"] += 1
                    provision_report["confidence_analysis"]["low"]["total_amount"] += row['amount']
            
            # Identify upcoming provisions (within next 30 days)
            today = pd.Timestamp.now()
            df['expected_date'] = pd.to_datetime(df['expected_date'])
            upcoming = df[df['expected_date'] <= (today + pd.Timedelta(days=30))]
            for _, row in upcoming.iterrows():
                provision_report["upcoming_provisions"].append({
                    "expense_category": row['expense_category'],
                    "vendor": row['vendor_name'],
                    "amount": row['amount'],
                    "expected_date": row['expected_date'].strftime('%Y-%m-%d'),
                    "confidence_score": row['confidence_score']
                })
            
            # Check for validation issues
            for _, row in df.iterrows():
                # Check for missing dates
                if pd.isna(row['last_provision_date']) or pd.isna(row['expected_date']):
                    provision_report["validation_issues"].append({
                        "expense_category": row['expense_category'],
                        "vendor": row['vendor_name'],
                        "issue": "Missing date information"
                    })
                
                # Check for invalid frequencies
                if row['frequency'] not in ['Monthly', 'Quarterly', 'Annual']:
                    provision_report["validation_issues"].append({
                        "expense_category": row['expense_category'],
                        "vendor": row['vendor_name'],
                        "issue": f"Invalid frequency: {row['frequency']}"
                    })
                
                # Check for negative amounts
                if row['amount'] <= 0:
                    provision_report["validation_issues"].append({
                        "expense_category": row['expense_category'],
                        "vendor": row['vendor_name'],
                        "issue": f"Invalid amount: {row['amount']}"
                    })
                
                # Check for invalid confidence scores
                if not (0 <= row['confidence_score'] <= 1):
                    provision_report["validation_issues"].append({
                        "expense_category": row['expense_category'],
                        "vendor": row['vendor_name'],
                        "issue": f"Invalid confidence score: {row['confidence_score']}"
                    })
            
            try:
                df.to_sql("provision_planning", engine, if_exists="append", index=False)
                query = text("SELECT * FROM provision_planning ORDER BY expected_date, expense_category")
                with engine.connect() as conn:
                    result_db = conn.execute(query)
                stored_df = pd.DataFrame(result_db.fetchall(), columns=result_db.keys())
                
                return ProcessResponse(
                    status="success",
                    message="Data verified and stored successfully",
                    data=stored_df.to_dict(orient='records'),
                    llm_feedback=llm_feedback,
                    reconciliation_report=provision_report
                )
            except Exception as e:
                return ProcessResponse(
                    status="error",
                    message="",
                    error=f"Unexpected error during DB insertion: {str(e)}",
                    llm_feedback=llm_feedback
                )
        else:
            return ProcessResponse(
                status="error",
                message="",
                error="Validation issues found.",
                llm_feedback=llm_feedback
            )
    except Exception as e:
        return ProcessResponse(
            status="error",
            message="",
            error=f"Error processing file: {str(e)}"
        )

# GET Endpoints
@app.get("/api/provision-planning")
async def get_provision_planning():
    try:
        data = get_provision_planning_data()
        if not data:
            return ProcessResponse(
                status="success",
                message="No provision planning data found",
                data=[]
            )
        return ProcessResponse(
            status="success",
            message="Provision planning data retrieved successfully",
            data=data
        )
    except Exception as e:
        return ProcessResponse(
            status="error",
            message="",
            error=str(e)
        )

@app.get("/api/consolidation")
async def get_consolidation():
    try:
        data = get_consolidation_data()
        if not data:
            return ProcessResponse(
                status="success",
                message="No consolidation data found",
                data=[]
            )
        return ProcessResponse(
            status="success",
            message="Consolidation data retrieved successfully",
            data=data
        )
    except Exception as e:
        return ProcessResponse(
            status="error",
            message="",
            error=str(e)
        )

@app.get("/api/financial-report")
async def get_financial_report():
    try:
        data = get_financial_report_data()
        if not data:
            return ProcessResponse(
                status="success",
                message="No financial data found in any of the source tables",
                data=[],
                llm_feedback="No data available for analysis",
                validation_report={
                    "status": "pending",
                    "message": "No financial data available",
                    "required_steps": [
                        "Process provision planning data",
                        "Process consolidation data",
                        "Process journal entries",
                        "Process intercompany reconciliation",
                        "Process GST validation",
                        "Process TDS validation"
                    ]
                }
            )
        
        # Calculate total entries correctly
        total_entries = (
            data["summary"]["total_provisions"] +
            data["summary"]["total_consolidated_entries"] +
            data["summary"]["total_journal_entries"] +
            data["summary"]["total_interco_transactions"] +
            data["summary"]["total_gst_entries"] +
            data["summary"]["total_tds_entries"]
        )
        
        # Create validation report
        validation_report = {
            "status": "success",
            "data_sources": data["summary"]["data_sources"],
            "total_entries": total_entries,
            "missing_sources": [source for source, has_data in data["summary"]["data_sources"].items() if not has_data]
        }
        
        return ProcessResponse(
            status="success",
            message="Financial report data retrieved and analyzed successfully",
            data=data["financial_data"],
            llm_feedback=data["llm_analysis"],
            validation_report=validation_report
        )
    except Exception as e:
        return ProcessResponse(
            status="error",
            message="",
            error=f"Error retrieving financial report data: {str(e)}"
        )

@app.get("/api/audit-log")
async def get_audit_log():
    try:
        data = get_audit_log_data()
        if not data:
            return ProcessResponse(
                status="success",
                message="No audit data found",
                data=[],
                llm_feedback="No audit data available for analysis",
                validation_report={
                    "status": "pending",
                    "message": "No audit data available",
                    "required_steps": [
                        "Process journal entries",
                        "Process intercompany reconciliation",
                        "Process tax compliance data",
                        "Process provision planning",
                        "Process consolidation data"
                    ]
                }
            )
        
        # Create audit summary
        audit_summary = {
            "journal_entries": len(data["audit_data"]["journal_reconciliation"]),
            "interco_transactions": len(data["audit_data"]["interco_reconciliation"]),
            "gst_entries": len(data["audit_data"]["tax_compliance"]["gst"]),
            "tds_entries": len(data["audit_data"]["tax_compliance"]["tds"]),
            "provisions": len(data["audit_data"]["provisions"]),
            "consolidation_entries": len(data["audit_data"]["consolidation"])
        }
        
        return ProcessResponse(
            status="success",
            message="Audit data retrieved and analyzed successfully",
            llm_feedback=data["llm_analysis"],
            validation_report=audit_summary
        )
    except Exception as e:
        return ProcessResponse(
            status="error",
            message="",
            error=f"Error retrieving audit data: {str(e)}"
        )

@app.get("/api/journal-entries")
async def get_journal_entries():
    try:
        data = get_journal_entries_data()
        if not data:
            return ProcessResponse(
                status="success",
                message="No journal entries found",
                data=[]
            )
        return ProcessResponse(
            status="success",
            message="Journal entries retrieved successfully",
            data=data
        )
    except Exception as e:
        return ProcessResponse(
            status="error",
            message="",
            error=str(e)
        )

@app.get("/api/interco-reconciliation")
async def get_interco_reconciliation():
    try:
        data = get_interco_reconciliation_data()
        if not data:
            return ProcessResponse(
                status="success",
                message="No intercompany reconciliation data found",
                data=[]
            )
        return ProcessResponse(
            status="success",
            message="Intercompany reconciliation data retrieved successfully",
            data=data
        )
    except Exception as e:
        return ProcessResponse(
            status="error",
            message="",
            error=str(e)
        )

@app.get("/api/gst-validation")
async def get_gst_validation():
    try:
        data = get_gst_validation_data()
        if not data:
            return ProcessResponse(
                status="success",
                message="No GST validation data found",
                data=[]
            )
        return ProcessResponse(
            status="success",
            message="GST validation data retrieved successfully",
            data=data
        )
    except Exception as e:
        return ProcessResponse(
            status="error",
            message="",
            error=str(e)
        )

@app.get("/api/tds-validation")
async def get_tds_validation():
    try:
        data = get_tds_validation_data()
        if not data:
            return ProcessResponse(
                status="success",
                message="No TDS validation data found",
                data=[]
            )
        return ProcessResponse(
            status="success",
            message="TDS validation data retrieved successfully",
            data=data
        )
    except Exception as e:
        return ProcessResponse(
            status="error",
            message="",
            error=str(e)
        )

@app.get("/api/traces-data")
async def get_traces():
    try:
        data = get_traces_data()
        if not data:
            return ProcessResponse(
                status="success",
                message="No TRACES data found",
                data=[]
            )
        return ProcessResponse(
            status="success",
            message="TRACES data retrieved successfully",
            data=data
        )
    except Exception as e:
        return ProcessResponse(
            status="error",
            message="",
            error=str(e)
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
