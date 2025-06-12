import asyncio
import logging
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn
from mcp_client import main as mcp_main
import shutil
from mcp_server import journal_validator_agent, interco_reconciliation_agent
import pandas as pd
import os
from mcp_server import csv_data_validation_ingestion_agent
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Financial Closure System API",
    description="API endpoints for automated financial closure system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class AgentRequest(BaseModel):
    question: str
    entry_id: Optional[str] = None
    user: Optional[str] = None
    action: Optional[str] = None

class AgentResponse(BaseModel):
    result: Dict[str, Any]
    status: str
    message: str

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/api/journal-validator")
async def journal_validator_endpoint(
    file: UploadFile = File(...),
    question: str = Form("Validate journal entries for nulls, balance, and duplicates."),
    entry_id: str = Form(None),
    user: str = Form(None),
    action: str = Form(None)
):
    # Save the uploaded file as JOURNALS.csv
    with open("JOURNALS.csv", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Call your MCP main function as before
    response = await mcp_main(
        question=question,
        entry_id=entry_id,
        user=user,
        action=action
    )

    return {
        "result": response,
        "status": "completed",
        "message": "Journal validation completed successfully"
    }

@app.post("/api/interco-reconciliation", response_model=AgentResponse)
async def interco_reconciliation_endpoint(request: AgentRequest):
    try:
        response = await mcp_main(
            question=request.question,
            entry_id=request.entry_id,
            user=request.user,
            action=request.action
        )

        return AgentResponse(
            result=response,
            status="completed",
            message="Intercompany reconciliation completed successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/gst-validator", response_model=AgentResponse)
async def gst_validator_endpoint(request: AgentRequest):
    try:
        response = await mcp_main(
            question=request.question,
            entry_id=request.entry_id,
            user=request.user,
            action=request.action
        )

        return AgentResponse(
            result=response,
            status="completed",
            message="GST validation completed successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tds-validator", response_model=AgentResponse)
async def tds_validator_endpoint(request: AgentRequest):
    try:
        response = await mcp_main(
            question=request.question,
            entry_id=request.entry_id,
            user=request.user,
            action=request.action
        )
        return AgentResponse(
            result=response,
            status="completed",
            message="TDS validation completed successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/provision", response_model=AgentResponse)
async def provision_endpoint(request: AgentRequest):
    try:
        response = await mcp_main(
            question=request.question,
            entry_id=request.entry_id,
            user=request.user,
            action=request.action
        )

        return AgentResponse(
            result=response,
            status="completed",
            message="Provision analysis completed successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/consolidation", response_model=AgentResponse)
async def consolidation_endpoint(request: AgentRequest):
    try:
        response = await mcp_main(
            question=request.question,
            entry_id=request.entry_id,
            user=request.user,
            action=request.action
        )

        return AgentResponse(
            result=response,
            status="completed",
            message="Consolidation completed successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/report", response_model=AgentResponse)
async def report_endpoint(request: AgentRequest):
    try:
        response = await mcp_main(
            question=request.question,
            entry_id=request.entry_id,
            user=request.user,
            action=request.action
        )

        return AgentResponse(
            result=response,
            status="completed",
            message="Report generation completed successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/audit", response_model=AgentResponse)
async def audit_endpoint(request: AgentRequest):
    try:
        response = await mcp_main(
            question=request.question,
            entry_id=request.entry_id,
            user=request.user,
            action=request.action
        )

        return AgentResponse(
            result=response,
            status="completed",
            message="Audit analysis completed successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/csv-validate-ingest")
async def csv_validate_ingest(file: UploadFile = File(...)):
    """
    Endpoint to validate, store, and process CSV files.
    Handles:
    1. File validation
    2. Database storage
    3. Agent routing
    4. Returns structured response with results
    """
    try:
        # Save the uploaded file to DATA/Transactional/
        data_dir = os.path.join(os.path.dirname(__file__), '../DATA/Transactional')
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, file.filename)
        
        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Call the validation and ingestion agent
        response = csv_data_validation_ingestion_agent({"filename": file.filename})
        
        # If validation failed, return the error response
        if response.get("status") == "fail":
            return JSONResponse(
                status_code=400,
                content=response
            )
        
        # If successful, return the complete response
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "File processed successfully",
                "validation_result": response.get("database_result", {}),
                "agent_result": response.get("agent_result", {}),
                "sample_data": response.get("database_result", {}).get("sample_data", [])
            }
        )

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error processing file: {str(e)}"
            }
        )

# Add new endpoints for specific agent results
@app.get("/api/agent-results/{filename}")
async def get_agent_results(filename: str):
    """
    Get the results from the specific agent that processed the file
    """
    try:
        # Call the validation and ingestion agent to get results
        response = csv_data_validation_ingestion_agent({"filename": filename})
        
        if response.get("status") == "fail":
            return JSONResponse(
                status_code=400,
                content=response
            )
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "agent_result": response.get("agent_result", {}),
                "database_result": response.get("database_result", {})
            }
        )
    except Exception as e:
        logger.error(f"Error getting agent results: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error getting agent results: {str(e)}"
            }
        )

# Add endpoint to get database table contents
@app.get("/api/table-data/{table_name}")
async def get_table_data(table_name: str):
    """
    Get the contents of a specific database table
    """
    try:
        conn = get_db_connection()
        if not conn:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "Failed to connect to database"
                }
            )

        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get table contents
        cursor.execute(f"SELECT * FROM {table_name}")
        records = cursor.fetchall()
        
        cursor.close()
        conn.close()

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "table_name": table_name,
                "record_count": len(records),
                "data": records
            }
        )
    except Exception as e:
        logger.error(f"Error getting table data: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error getting table data: {str(e)}"
            }
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running"}

# 1. (Optional) Dummy login endpoint
@app.post("/login")
def login(request: LoginRequest):
    if request.username == "admin" and request.password == "password":
        return {"token": "dummy-token"}
    return JSONResponse(status_code=401, content={"error": "Invalid credentials"})

# 2. Upload journal entries
@app.post("/upload-journal/")
async def upload_journal(file: UploadFile = File(...)):
    with open("JOURNALS.csv", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # Read the saved CSV and return its content
    df = pd.read_csv("JOURNALS.csv")
    data = df.to_dict(orient="records")
    #print(data)
    return {"status": "uploaded", "data": data}

# 3. Trigger JournalValidator Agent
@app.post("/validate-journal/")
def validate_journal():
    input_data = {
        "question": "Validate journal entries for nulls, balance, and duplicates.",
        "entry_id": None,
        "user": None,
        "action": None
    }
    response = journal_validator_agent(input_data)
    return {
        "result": response,
        "status": "completed",
        "message": "Journal validation completed successfully"
    }

# 4. Trigger IntercoReconciliation Agent
@app.post("/run-reconciliation/")
def run_reconciliation():
    return interco_reconciliation_agent({})

# 5. (Optional) Checklist status endpoint
@app.get("/checklist-status/")
def checklist_status():
    # You can expand this to track real status
    return {
        "journal_uploaded": True,
        "journal_validated": True,
        "reconciliation_done": True,
        "provisions_reviewed": False,
        "closure_complete": False
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001) 