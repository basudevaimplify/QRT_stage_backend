# Financial Server Backend

A robust FastAPI-based backend server for processing and analyzing financial data, featuring AI-powered validation and analysis.

## Features

- Journal Entry Processing & Validation
- Intercompany Reconciliation
- Financial Consolidation
- GST Validation & Compliance
- TDS Validation & Compliance
- Provision Planning
- AI-powered Analysis using OpenAI
- PostgreSQL Database Integration
- Comprehensive Audit Logging

## Tech Stack

- Python 3.8+
- FastAPI
- SQLAlchemy
- PostgreSQL
- OpenAI API
- Pandas
- LangGraph
- Uvicorn

## Prerequisites

- Python 3.8 or higher
- PostgreSQL database
- OpenAI API key
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/basudevaimplify/QRT_stage.git
cd QRT_stage/newbackend
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
DATABASE_URL=postgresql+psycopg2://postgres:1@localhost:5432/financial_db
```

5. Set up the database:
```bash
python create_tables.py
```

## Running the Server

1. Start the server:
```bash
uvicorn financial_server:app --reload --host 0.0.0.0 --port 8000
```

2. Run tests:
```bash
python test_server.py
```

Or use the provided script:
```bash
./run_and_test.sh
```

## API Endpoints

### POST Endpoints
- `/process_journal_entries` - Process and validate journal entries
- `/process_interco_entries` - Process intercompany transactions
- `/process_consolidated_entries` - Process consolidated financial data
- `/process_gst_entries` - Process and validate GST entries
- `/process_tds_entries` - Process and validate TDS entries
- `/process_provision_entries` - Process provision planning data

### GET Endpoints
- `/api/provision-planning` - Get provision planning data
- `/api/consolidation` - Get consolidation data
- `/api/financial-report` - Get comprehensive financial report
- `/api/audit-log` - Get audit trail data
- `/api/journal-entries` - Get journal entries
- `/api/interco-reconciliation` - Get intercompany reconciliation data
- `/api/gst-validation` - Get GST validation data
- `/api/tds-validation` - Get TDS validation data
- `/api/traces-data` - Get TRACES data

## Database Schema

The application uses PostgreSQL with the following main tables:
- journal_entries
- interco_reconciliation
- consolidation
- gst_validation
- tds_validation
- provision_planning
- audit_log

See `database_schema.sql` for complete schema details.

## Testing

The application includes:
- Unit tests in `test_server.py`
- Sample data generation scripts:
  - `generate_csv_files.py` - Generate test CSV files
  - `generate_large_csv_files.py` - Generate large test datasets

## Project Structure

```
newbackend/
├── financial_server.py      # Main FastAPI application
├── api.py                   # API endpoints and models
├── create_tables.py         # Database initialization
├── database_schema.sql      # Database schema
├── test_server.py          # Test suite
├── requirements.txt        # Python dependencies
├── csv_files/             # Sample CSV files
└── ...other files
```

## Development

1. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes

3. Run tests:
```bash
python test_server.py
```

4. Commit and push:
```bash
git add .
git commit -m "Your commit message"
git push origin feature/your-feature-name
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License.

## Contact

Basudev - basudev@aimplify.tech

Project Link: [https://github.com/basudevaimplify/QRT_stage](https://github.com/basudevaimplify/QRT_stage) 