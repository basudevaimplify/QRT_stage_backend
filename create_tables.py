from sqlalchemy import create_engine, text
import os

# Database connection URL
DATABASE_URL = "postgresql+psycopg2://postgres:1@localhost:5432/financial_db"

def create_tables():
    try:
        # Create database engine
        engine = create_engine(DATABASE_URL)
        
        # Read schema file
        with open('database_schema.sql', 'r') as f:
            schema = f.read()
        
        # Execute schema to create tables
        with engine.connect() as conn:
            conn.execute(text(schema))
            conn.commit()
            print("Successfully created all tables!")
            
    except Exception as e:
        print(f"Error creating tables: {str(e)}")

if __name__ == "__main__":
    create_tables() 