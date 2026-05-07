from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()  # reads your .env file

DATABASE_URL = os.getenv("DATABASE_URL")

# Engine is the actual connection to PostgreSQL
# pool_pre_ping checks connection is alive before using it
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Every database operation happens inside a Session
# autocommit=False means changes only save when you explicitly commit
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class that all your table models will inherit from
Base = declarative_base()

def get_db():
    """
    Dependency function — FastAPI calls this automatically for every request.
    Gives the route a database session, then closes it cleanly when done.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()