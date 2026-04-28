import os
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi import HTTPException

# Defaults to your provided local MongoDB URI
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/career_intelligence_system")
DB_NAME = "career_intelligence_system"

class Database:
    client = None
    db = None

db_instance = Database()

async def connect_to_mongo():
    try:
        logging.info(f"Connecting to MongoDB at {MONGO_URI}...")
        db_instance.client = AsyncIOMotorClient(MONGO_URI)
        db_instance.db = db_instance.client[DB_NAME]

        # Bug 3 Fix: Verify the connection is actually alive with a ping
        await db_instance.client.admin.command("ping")
        logging.info("Successfully connected to MongoDB.")

        # Bug 11 Fix: Create indexes for fast queries on login/history
        await db_instance.db["users"].create_index("email", unique=True)
        await db_instance.db["resume_analyses"].create_index("user_email")
        logging.info("Database indexes ensured.")

    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        db_instance.db = None  # Ensure it stays None so get_db() can guard it

async def close_mongo_connection():
    if db_instance.client:
        logging.info("Closing MongoDB connection.")
        db_instance.client.close()

def get_db():
    # Bug 3 Fix: Return a 503 instead of silently returning None
    if db_instance.db is None:
        raise HTTPException(
            status_code=503,
            detail="Database connection is not available. Please try again later."
        )
    return db_instance.db