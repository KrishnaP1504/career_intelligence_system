from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from core.database import connect_to_mongo, close_mongo_connection
from api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Connect to MongoDB
    await connect_to_mongo()
    yield
    # Shutdown: Close MongoDB connection
    await close_mongo_connection()


app = FastAPI(
    title="Career Intelligence API",
    description="Backend ML Engine for IT Resume Scoring",
    version="1.0.0",
    lifespan=lifespan,
)

# Bug 7 Fix: Restrict CORS to known origins instead of wildcard + credentials
# (Wildcard "*" with allow_credentials=True is rejected by all browsers)
ALLOWED_ORIGINS = [
    "http://localhost:8501",    # Streamlit default port
    "http://127.0.0.1:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)