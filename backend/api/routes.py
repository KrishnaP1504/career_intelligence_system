import os
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.hash import pbkdf2_sha256
import jwt

from core.database import get_db
from models.schemas import SaveRecordRequest, UserAuth, PasswordReset

router = APIRouter()

# ─── JWT CONFIG ────────────────────────────────────────────────────────────────
# Bug 4 Fix: Implement JWT-based authentication
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-this-in-production-use-a-long-random-string")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

security = HTTPBearer()


def create_access_token(email: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": email, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Dependency: validates Bearer token and returns the authenticated user's email."""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token payload.")
        return email
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired. Please log in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid authentication token.")


def verify_password(plain_password, hashed_password):
    return pbkdf2_sha256.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pbkdf2_sha256.hash(password)


# ─── HEALTH CHECK ──────────────────────────────────────────────────────────────
# Bug 2 Fix: Add /health endpoint so the frontend ping_server() works
@router.get("/health")
async def health_check():
    return {"status": "ok"}


# ─── AUTHENTICATION ENDPOINTS ──────────────────────────────────────────────────

@router.post("/signup", status_code=status.HTTP_201_CREATED)
async def create_user(user: UserAuth, db=Depends(get_db)):
    existing_user = await db["users"].find_one({"email": user.email.lower()})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = get_password_hash(user.password)
    new_user = {
        "email": user.email.lower(),
        "password": hashed_password,
        "created_at": datetime.now(timezone.utc),
    }
    await db["users"].insert_one(new_user)
    return {"message": "User created successfully"}


@router.post("/login")
async def login_user(user: UserAuth, db=Depends(get_db)):
    db_user = await db["users"].find_one({"email": user.email.lower()})
    if not db_user or not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Bug 4 Fix: Issue a JWT token so subsequent requests are authenticated
    token = create_access_token(db_user["email"])
    return {
        "message": "Login successful",
        "email": db_user["email"],
        "access_token": token,
        "token_type": "bearer",
    }


@router.post("/forgot-password")
async def forgot_password(reset: PasswordReset, db=Depends(get_db)):
    # NOTE (Bug 6): Production apps must send an email OTP instead of allowing
    # direct resets. This endpoint is intentionally simplified for the prototype.
    db_user = await db["users"].find_one({"email": reset.email.lower()})
    if not db_user:
        raise HTTPException(status_code=404, detail="Email not found")

    hashed_password = get_password_hash(reset.new_password)
    await db["users"].update_one(
        {"email": reset.email.lower()},
        {"$set": {"password": hashed_password}}
    )
    return {"message": "Password reset successfully"}


# ─── HISTORY ENDPOINT ──────────────────────────────────────────────────────────

@router.get("/user-history/{email}")
async def get_user_history(
    email: str,
    db=Depends(get_db),
    current_user: str = Depends(verify_token),   # Bug 4 Fix: require auth
):
    # Bug 5 Fix: Prevent IDOR — only allow users to read their own history
    if current_user.lower() != email.lower():
        raise HTTPException(
            status_code=403,
            detail="Access denied: you can only view your own history."
        )

    cursor = db["resume_analyses"].find({"user_email": email.lower()}).sort("_id", -1)
    history = await cursor.to_list(length=50)

    for item in history:
        item["_id"] = str(item["_id"])
    return history


# Bug 10 Fix: Return 201 Created for resource creation
@router.post("/save-record", status_code=status.HTTP_201_CREATED)
async def save_record(
    request: SaveRecordRequest,
    db=Depends(get_db),
    current_user: str = Depends(verify_token),   # Bug 4 Fix: require auth
):
    # Prevent saving records on behalf of another user
    if current_user.lower() != request.user_email.lower():
        raise HTTPException(
            status_code=403,
            detail="Access denied: cannot save a record for another user."
        )

    document = request.model_dump()

    try:
        new_record = await db["resume_analyses"].insert_one(document)
        return {"status": "success", "id": str(new_record.inserted_id)}
    except Exception as e:
        print(f"Database Error: {e}")
        raise HTTPException(status_code=500, detail="Could not save to database")