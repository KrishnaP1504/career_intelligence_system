from pydantic import BaseModel, EmailStr, field_validator
from typing import List, Optional
from datetime import datetime, timezone


# Bug 9 Fix: EmailStr validates email format; field_validator enforces password length
class UserAuth(BaseModel):
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def password_min_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long.")
        return v


class PasswordReset(BaseModel):
    email: EmailStr
    new_password: str

    @field_validator("new_password")
    @classmethod
    def password_min_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("New password must be at least 8 characters long.")
        return v


class SaveRecordRequest(BaseModel):
    user_email: str
    skills: List[str]
    target_domain: str
    experience: float
    education: str
    probability: float
    label: str
    skill_gap: float
    
    
    project_count: int
    certification_count: int
    domain_fit_score: float

    # Bug 12 Fix: Auto-stamp every saved record with UTC time
    timestamp: Optional[datetime] = None

    @field_validator("timestamp", mode="before")
    @classmethod
    def set_timestamp(cls, v):
        return v or datetime.now(timezone.utc)