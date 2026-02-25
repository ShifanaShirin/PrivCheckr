from pydantic import BaseModel, EmailStr, field_validator
import re


class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str

    @field_validator("password")
    def validate_password(cls, value):

        if len(value) < 8:
            raise ValueError("Password must be at least 8 characters long")

        if len(value.encode("utf-8")) > 72:
            raise ValueError("Password must not exceed 72 characters")

        if not re.search(r"[A-Z]", value):
            raise ValueError("Password must contain at least one uppercase letter")

        if not re.search(r"[a-z]", value):
            raise ValueError("Password must contain at least one lowercase letter")

        if not re.search(r"\d", value):
            raise ValueError("Password must contain at least one number")

        return value


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class AnalysisResponse(BaseModel):
    id: int
    detected_language: str
    risk_level: str
    dpdp_score: str

    class Config:
        orm_mode = True