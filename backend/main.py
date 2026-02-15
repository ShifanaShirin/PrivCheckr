from fastapi import FastAPI
from pydantic import BaseModel
from langdetect import detect
from fastapi.middleware.cors import CORSMiddleware
from deep_translator import GoogleTranslator
import pycountry

# ✅ Correct package imports
from backend.database import engine
from backend.models import Base

from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException
from backend.database import get_db
from backend.models import User
from backend.schemas import UserCreate, UserLogin
from backend.auth import hash_password, verify_password, create_access_token

from backend.nlp_pipeline import preprocess_text


# ---------- APP ----------
app = FastAPI(
    title="Privacy Policy Analysis System",
    description="Multilingual DPDP Compliance Evaluation API",
    version="1.0"
)

# ---------- CREATE DATABASE TABLES ----------
Base.metadata.create_all(bind=engine)

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# ---------- INPUT MODEL ----------
class PolicyInput(BaseModel):
    policy_text: str


# ---------- LANGUAGE HELPERS ----------
def normalize_language_code(lang_code: str) -> str:
    if lang_code.startswith("zh"):
        return "zh"
    return lang_code


def get_language_name(lang_code: str) -> str:
    if lang_code == "zh":
        return "Chinese"
    try:
        lang = pycountry.languages.get(alpha_2=lang_code)
        return lang.name if lang else lang_code
    except:
        return lang_code


# ---------- ROOT ----------
@app.get("/")
def root():
    return {"message": "Backend is running"}

# ---------- REGISTER ----------
@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):

    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_pwd = hash_password(user.password)

    new_user = User(
        name=user.name,
        email=user.email,
        password=hashed_pwd
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": "User registered successfully"}


# ---------- LOGIN ----------
@app.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):

    db_user = db.query(User).filter(User.email == user.email).first()

    if not db_user or not verify_password(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token(data={"sub": db_user.email})

    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

# ---------- ANALYZE ----------
@app.post("/analyze")
def analyze_policy(data: PolicyInput):

    # ---------- EMPTY INPUT ----------
    if not data.policy_text.strip():
        return {"error": "Empty privacy policy text"}

    original_text = data.policy_text.strip().lower()

    # ---------- LANGUAGE DETECTION ----------
    try:
        detected_code = detect(original_text)
        detected_code = normalize_language_code(detected_code)
    except:
        detected_code = "unknown"

    detected_language = get_language_name(detected_code)

    # ---------- TRANSLATION ----------
    analysis_text = original_text
    translated = False

    if detected_code != "en":
        try:
            analysis_text = GoogleTranslator(
                source="auto",
                target="en"
            ).translate(original_text).lower()

            if analysis_text != original_text:
                translated = True
        except:
            analysis_text = original_text
    
    # ---------- NLP PREPROCESSING ----------
    processed_text = preprocess_text(analysis_text)

    # ---------- DATA TYPE DETECTION ----------
    data_types = []

    if any(w in analysis_text for w in [
        "name", "email", "phone", "address", "location"
    ]):
        data_types.append("Personal Data")

    if any(w in analysis_text for w in [
        "aadhaar", "biometric", "health", "bank", "financial"
    ]):
        data_types.append("Sensitive Data")

    # ---------- NEGATIVE INDICATORS ----------
    STRONG_NEGATIVE = [
        "without consent",
        "without notice",
        "indefinitely",
        "no guarantee",
        "we reserve the right"
    ]

    WEAK_NEGATIVE = [
        "may collect",
        "may share",
        "may use",
        "at any time",
        "subject to change"
    ]

    strong_negative_found = any(p in analysis_text for p in STRONG_NEGATIVE)
    weak_negative_found = any(p in analysis_text for p in WEAK_NEGATIVE)

    # ---------- DPDP PRINCIPLE CHECKS ----------
    dpdp_checks = {
        "consent": (
            any(w in analysis_text for w in [
                "consent", "agree", "agreement", "permission"
            ]) and not strong_negative_found
        ),

        "purpose": (
            any(w in analysis_text for w in [
                "purpose", "use", "used for", "intended"
            ]) and not strong_negative_found
        ),

        "retention": (
            any(w in analysis_text for w in [
                "retain", "retained", "stored", "storage",
                "limited period", "necessary period"
            ]) and "indefinitely" not in analysis_text
        ),

        "user_rights": any(w in analysis_text for w in [
            "right to access", "right to update",
            "right to delete", "can delete", "can remove"
        ]),

        "grievance": any(w in analysis_text for w in [
            "contact us", "grievance",
            "complaint", "reach us", "support"
        ])
    }

    # ---------- SCORE ----------
    score = sum(dpdp_checks.values())

    # ---------- RISK LEVEL ----------
    if score <= 1 or strong_negative_found:
        risk = "High"
    elif score <= 3 or weak_negative_found:
        risk = "Medium"
    else:
        risk = "Low"

    # ---------- RECOMMENDATIONS ----------
    recommendations = []

    if not dpdp_checks["consent"]:
        recommendations.append("Add clear user consent before data collection.")

    if not dpdp_checks["purpose"]:
        recommendations.append("Clearly specify the purpose of data usage.")

    if not dpdp_checks["retention"]:
        recommendations.append("Define a clear data retention period.")

    if not dpdp_checks["user_rights"]:
        recommendations.append("Mention user rights such as access, update, or deletion.")

    if not dpdp_checks["grievance"]:
        recommendations.append("Provide grievance redressal or contact details.")

    if not recommendations:
        recommendations.append("Privacy policy shows strong DPDP compliance.")

    # ---------- EXPLANATION ----------
    explanation = [
        "User consent is mentioned." if dpdp_checks["consent"]
        else "User consent is not clearly specified.",

        "Purpose limitation is mentioned." if dpdp_checks["purpose"]
        else "Purpose of data usage is unclear.",

        "Data retention or storage is addressed." if dpdp_checks["retention"]
        else "Data retention details are missing.",

        "User rights are mentioned." if dpdp_checks["user_rights"]
        else "User rights are not clearly stated."
    ]

    # ---------- FINAL RESPONSE ----------
    return {
        "detected_language": detected_language,
        "translated_to_english": translated,
        "data_types": data_types,
        "dpdp_score": f"{score}/5",
        "risk_level": risk,
        "dpdp_breakdown": dpdp_checks,
        "explanation": explanation,
        "recommendations": recommendations
    }
