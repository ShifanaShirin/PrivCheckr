from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from langdetect import detect
from deep_translator import GoogleTranslator
import pycountry
import os

# Database & Models
from backend.database import engine, get_db
from backend.models import Base, User, Analysis
from backend.schemas import UserCreate, UserLogin, AnalysisResponse
from backend.auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user
)

# NLP + ML
from backend.nlp_pipeline import preprocess_text
from backend.ml_classifier import classify_sentences
from backend.dpdp_evaluator import evaluate_dpdp


# ============================================================
# APP CONFIG
# ============================================================

app = FastAPI(
    title="Privacy Policy Analysis System",
    description="Multilingual DPDP Compliance Evaluation API",
    version="4.0"
)

# Create tables safely at startup
Base.metadata.create_all(bind=engine)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)


# ============================================================
# INPUT MODEL
# ============================================================

class PolicyInput(BaseModel):
    policy_text: str


# ============================================================
# LANGUAGE HELPERS
# ============================================================

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


# ============================================================
# AUTH ROUTES
# ============================================================

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


@app.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):

    db_user = db.query(User).filter(User.email == user.email).first()

    if not db_user or not verify_password(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token(
    data={
        "sub": str(db_user.id),
        "email": db_user.email,
        "name": db_user.name
    }
)

    return {
        "access_token": access_token,
        "token_type": "bearer"
    }


# ============================================================
# ANALYZE (PROTECTED)
# ============================================================

@app.post("/analyze")
def analyze_policy(
    data: PolicyInput,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):

    if not data.policy_text.strip():
        raise HTTPException(status_code=400, detail="Empty privacy policy text")

    original_text = data.policy_text.strip().lower()

    # ---------------- Language Detection ----------------
    try:
        detected_code = detect(original_text)
        detected_code = normalize_language_code(detected_code)
    except:
        detected_code = "unknown"

    detected_language = get_language_name(detected_code)

    # ---------------- Translation ----------------
    analysis_text = original_text
    translated = False

    if detected_code not in ["en", "unknown"]:
        try:
            translated_text = GoogleTranslator(
                source="auto",
                target="en"
            ).translate(original_text)

            if translated_text:
                analysis_text = translated_text.lower()
                translated = True
        except:
            analysis_text = original_text

    # ---------------- NLP + ML ----------------
    processed_text = preprocess_text(analysis_text)
    classification_results = classify_sentences(processed_text)

    dpdp_checks, score, risk = evaluate_dpdp(classification_results)
    consent = dpdp_checks["consent"]
    purpose = dpdp_checks["purpose"]
    retention = dpdp_checks["retention"]
    rights = dpdp_checks["user_rights"]
    grievance = dpdp_checks["grievance"]

    # ---------------- Data Type Detection ----------------
    data_types = []

    if any(w in analysis_text for w in
           ["name", "email", "phone", "address", "location"]):
        data_types.append("Personal Data")

    if any(w in analysis_text for w in
           ["aadhaar", "biometric", "health", "bank", "financial"]):
        data_types.append("Sensitive Data")

    # ---------------- Explanation ----------------
    explanation = []

    if consent:
        explanation.append("User consent clause detected.")
    else:
        explanation.append("Missing clear user consent clause.")

    if purpose:
        explanation.append("Purpose of data collection identified.")
    else:
        explanation.append("Purpose limitation not clearly defined.")

    if retention:
        explanation.append("Data retention policy mentioned.")
    else:
        explanation.append("No clear data retention period specified.")

    if rights:
        explanation.append("User rights such as access or deletion detected.")
    else:
        explanation.append("User rights not clearly stated.")

    if grievance:
        explanation.append("Grievance or contact mechanism available.")
    else:
        explanation.append("No grievance redressal mechanism mentioned.")

    # ---------------- Recommendations ----------------
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

    # ---------------- Store in DB ----------------
    new_analysis = Analysis(
        policy_text=analysis_text,
        detected_language=detected_language,
        risk_level=risk,
        dpdp_score=f"{score}/7",
        owner_id=current_user.id
    )

    db.add(new_analysis)
    db.commit()

    # ---------------- Response ----------------
    return {
        "detected_language": detected_language,
        "translated_to_english": translated,
        "data_types": data_types,
        "dpdp_score": f"{score}/7",
        "risk_level": risk,
        "dpdp_breakdown": dpdp_checks,
        "classification_summary": classification_results,
        "recommendations": recommendations,
        "explanation": explanation
    }


# ============================================================
# USER HISTORY
# ============================================================

@app.get("/my-analyses", response_model=list[AnalysisResponse])
def get_user_analyses(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return db.query(Analysis).filter(
        Analysis.owner_id == current_user.id
    ).all()


# ============================================================
# FRONTEND ROUTES
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
frontend_path = os.path.join(BASE_DIR, "frontend")


@app.get("/", response_class=FileResponse)
def serve_home():
    return os.path.join(frontend_path, "index.html")


@app.get("/login.html", response_class=FileResponse)
def serve_login():
    return os.path.join(frontend_path, "login.html")


@app.get("/register.html", response_class=FileResponse)
def serve_register():
    return os.path.join(frontend_path, "register.html")


@app.get("/dashboard.html", response_class=FileResponse)
def serve_dashboard():
    return os.path.join(frontend_path, "dashboard.html")


@app.get("/analyze.html", response_class=FileResponse)
def serve_analyze():
    return os.path.join(frontend_path, "analyze.html")


@app.get("/about.html", response_class=FileResponse)
def serve_about():
    return os.path.join(frontend_path, "about.html")


@app.get("/faq.html", response_class=FileResponse)
def serve_faq():
    return os.path.join(frontend_path, "faq.html")