from fastapi import FastAPI
from pydantic import BaseModel
from langdetect import detect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Privacy Policy Analysis System",
    description="Multilingual DPDP Compliance Evaluation API",
    version="1.0"
)

# ✅ CORS CONFIGURATION (CRITICAL)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # allow frontend access
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# ---------- Input Model ----------
class PolicyInput(BaseModel):
    policy_text: str


# ---------- Root Endpoint ----------
@app.get("/")
def root():
    return {"message": "Backend is running"}


# ---------- Analyze Endpoint ----------
@app.post("/analyze")
def analyze_policy(data: PolicyInput):

    # 🔹 Handle empty input
    if not data.policy_text.strip():
        return {
            "error": "Empty privacy policy text"
        }

    text = data.policy_text.lower()

    # 🔹 Language detection
    try:
        language = detect(text)
    except:
        language = "unknown"

    # 🔹 Data detection
    data_types = []

    if any(word in text for word in ["name", "email", "phone", "address", "location"]):
        data_types.append("Personal Data")

    if any(word in text for word in ["aadhaar", "biometric", "health", "bank", "financial"]):
        data_types.append("Sensitive Data")

    # 🔹 DPDP principle-wise checks
    dpdp_checks = {
        "consent": "consent" in text,
        "purpose": ("purpose" in text or "use" in text),
        "retention": ("store" in text or "retain" in text),
        "user_rights": "right" in text,
        "grievance": ("grievance" in text or "contact" in text)
    }

    # 🔹 DPDP score calculation
    score = sum(dpdp_checks.values())


    # 🔹 Risk level
    if score <= 1:
        risk = "High"
    elif score <= 3:
        risk = "Medium"
    else:
        risk = "Low"

    # 🔹 Recommendations
    recommendations = []

    if "Personal Data" in data_types and score < 3:
        recommendations.append(
            "Clearly specify the purpose of personal data collection."
        )

    if "Sensitive Data" in data_types:
        recommendations.append(
            "Explicit user consent is required for sensitive personal data as per DPDP Act."
        )

    if "consent" not in text:
        recommendations.append(
            "Add clear user consent statements in the privacy policy."
        )

    if score < 2:
        recommendations.append(
            "Mention user rights and grievance redressal mechanisms."
        )

    if not recommendations:
        recommendations.append(
            "Privacy policy shows basic DPDP compliance."
        )

    # 🔹 Explanation
    explanation = []

    if dpdp_checks["consent"]:
        explanation.append("User consent is clearly mentioned.")
    else:
        explanation.append("User consent is not clearly specified.")

    if dpdp_checks["purpose"]:
        explanation.append("Purpose of data usage is mentioned.")
    else:
        explanation.append("Purpose of data usage is unclear.")

    if dpdp_checks["retention"]:
        explanation.append("Data storage or retention is addressed.")
    else:
        explanation.append("Data retention details are missing.")

    if dpdp_checks["user_rights"]:
        explanation.append("User rights are mentioned.")
    else:
        explanation.append("User rights are not clearly explained.")

    # 🔹 Final response
    return {
    "detected_language": language,
    "data_types": data_types,
    "dpdp_score": f"{score}/5",
    "risk_level": risk,
    "dpdp_breakdown": dpdp_checks,
    "explanation": explanation,
    "recommendations": recommendations
    }

