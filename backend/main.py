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

    # 🔹 DPDP scoring (partial – interim level)
    score = 0
    if "consent" in text:
        score += 1
    if "purpose" in text or "use" in text:
        score += 1
    if "store" in text or "retain" in text:
        score += 1
    if "right" in text:
        score += 1
    if "grievance" in text or "contact" in text:
        score += 1

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

    # 🔹 Final response
    return {
        "detected_language": language,
        "data_types": data_types,
        "dpdp_score": f"{score}/5",
        "risk_level": risk,
        "recommendations": recommendations
    }
