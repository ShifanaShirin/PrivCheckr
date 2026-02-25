import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "dpdp_model.pkl")

model = None
vectorizer = None

if os.path.exists(MODEL_PATH):
    model, vectorizer = joblib.load(MODEL_PATH)


def classify_sentences(processed_text):

    results = {
        "consent": 0,
        "purpose": 0,
        "retention": 0,
        "rights": 0,
        "grievance": 0,
        "risk": 0
    }

    sentences = [item["sentence"].lower() for item in processed_text]

    for s in sentences:

        # ---------- CONSENT ----------
        if any(w in s for w in [
            "user consent",
            "with your consent",
            "provide consent",
            "agree to this policy"
        ]):
            results["consent"] = 1

        # ---------- PURPOSE ----------
        if any(w in s for w in [
            "purpose of collecting",
            "purpose of data",
            "used to provide services",
            "used to improve our services"
        ]):
            results["purpose"] = 1

        # ---------- RETENTION ----------
        if any(w in s for w in [
            "data retention",
            "retain your data",
            "retain personal data",
            "delete your data after"
        ]):
            results["retention"] = 1

        # ---------- USER RIGHTS ----------
        if any(w in s for w in [
            "right to access",
            "right to delete",
            "right to update",
            "withdraw consent"
        ]):
            results["rights"] = 1

        # ---------- GRIEVANCE ----------
        if any(w in s for w in [
            "grievance officer",
            "contact us at",
            "privacy officer",
            "complaints regarding data"
        ]):
            results["grievance"] = 1

        # ---------- RISK ----------
        if any(w in s for w in [
            "third party",
            "advertisers",
            "sell data",
            "indefinitely",
            "share with partners"
        ]):
            results["risk"] += 1

    return results