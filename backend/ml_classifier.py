import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "dpdp_model.pkl")

model = None
vectorizer = None

if os.path.exists(MODEL_PATH):
    model, vectorizer = joblib.load(MODEL_PATH)


def classify_sentences(processed_text):

    def is_negated(sentence):
        negations = ["no", "not", "never", "without", "do not", "does not"]
        return any(neg in sentence for neg in negations)

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

        # =========================
        # 1️⃣ MACHINE LEARNING
        # =========================
        if model and vectorizer:
            features = vectorizer.transform([s])
            prediction = model.predict(features)[0]

            # Prevent ML false positives from negation
            if prediction in results and not is_negated(s):
                results[prediction] += 1

        # =========================
        # 2️⃣ RULE BASED BOOST
        # =========================

        # CONSENT
        consent_keywords = [
            "user consent",
            "with your consent",
            "provide consent",
            "agree to this policy",
            "accept our privacy policy"
        ]
        if any(w in s for w in consent_keywords) and not is_negated(s):
            results["consent"] += 1


        # PURPOSE
        purpose_keywords = [
            "used to provide services",
            "improve our services",
            "enhance user experience",
            "used for communication"
        ]
        if any(w in s for w in purpose_keywords) and not is_negated(s):
            results["purpose"] += 1


        # RETENTION
        retention_keywords = [
            "retain personal data",
            "data retention",
            "delete your data",
            "stored for a limited period",
            "retain your data"
        ]
        if any(w in s for w in retention_keywords) and not is_negated(s):
            results["retention"] += 1


        # USER RIGHTS 
        rights_keywords = [
            "right to access",
            "right to delete",
            "right to update",
            "withdraw consent",
            "request deletion"
        ]
        if any(w in s for w in rights_keywords) and not is_negated(s):
            results["rights"] += 1


        # GRIEVANCE
        grievance_keywords = [
            "grievance officer",
            "contact support",
            "privacy officer",
            "complaints regarding data",
            "contact us at"
        ]
        if any(w in s for w in grievance_keywords) and not is_negated(s):
            results["grievance"] += 1


        # =========================
        # 3️⃣ RISK DETECTION 
        # =========================
        risk_keywords = [
            "third party",
            "advertisers",
            "sell data",
            "share",
            "external organizations",
            "indefinitely"
        ]

        if any(w in s for w in risk_keywords) and not is_negated(s):
            results["risk"] += 1

    return results