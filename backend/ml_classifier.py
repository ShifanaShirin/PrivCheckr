import joblib
import os

MODEL_PATH = "model/dpdp_model.pkl"

if os.path.exists(MODEL_PATH):
    model, vectorizer = joblib.load(MODEL_PATH)
else:
    model = None
    vectorizer = None


def classify_sentences(processed_text):
    """
    processed_text = output from preprocess_text()
    """

    if not model or not vectorizer:
        return {}

    sentences = [item["sentence"] for item in processed_text]

    X = vectorizer.transform(sentences)
    predictions = model.predict(X)

    results = {
        "consent": 0,
        "purpose": 0,
        "retention": 0,
        "rights": 0,
        "grievance": 0,
        "risk": 0
    }

    for label in predictions:
        if label in results:
            results[label] += 1

    return results
