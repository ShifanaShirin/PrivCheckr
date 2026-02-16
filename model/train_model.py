import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample training dataset (expand later)
texts = [
    "We collect personal data with your consent",
    "Your data will be used for service improvement",
    "We retain your data for 6 months",
    "You have the right to delete your account",
    "Contact us for grievance redressal",
    "We may share your data without notice",
    "Data may be stored indefinitely"
]

labels = [
    "consent",
    "purpose",
    "retention",
    "rights",
    "grievance",
    "risk",
    "risk"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

os.makedirs("model", exist_ok=True)

joblib.dump((model, vectorizer), "model/dpdp_model.pkl")

print("Model trained and saved successfully!")
