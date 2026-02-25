import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# =============================
# TRAINING DATASET
# =============================

texts_consent = [
"We collect personal data only with your consent.",
"Users must agree before we process their personal information.",
"You may withdraw consent at any time.",
"Consent is required before collecting personal data.",
"We ask for your permission before storing personal information.",
"Users explicitly consent when registering for the service.",
"Data is collected only after obtaining user consent.",
"Users may revoke consent by contacting support.",
"Your consent allows us to process personal information.",
"We require consent before processing personal data."
]

texts_purpose = [
"Personal data is used to improve our services.",
"We process information to provide better services.",
"User data helps us enhance platform functionality.",
"The purpose of collecting information is to improve service delivery.",
"Information is used to manage user accounts.",
"We process data to communicate with users.",
"User information helps personalize content.",
"Data is used to provide support services.",
"Collected information helps improve user experience.",
"We process personal data for service delivery."
]

texts_retention = [
"We retain personal data only as long as necessary.",
"User information is stored for a limited time.",
"Data is deleted when no longer required.",
"We retain information for legal compliance.",
"Personal data is removed after the service ends.",
"We store personal information for operational purposes.",
"Data is retained according to company policy.",
"Old user data is securely deleted.",
"Personal information is kept only for required duration.",
"Retention policies ensure timely deletion of data."
]

texts_rights = [
"Users have the right to access their personal data.",
"Users can request correction of inaccurate information.",
"You may request deletion of your personal data.",
"Users can update their personal information.",
"You may request a copy of stored personal data.",
"Users have the right to modify personal information.",
"You can restrict processing of your personal data.",
"Users may exercise privacy rights through support.",
"You can review and manage your personal data.",
"Users may update or delete their stored information."
]

texts_grievance = [
"For complaints contact our grievance officer.",
"Users may email our privacy team for concerns.",
"Contact support for privacy issues.",
"Our grievance officer handles privacy complaints.",
"You can report privacy issues via email.",
"Users may contact us for data protection concerns.",
"Privacy complaints are handled by our support team.",
"Contact privacy@example.com for grievance redressal.",
"Users may file complaints regarding data handling.",
"Our privacy team addresses data concerns."
]

texts_risk = [
"We may share your data with third parties.",
"Personal information may be sold to advertisers.",
"Data may be stored indefinitely.",
"We may disclose your information without notice.",
"User data may be shared with partners.",
"Your information may be transferred to third parties.",
"We cannot guarantee data security.",
"Information may be retained permanently.",
"We may share personal data for marketing purposes.",
"User data may be disclosed to external organizations."
]


# =============================
# COMBINE DATA
# =============================

texts = (
    texts_consent +
    texts_purpose +
    texts_retention +
    texts_rights +
    texts_grievance +
    texts_risk
)

labels = (
    ["consent"] * len(texts_consent) +
    ["purpose"] * len(texts_purpose) +
    ["retention"] * len(texts_retention) +
    ["rights"] * len(texts_rights) +
    ["grievance"] * len(texts_grievance) +
    ["risk"] * len(texts_risk)
)


# =============================
# VECTORIZE TEXT
# =============================

vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    stop_words="english"
)

X = vectorizer.fit_transform(texts)


# =============================
# TRAIN MODEL
# =============================

model = LogisticRegression(max_iter=1000)
model.fit(X, labels)


# =============================
# SAVE MODEL
# =============================

os.makedirs("model", exist_ok=True)

joblib.dump((model, vectorizer), "dpdp_model.pkl")

print("DPDP model trained successfully!")