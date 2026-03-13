import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =============================
# TRAINING DATASET
# =============================

texts_consent = [
"We collect personal data only with your consent.",
"You must agree to this policy before using the service.",
"Users must provide consent before personal data is processed.",
"Your consent allows us to collect and store personal information.",
"You may withdraw your consent at any time.",
"We ask for your permission before collecting personal information.",
"Consent is required before accessing certain features.",
"By registering, you agree to our data processing practices.",
"You provide consent by accepting our privacy policy.",
"Users must confirm consent before submitting personal information.",
"Your information will be processed only with your approval.",
"We obtain explicit consent before collecting sensitive information.",
"You may revoke your consent by contacting support.",
"We request user consent prior to data collection.",
"Personal information is collected after obtaining user permission.",
"Users may grant or deny consent for data processing.",
"You can withdraw consent at any time without affecting service access.",
"Consent is obtained during account registration.",
"We require clear user consent before storing personal data.",
"Users must acknowledge the privacy terms before data collection."
]

texts_purpose = [
"Personal data is used to improve our services.",
"We use information to operate and maintain our platform.",
"Your data helps us enhance user experience.",
"We process personal information to deliver requested services.",
"User data is used to manage accounts and provide support.",
"Information collected is used for service improvement.",
"We use personal data to communicate updates.",
"Data is processed to personalize content and recommendations.",
"Collected information is used for analytics and service enhancement.",
"We process personal information to respond to user inquiries.",
"User data helps us monitor platform performance.",
"We use information to ensure platform security.",
"Personal data is processed for operational purposes.",
"We use collected information to improve application functionality.",
"User data supports service delivery and maintenance.",
"We process personal information to send notifications.",
"Information is used to provide technical support.",
"Collected data helps optimize system performance.",
"User data is processed to improve customer support.",
"We use personal information for internal service management."
]

texts_retention = [
"We retain personal data only as long as necessary.",
"User information is stored for a limited period.",
"Personal data is deleted after the purpose is fulfilled.",
"We retain information to comply with legal obligations.",
"Data is stored only for operational requirements.",
"We keep personal data for as long as the account is active.",
"Information may be retained to resolve disputes.",
"User data is deleted when no longer required.",
"We store personal data according to retention policies.",
"Data retention periods depend on service requirements.",
"We may retain information for security purposes.",
"Personal information is removed after account deletion.",
"Data may be stored temporarily for service delivery.",
"We delete outdated personal information regularly.",
"User data is kept only for necessary duration.",
"Retention periods follow legal and operational guidelines.",
"Personal data may be retained for audit purposes.",
"We securely delete data when it is no longer needed.",
"Retention policies ensure proper data management.",
"Information is archived according to company policy."
]

texts_rights = [
"Users have the right to access their personal data.",
"You may request correction of inaccurate information.",
"Users may request deletion of their personal data.",
"You have the right to review your stored information.",
"Users can update their personal details anytime.",
"You may request a copy of your personal data.",
"Users may restrict processing of their information.",
"You may object to certain data processing activities.",
"Users can request data portability.",
"You may modify your personal information.",
"Users may request removal of outdated information.",
"You may access your account data anytime.",
"Users have the right to correct incorrect data.",
"You can manage your privacy preferences.",
"Users may delete their account and associated data.",
"You may request information about data processing.",
"Users may withdraw consent for data processing.",
"You have the right to know how your data is used.",
"Users can control the sharing of their information.",
"You may request restriction of certain processing activities."
]

texts_grievance = [
"For complaints contact our grievance officer.",
"Users may contact our privacy team for concerns.",
"You may email support for privacy issues.",
"Privacy complaints can be submitted through our support system.",
"Our grievance officer handles privacy complaints.",
"Users may contact customer support for data concerns.",
"You may report privacy violations to our support team.",
"Contact privacy@example.com for grievance redressal.",
"Users may submit complaints regarding data handling.",
"Our privacy team responds to user concerns.",
"You may file complaints about data misuse.",
"Users may contact the company regarding privacy issues.",
"Privacy concerns can be addressed through our help desk.",
"Our grievance mechanism ensures complaint resolution.",
"You may report security concerns through email.",
"Users may contact us for data protection queries.",
"Privacy complaints are investigated by our team.",
"You may raise issues related to personal data processing.",
"Users can contact support for privacy clarification.",
"Our team resolves user complaints regarding data privacy."
]

texts_risk = [
"We may share your data with third parties.",
"Personal information may be sold to advertisers.",
"User data may be disclosed to partners.",
"Your information may be transferred to external organizations.",
"We may share personal data for marketing purposes.",
"User data may be used for advertising.",
"Information may be shared with affiliated companies.",
"Data may be retained indefinitely.",
"We cannot guarantee complete data security.",
"Personal information may be transferred internationally.",
"Your data may be disclosed during business transfers.",
"We may share information with analytics providers.",
"User data may be used for targeted advertisements.",
"Personal information may be processed by external vendors.",
"We may disclose data to comply with legal requirements.",
"Information may be shared with service providers.",
"User data may be used for promotional activities.",
"We may transfer personal information to partners.",
"Personal data may be processed outside your country.",
"Your information may be shared with marketing partners."
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
# TRAIN TEST SPLIT
# =============================

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42
)


# =============================
# VECTORIZE TEXT
# =============================

vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# =============================
# TRAIN MODEL
# =============================

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)


# =============================
# PERFORMANCE ANALYSIS
# =============================

predictions = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Performance")
print("======================")

print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))


# =============================
# SAVE MODEL
# =============================

os.makedirs("model", exist_ok=True)

joblib.dump((model, vectorizer), "dpdp_model.pkl")

print("\nDPDP model trained successfully!")