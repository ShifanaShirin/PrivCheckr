def evaluate_dpdp(classification_results):
    """
    classification_results = output from classify_sentences()
    """

    consent = classification_results.get("consent", 0) > 0
    purpose = classification_results.get("purpose", 0) > 0
    retention = classification_results.get("retention", 0) > 0
    rights = classification_results.get("rights", 0) > 0
    grievance = classification_results.get("grievance", 0) > 0
    risk_flags = classification_results.get("risk", 0)

    dpdp_checks = {
        "consent": consent,
        "purpose": purpose,
        "retention": retention,
        "user_rights": rights,
        "grievance": grievance
    }

    # Weighted scoring
    score = (
        (2 if consent else 0) +
        (1 if purpose else 0) +
        (1 if retention else 0) +
        (2 if rights else 0) +
        (1 if grievance else 0)
    )

    # Risk determination
    if risk_flags > 1:
        risk_level = "High"
    elif risk_flags == 1:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return dpdp_checks, score, risk_level
