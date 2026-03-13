def evaluate_dpdp(classification_results):

    consent = classification_results.get("consent", 0)
    purpose = classification_results.get("purpose", 0)
    retention = classification_results.get("retention", 0)
    rights = classification_results.get("rights", 0)
    grievance = classification_results.get("grievance", 0)
    risk_flags = classification_results.get("risk", 0)

    dpdp_checks = {
        "consent": bool(consent),
        "purpose": bool(purpose),
        "retention": bool(retention),
        "user_rights": bool(rights),
        "grievance": bool(grievance)
    }

    # -----------------------
    # DPDP Score
    # -----------------------

    score = (
        (2 if consent else 0) +
        (1 if purpose else 0) +
        (1 if retention else 0) +
        (2 if rights else 0) +
        (1 if grievance else 0)
    )

    max_score = 7

    # -----------------------
    # Risk Evaluation
    # -----------------------

    if score >= 6 and risk_flags == 0:
        risk_level = "Low"

    elif score >= 4 and risk_flags <= 2:
        risk_level = "Medium"

    else:
        risk_level = "High"

    return dpdp_checks, score, risk_level