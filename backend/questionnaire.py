# ================================================================
# RETINASCAN AI — Patient Symptom Questionnaire
# ================================================================
# Collects patient-reported visual symptoms via multiple-choice
# questions and compares them against expected symptom profiles
# for each disease class.
#
# IMPORTANT:
#   - The questionnaire does NOT diagnose any disease.
#   - It does NOT override the OCT model prediction.
#   - It provides a consistency check between patient-reported
#     symptoms and the AI-predicted OCT category.
#   - Sudden vision loss or sudden significant vision changes
#     should always prompt urgent professional assessment.
# ================================================================


# ================================================================
# SYMPTOM QUESTIONS
# ================================================================

SYMPTOM_QUESTIONS = [
    {
        "id": "blurred_vision",
        "question": "Are you experiencing blurred vision?",
        "options": ["No", "Mild", "Moderate", "Severe"],
        "type": "single",
    },
    {
        "id": "distorted_vision",
        "question": "Do straight lines appear wavy or distorted?",
        "options": ["No", "Occasionally", "Frequently", "Constantly"],
        "type": "single",
    },
    {
        "id": "dark_spots",
        "question": "Do you see dark or blank spots in your vision?",
        "options": ["No", "Small spots", "Moderate area", "Large area"],
        "type": "single",
    },
    {
        "id": "reading_difficulty",
        "question": "Do you have difficulty reading or recognising faces?",
        "options": ["No", "Mild difficulty", "Moderate difficulty", "Severe difficulty"],
        "type": "single",
    },
    {
        "id": "reduced_central_vision",
        "question": "Have you noticed reduced central vision?",
        "options": ["No", "Slightly reduced", "Moderately reduced", "Severely reduced"],
        "type": "single",
    },
    {
        "id": "sudden_changes",
        "question": "Have you experienced sudden changes in your vision?",
        "options": ["No", "Yes, minor", "Yes, significant"],
        "type": "single",
    },
    {
        "id": "symptom_duration",
        "question": "How long have you been experiencing these symptoms?",
        "options": [
            "No symptoms",
            "Less than 1 week",
            "1 to 4 weeks",
            "1 to 6 months",
            "More than 6 months",
        ],
        "type": "single",
    },
    {
        "id": "affected_eyes",
        "question": "Which eye(s) are affected?",
        "options": ["Neither", "One eye", "Both eyes", "Not sure"],
        "type": "single",
    },
]


# ================================================================
# EXPECTED SYMPTOM PROFILES PER DISEASE CLASS
# ================================================================
# These are general associations between retinal conditions and
# commonly reported symptoms. They are NOT diagnostic criteria.

_DISEASE_SYMPTOM_ASSOCIATIONS = {
    "CNV": {
        "description": (
            "Choroidal neovascularization (wet AMD) is often associated "
            "with distorted vision, central vision loss, and dark spots."
        ),
        "expected_positive": [
            "blurred_vision",
            "distorted_vision",
            "dark_spots",
            "reduced_central_vision",
        ],
    },
    "DME": {
        "description": (
            "Diabetic macular edema is often associated with blurred "
            "vision, difficulty reading, and reduced central vision."
        ),
        "expected_positive": [
            "blurred_vision",
            "reading_difficulty",
            "reduced_central_vision",
        ],
    },
    "DRUSEN": {
        "description": (
            "Drusen deposits (dry AMD) may be asymptomatic in early "
            "stages. Later stages may cause mild blurring, difficulty "
            "reading, or distorted vision."
        ),
        "expected_positive": [
            "blurred_vision",
            "distorted_vision",
            "reading_difficulty",
        ],
    },
    "NORMAL": {
        "description": (
            "A normal retinal scan would be expected in a patient "
            "without significant visual symptoms."
        ),
        "expected_positive": [],
    },
}


def get_questions():
    """Return the list of symptom questions."""
    return SYMPTOM_QUESTIONS


def _is_positive_response(question_id, answer):
    """Determine if an answer indicates a positive (symptomatic) response."""
    if answer is None:
        return False

    # The first option in each question is the 'no symptom' option
    question = None
    for q in SYMPTOM_QUESTIONS:
        if q["id"] == question_id:
            question = q
            break

    if question is None:
        return False

    # First option is always the negative/no-symptom response
    if answer == question["options"][0]:
        return False

    # For symptom_duration: "No symptoms" is negative
    if question_id == "symptom_duration" and answer == "No symptoms":
        return False

    # For affected_eyes: "Neither" is negative
    if question_id == "affected_eyes" and answer == "Neither":
        return False

    return True


def evaluate_consistency(prediction, answers):
    """
    Compare patient-reported symptoms against the expected symptom
    profile for the predicted disease class.

    Args:
        prediction: str — AI-predicted class (CNV, DME, DRUSEN, NORMAL)
        answers: dict — {question_id: selected_answer}

    Returns:
        dict with consistency analysis results.
    """
    if prediction not in _DISEASE_SYMPTOM_ASSOCIATIONS:
        return {
            "consistent": None,
            "summary": "Cannot assess consistency — prediction class not recognised.",
            "details": [],
            "urgent_referral": False,
            "symptom_count": 0,
            "reported_symptoms": [],
        }

    profile = _DISEASE_SYMPTOM_ASSOCIATIONS[prediction]
    expected_positive_ids = set(profile["expected_positive"])

    # Count reported symptoms (positive responses)
    reported_symptoms = []
    for q in SYMPTOM_QUESTIONS:
        qid = q["id"]
        answer = answers.get(qid)
        if _is_positive_response(qid, answer):
            reported_symptoms.append({
                "id": qid,
                "question": q["question"],
                "answer": answer,
            })

    symptom_count = len(reported_symptoms)
    reported_ids = {s["id"] for s in reported_symptoms}

    # Check for urgent referral: sudden significant changes
    sudden_answer = answers.get("sudden_changes", "No")
    urgent_referral = (sudden_answer == "Yes, significant")

    # Consistency logic
    details = []

    if prediction == "NORMAL":
        # NORMAL + no symptoms = consistent
        # NORMAL + symptoms = inconsistent (recommend review)
        if symptom_count == 0:
            consistent = True
            summary = (
                "The patient reports no significant visual symptoms, "
                "which is consistent with the Normal OCT classification."
            )
        else:
            consistent = False
            summary = (
                "The patient reports visual symptoms despite a Normal "
                "OCT classification. Professional review is recommended "
                "to evaluate whether additional clinical assessment "
                "is warranted."
            )
            for s in reported_symptoms:
                details.append(
                    f"Patient reports: {s['answer']} — {s['question']}"
                )
    else:
        # Disease class: check overlap with expected symptoms
        matching = reported_ids & expected_positive_ids
        unexpected_absent = expected_positive_ids - reported_ids

        if symptom_count == 0:
            consistent = False
            summary = (
                f"The OCT scan was classified as {prediction}, but the "
                f"patient reports no visual symptoms. Some retinal "
                f"conditions can be present before symptoms appear, but "
                f"professional review is recommended."
            )
        elif len(matching) >= 2:
            consistent = True
            summary = (
                f"The patient-reported symptoms are generally consistent "
                f"with the {prediction} OCT classification. "
                f"{len(matching)} of the reported symptoms match the "
                f"expected symptom profile."
            )
        elif len(matching) >= 1:
            consistent = True
            summary = (
                f"Some patient-reported symptoms are consistent with "
                f"the {prediction} classification, though the symptom "
                f"profile is limited. Professional review is recommended."
            )
        else:
            consistent = False
            summary = (
                f"The patient-reported symptoms do not closely match "
                f"the expected profile for {prediction}. This does not "
                f"necessarily indicate an incorrect classification, but "
                f"professional review is recommended."
            )

        if unexpected_absent:
            absent_names = []
            for qid in unexpected_absent:
                for q in SYMPTOM_QUESTIONS:
                    if q["id"] == qid:
                        absent_names.append(q["question"])
                        break
            if absent_names:
                details.append(
                    "Expected symptoms not reported: "
                    + "; ".join(absent_names)
                )

    # Urgent referral warning
    if urgent_referral:
        details.insert(0, (
            "URGENT: Patient reports sudden significant vision changes. "
            "Immediate professional medical assessment is strongly "
            "recommended regardless of OCT classification."
        ))

    return {
        "consistent": consistent,
        "summary": summary,
        "details": details,
        "urgent_referral": urgent_referral,
        "symptom_count": symptom_count,
        "reported_symptoms": reported_symptoms,
        "disease_profile_description": profile["description"],
    }

