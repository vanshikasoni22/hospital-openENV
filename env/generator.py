import random


# ✅ Only keep symptom names (no fixed department mapping here)
SYMPTOMS = [
    "chest pain",
    "head injury",
    "fracture",
    "shortness of breath",
    "fever",
    "unconscious",
    "severe bleeding",
    "palpitations",
    "dizziness",
    "cough"
]


# =========================================================
# 🧠 DEPARTMENT DECISION (CORE LOGIC)
# =========================================================
def get_department(symptoms):

    # ensure list
    if isinstance(symptoms, str):
        symptoms = [symptoms]

    symptoms = [s.lower() for s in symptoms]

    # 🔥 STEP 1: ABSOLUTE PRIORITY (CRITICAL CASES)
    if any(s in ["unconscious", "severe bleeding"] for s in symptoms):
        return "emergency"

    if "chest pain" in symptoms and "shortness of breath" in symptoms:
        return "cardiology"

    # 🧠 STEP 2: MULTI-SYMPTOM SCORING
    dept_scores = {
        "cardiology": 0,
        "pulmonology": 0,
        "neurology": 0,
        "orthopedics": 0,
        "emergency": 0,
        "general": 0
    }

    for s in symptoms:

        if "chest pain" in s:
            dept_scores["cardiology"] += 3

        if "palpitations" in s:
            dept_scores["cardiology"] += 2

        if "shortness of breath" in s or "cough" in s:
            dept_scores["pulmonology"] += 3

        if "head injury" in s or "dizziness" in s:
            dept_scores["neurology"] += 3

        if "fracture" in s:
            dept_scores["orthopedics"] += 3

        if "bleeding" in s or "trauma" in s:
            dept_scores["emergency"] += 2

        if "fever" in s:
            dept_scores["general"] += 2

    # small bias toward general
    dept_scores["general"] += 1

    return max(dept_scores, key=dept_scores.get)


# =========================================================
# 🧠 PATIENT GENERATION
# =========================================================
def generate_patient(task="easy"):

    # 🔁 STEP 1: Select symptoms based on difficulty
    if task == "easy":
        symptoms = [random.choice(SYMPTOMS)]

    elif task == "medium":
        symptoms = random.sample(SYMPTOMS, 2)

    elif task == "hard":
        symptoms = random.sample(SYMPTOMS, random.randint(2, 4))

    else:
        raise ValueError("Invalid task level")

    # 🔥 Optional: increase emergency cases (better training)
    if random.random() < 0.2:
        symptoms = ["unconscious"]

    # 🧠 STEP 2: Assign department using unified logic
    department = get_department(symptoms)

    # 🧍 STEP 3: Create patient
    patient = {
        "symptoms": symptoms,  # ✅ always a list
        "age": random.randint(20, 80),
        "heart_rate": random.randint(60, 140),
        "blood_pressure": random.randint(80, 160),
        "department": department
    }

    # =========================================================
    # 🧠 STEP 4: SERIOUSNESS SCORING (MATCH INFERENCE)
    # =========================================================

    score = 1  # base

    # 🔴 Vital-based scoring
    if patient["heart_rate"] > 120:
        score += 2
    elif patient["heart_rate"] > 100:
        score += 1

    if patient["blood_pressure"] < 90:
        score += 2
    elif patient["blood_pressure"] < 100:
        score += 1

    # 🧠 Symptom-based scoring (multi-symptom aware)
    for symptom in symptoms:
        if symptom in ["unconscious", "severe bleeding"]:
            score += 3
        elif symptom in ["chest pain", "shortness of breath"]:
            score += 2
        elif symptom in ["head injury"]:
            score += 2
        elif symptom in ["fracture"]:
            score += 1
        elif symptom in ["fever"]:
            score += 0

    # 👶 Age-based risk
    if patient["age"] > 65:
        score += 1

    # ✅ Final seriousness (1–5 scale)
    patient["true_seriousness"] = min(5, score)

    return patient