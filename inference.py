from env.hospital_env import HospitalEnv

MAX_STEPS = 10


def get_action(state):
    symptoms = state["symptoms"]
    
    # ✅ handle both string and list (important for medium/hard)
    if isinstance(symptoms, list):
        symptoms_list = [s.lower() for s in symptoms]
        symptoms_text = " ".join(symptoms_list)
    else:
        symptoms_text = symptoms.lower()
        symptoms_list = [symptoms_text]

    age = state["age"]
    hr = state["heart_rate"]
    bp = state["blood_pressure"]

    # =========================================================
    # 🔥 STEP 1: ABSOLUTE PRIORITY (MATCH GENERATOR)
    # =========================================================

    # 🚨 Emergency override (CRITICAL FIX)
    if "unconscious" in symptoms_text or "severe bleeding" in symptoms_text:
        return {
            "department": "emergency",
            "seriousness": 5
        }

    # ❤️ Critical cardiology combo
    if "chest pain" in symptoms_text and "shortness of breath" in symptoms_text:
        return {
            "department": "cardiology",
            "seriousness": 5
        }

    # =========================================================
    # 🧠 STEP 2: DEPARTMENT SCORING (SECONDARY)
    # =========================================================

    dept_scores = {
        "cardiology": 0,
        "pulmonology": 0,
        "neurology": 0,
        "orthopedics": 0,
        "emergency": 0,
        "general": 0
    }

    for symptom in symptoms_list:

        if "chest pain" in symptom:
            dept_scores["cardiology"] += 3

        if "palpitations" in symptom:
            dept_scores["cardiology"] += 2

        if "shortness of breath" in symptom or "cough" in symptom:
            dept_scores["pulmonology"] += 3

        if "head injury" in symptom or "dizziness" in symptom:
            dept_scores["neurology"] += 3

        if "fracture" in symptom:
            dept_scores["orthopedics"] += 3

        # ⚠️ keep but lower importance (since override handled above)
        if "bleeding" in symptom or "trauma" in symptom:
            dept_scores["emergency"] += 2

        if "fever" in symptom:
            dept_scores["general"] += 2

    # fallback
    dept_scores["general"] += 1

    department = max(dept_scores, key=dept_scores.get)

    # =========================================================
    # 🧠 STEP 3: SERIOUSNESS (ALIGNED WITH GENERATOR)
    # =========================================================

    score = 1  # base

    # 🔴 vitals
    if hr > 120:
        score += 2
    elif hr > 100:
        score += 1

    if bp < 90:
        score += 2
    elif bp < 100:
        score += 1

    # 🧠 symptoms severity
    if any(x in symptoms_text for x in ["unconscious", "severe bleeding"]):
        score += 3
    elif any(x in symptoms_text for x in ["chest pain", "shortness of breath"]):
        score += 2
    elif any(x in symptoms_text for x in ["head injury"]):
        score += 2
    elif any(x in symptoms_text for x in ["fracture"]):
        score += 1

    # 👶 age risk
    if age > 65:
        score += 1

    seriousness = min(5, score)

    # =========================================================
    # ✅ FINAL OUTPUT
    # =========================================================

    return {
        "seriousness": seriousness,
        "department": department
    }


def main():
    env = HospitalEnv(task="medium")  # change to "hard" later
    state = env.reset()

    total_reward = 0

    for step in range(MAX_STEPS):
        action = get_action(state)

        next_state, reward, done, info = env.step(action)

        print(f"Step {step+1}")
        print("State:", state)
        print("Action:", action)
        print("Reward:", reward)
        print("-" * 30)

        total_reward += reward
        state = next_state

        if done:
            break

    print("Total Reward:", total_reward)


if __name__ == "__main__":
    main()