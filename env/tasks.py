def grade_easy(patient, action):
    # Simple check: is priority within 1 of true priority?
    if abs(action['priority'] - patient["true_priority"]) <= 1:
        return 1.0
    return 0.0

def grade_medium(patient, action):
    # Medium: Priority must be exact and department must match
    if action['priority'] == patient["true_priority"] and action['department'] == patient["department"]:
        return 1.0
    return 0.0

def grade_hard(patient, action):
    # Hard: Penalty for low priority on high urgency patients
    score = 0.0
    if action['priority'] == patient["true_priority"] and action['department'] == patient["department"]:
        score += 1.0
    
    if patient["true_priority"] >= 4 and action['priority'] <= 2:
        score -= 1.0
    
    return max(0.0, score)
