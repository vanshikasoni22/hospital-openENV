# EASY → only department match
def easy_task_reward(patient, action):
    return 1 if action["department"] == patient["department"] else 0


# MEDIUM → department + priority
def medium_task_reward(patient, action):
    score = 0

    if action["department"] == patient["department"]:
        score += 0.5

    if action["priority"] == patient["true_priority"]:
        score += 0.5

    return score


# HARD → full reward logic (your existing one)
def hard_task_reward(patient, action):
    reward = 0

    # Department match
    if action["department"] == patient["department"]:
        reward += 0.5
    else:
        reward -= 0.3

    # Priority match
    if action["priority"] == patient["true_priority"]:
        reward += 0.5
    else:
        reward -= 0.2

    # Extra penalty for critical misclassification
    if patient["true_priority"] == 5 and action["priority"] < 3:
        reward -= 0.5

    return reward