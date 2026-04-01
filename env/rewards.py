def compute_reward(patient, action):
    reward = 0.0
    done = True

    # urgency match
    if abs(action.priority - patient["true_priority"]) <= 1:
        reward += 0.5

    # department match
    if action.department == patient["department"]:
        reward += 0.3

    # penalty
    if patient["true_priority"] >= 4 and action.priority <= 2:
        reward -= 0.5

    return reward, done