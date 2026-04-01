import json
import random
from env.hospital_env import HospitalEnv

with open("data/patients.json") as f:
    data = json.load(f)

env = HospitalEnv(data)

def random_agent(state):
    return {
        "priority": random.randint(1, 5),
        "department": random.choice(["cardiology", "neurology", "general"])
    }

episodes = 10
total_reward = 0

for _ in range(episodes):
    state = env.reset()
    action = random_agent(state)

    _, reward, _, _ = env.step(action)

    print("State:", state)
    print("Action:", action)
    print("Reward:", reward)
    print("------")

    total_reward += reward

print("Average Reward:", total_reward / episodes)