from env.models import Action
from env.rewards import compute_reward
from env.generator import generate_patient


class HospitalEnv:

    def __init__(self, data=None):
        self.data = data
        self.patient = None

    def reset(self):
        self.patient = generate_patient()
        return self.state()

    def state(self):
        return {
            "symptoms": self.patient["symptoms"],
            "age": self.patient["age"],
            "heart_rate": self.patient["heart_rate"],
            "blood_pressure": self.patient["blood_pressure"]
        }

    def step(self, action_dict):
        # Convert dict → Action object
        action = Action(**action_dict)

        # ✅ FIX: compute_reward returns ONLY reward
        reward = compute_reward(self.patient, action.__dict__)

        # In this environment, each episode = 1 step
        done = True

        next_state = None

        return next_state, reward, done, {
            "true_priority": self.patient["true_priority"],
            "true_department": self.patient["department"]
        }