from agents.base_agent import BaseAgent
from stable_baselines3 import PPO

class PPOAgent(BaseAgent):
    def __init__(self, model_path):
        super().__init__(name="PPO", color=(0, 255, 100))
        self.model = PPO.load(model_path)
    
    def act(self, observation, car_state, sim_time):
        action, _ = self.model.predict(observation)
        return {
            "throttle": float(action[0]),
            "brake": float(action[1]),
            "steer": float(action[2])
        }