import numpy as np
from agents.base_agent import BaseAgent

class HeuristicAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Heuristic", color=(255, 100, 0))
        self.stuck_frames = 0

    def act(self, observation, car_state, sim_time):
        ray_ahead = observation[6]
        ray_right = observation[7]
        ray_left = observation[8]
        

        if ray_ahead < 0.2:
            self.stuck_frames += 1
        else:
            self.stuck_frames = 0

        if self.stuck_frames > 30:
            return {"throttle": 0.0, "brake": 1.0, "steer": 1.0}  # reverse and turn

        if ray_ahead < 0.3:
            steer = 1.0 if ray_left > ray_right else -1.0
            return {"throttle": 0.0, "brake": 1.0, "steer": steer}
        
        return {"throttle": 1.0, "brake": 0.0, "steer": float(ray_left - ray_right)*0.5}