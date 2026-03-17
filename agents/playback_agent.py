import numpy as np
from agents.base_agent import BaseAgent

class PlaybackAgent(BaseAgent):
    def __init__(self, telemetry_x, telemetry_y):
        super().__init__(name="F1 Playback", color=(255, 215, 0))
        self.telemetry_x = telemetry_x
        self.telemetry_y = telemetry_y
        self.frame = 0

    def act(self, observation, car_state, sim_time):
        x = self.telemetry_x[self.frame % len(self.telemetry_x)]
        y = self.telemetry_y[self.frame % len(self.telemetry_y)]
        self.frame += 1
        return {"x": x, "y": y, "playback": True}
