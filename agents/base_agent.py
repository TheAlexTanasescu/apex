class BaseAgent:
    def __init__(self, name, color):
        self.name = name
        self.color = color

    def act(self, observation, car_state, sim_time):
        raise NotImplementedError("Each agent must implement act()")