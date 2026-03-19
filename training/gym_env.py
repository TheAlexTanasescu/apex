from gymnasium import spaces
import gymnasium as gym
import numpy as np
import math
from sim.car import Car
from sim.track import Track

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800

class RacingEnv(gym.Env):
    def __init__(self, track_name):
        super().__init__()
        self.track = Track(track_name)
        self.track.load_track()
        self.track.transform(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(13,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.spawn_index = 0
        for i in range(len(self.track.x) - 1):
            dx = self.track.x[i] - self.track.x[0]
            dy = self.track.y[i] - self.track.y[0]
            dist = math.sqrt(dx**2 + dy**2)
            if dist > 50:
                self.spawn_index = i
                break
        
        dx = self.track.x[self.spawn_index + 1] - self.track.x[self.spawn_index]
        dy = self.track.y[self.spawn_index + 1] - self.track.y[self.spawn_index]
        self.spawn_angle = math.degrees(math.atan2(-dy, dx)) + 180
        self.car = Car(self.track.x[self.spawn_index], self.track.y[self.spawn_index], 0, self.spawn_angle, SCREEN_WIDTH, SCREEN_HEIGHT)

    def reset(self, seed=None, options=None):
        self.car.x = self.track.x[self.spawn_index]
        self.car.y = self.track.y[self.spawn_index]
        self.car.angle = self.spawn_angle
        self.car.speed = 0
        return self.car.get_observation(self.track) , {}

    def step(self, action):
        # convert action array to dict
        action_dict = {
            "throttle": float(action[0]),
            "brake": float(action[1]),
            "steer": float(action[2])
        }
        
        # get progress before update
        prev_progress = self.track.get_progress(self.car)
        
        # update car
        self.car.update(None, self.track, action_dict)
        
        # get progress after update
        new_progress = self.track.get_progress(self.car)
        
        # calculate reward
        reward = 0
        if new_progress > prev_progress and new_progress < prev_progress + 20:
            reward += new_progress - prev_progress  # forward progress
        if self.car.check_collision(self.track):
            reward -= 1  # wall hit penalty
        if self.car.speed < 0:
            reward -= 2  # reversing penalty
        
        # check if done
        terminated = False
        if new_progress > len(self.track.x) * 0.95:
            terminated = True  # completed lap
        
        obs = self.car.get_observation(self.track)
        return obs, reward, terminated, False, {}