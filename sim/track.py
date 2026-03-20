import numpy as np
import pygame
import fastf1
import math


class Track:
    def __init__(self, track_name):
        self.track_name = track_name
        self.lap_ready = False
        self.lap_start_time = 0
        self.last_lap_time = 0

    def load_track(self):
        session = fastf1.get_session(2025, self.track_name, "Q")
        session.load()

        fastest_lap = session.laps.pick_fastest()
        telemetry = fastest_lap.get_telemetry()

        self.x = telemetry["X"].values
        self.y = telemetry["Y"].values

       

    def transform(self, screen_width, screen_height):
        self.x = self.x - np.min(self.x)
        self.y = self.y - np.min(self.y)

        scale = min(screen_width / np.max(self.x), screen_height / np.max(self.y)) * 0.9

        self.x = self.x * scale + 20
        self.y = screen_height - (self.y * scale + 20)

        dx = np.diff(self.x)
        dy = np.diff(self.y)

        length = np.sqrt(dx**2 + dy**2)
        dx = dx / length
        dy = dy / length

        perp_x = -dy
        perp_y = dx

        width = 25
        self.width = 20

        self.outer_x = self.x[:-1] + perp_x * width
        self.outer_y = self.y[:-1] + perp_y * width
        self.inner_x = self.x[:-1] - perp_x * width
        self.inner_y = self.y[:-1] - perp_y * width

        self.outer_x = np.append(self.outer_x, self.outer_x[0])
        self.outer_y = np.append(self.outer_y, self.outer_y[0])
        self.inner_x = np.append(self.inner_x, self.inner_x[0])
        self.inner_y = np.append(self.inner_y, self.inner_y[0])

        from matplotlib.path import Path
        self.outer_path = Path(list(zip(self.outer_x, self.outer_y)))
        self.inner_path = Path(list(zip(self.inner_x, self.inner_y)))


    def draw(self, surface):
        points = list(zip(self.x + 10, self.y))
        inner_points = list(zip(self.inner_x + 10, self.inner_y))
        outer_points = list(zip(self.outer_x + 10, self.outer_y))
        pygame.draw.lines(surface, (255, 0, 0), True, points)
        pygame.draw.lines(surface, (255, 255, 255), True, inner_points)
        pygame.draw.lines(surface, (255, 255, 255), True, outer_points)

    def check_lap(self, car):
        dx = car.x - self.x[0]
        dy = car.y - self.y[0]
        distance = math.sqrt(dx**2 + dy**2)
        if (distance < 20) and self.lap_ready :

            print("Lap Finished")
            current_time = pygame.time.get_ticks()
            self.last_lap_time = current_time - self.lap_start_time
            self.lap_start_time = current_time
            print(self.last_lap_time/1000)
            self.lap_ready = False
        
        if (distance > 50):
            self.lap_ready = True

    def get_progress(self, car):
        min_dist = float('inf')
        closest_index = 0
        
        for i in range(len(self.x) - 1):
            dx = car.x - self.x[i]
            dy = car.y - self.y[i]
            dist = math.sqrt(dx**2 + dy**2)
            if dist < min_dist:
                min_dist = dist
                closest_index = i
        
        return closest_index

    def get_curvature(self, car, lookahead=10):
        idx = self.get_progress(car)
        if idx + lookahead >= len(self.x) - 1:
            return 0.0
        
        dx1 = self.x[idx + 1] - self.x[idx]
        dy1 = self.y[idx + 1] - self.y[idx]
        dx2 = self.x[idx + lookahead] - self.x[idx]
        dy2 = self.y[idx + lookahead] - self.y[idx]
        
        angle1 = math.atan2(dy1, dx1)
        angle2 = math.atan2(dy2, dx2)
        
        curvature = angle2 - angle1
        # normalize to -1 to 1
        while curvature > math.pi: curvature -= 2 * math.pi
        while curvature < -math.pi: curvature += 2 * math.pi
        
        return curvature / math.pi