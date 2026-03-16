import numpy as np
import pygame
import fastf1


class Track:
    def __init__(self, track_name):
        self.track_name = track_name

    def load_track(self):
        session = fastf1.get_session(2023, self.track_name, "Q")
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

        width = 15

        self.outer_x = self.x[:-1] - perp_x * width
        self.outer_y = self.y[:-1] - perp_y * width
        self.inner_x = self.x[:-1] + perp_x * width
        self.inner_y = self.y[:-1] + perp_y * width

    def draw(self, surface):
        points = list(zip(self.x, self.y))
        inner_points = list(zip(self.inner_x, self.inner_y))
        outer_points = list(zip(self.outer_x, self.outer_y))
        pygame.draw.lines(surface, (255, 0, 0), True, points)
        pygame.draw.lines(surface, (255, 255, 255), True, inner_points)
        pygame.draw.lines(surface, (255, 255, 255), True, outer_points)

