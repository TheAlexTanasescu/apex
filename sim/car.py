import pygame
import math
from sim.track import Track
from matplotlib.path import Path
import numpy as np


class Car:
    def __init__(self, x, y, speed, angle, screen_width, screen_height):
        self.x = x
        self.y = y
        self.speed = 0
        self.angle = 0
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.spawn_frames = 60

    def draw(self, surface):
        car_surface = pygame.Surface((10, 20), pygame.SRCALPHA)
        pygame.draw.rect(car_surface, (255, 0, 0), (0, 0, 10, 20))
        rotated = pygame.transform.rotate(car_surface, -self.angle)
        rect = rotated.get_rect(center=(self.x, self.y))
        surface.blit(rotated, rect)

    def update(self, keys, track, action=None):
        if action and not action.get("playback"):
            self.speed += action["throttle"] * 0.1
            self.speed -= action["brake"] * 0.2
            self.angle += action["steer"] * 2

        if action and action.get("playback"):
            self.x = action["x"]
            self.y = action["y"]

        if keys:
            if keys[pygame.K_UP]:
                self.speed += 0.1
            if keys[pygame.K_DOWN]:
                self.speed -= 0.1
            if keys[pygame.K_LEFT]:
                self.angle -= 1
            if keys[pygame.K_RIGHT]:
                self.angle += 1

        self.x -= self.speed * math.cos(math.radians(self.angle + 90))
        self.y -= self.speed * math.sin(math.radians(self.angle + 90))

        self.speed = min(self.speed, 3)
        self.speed = self.speed * 0.95

        self.check_and_clamp(track)

    def check_and_clamp(self, track):
        if self.spawn_frames > 0:
            self.spawn_frames -= 1
            return

        if self.check_collision(track):
            min_dist = float("inf")
            nearest_x, nearest_y = self.x, self.y

            for i in range(len(track.outer_x)):
                dx = self.x - track.outer_x[i]
                dy = self.y - track.outer_y[i]
                dist = math.sqrt(dx**2 + dy**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_x = track.outer_x[i]
                    nearest_y = track.outer_y[i]

            for i in range(len(track.inner_x)):
                dx = self.x - track.inner_x[i]
                dy = self.y - track.inner_y[i]
                dist = math.sqrt(dx**2 + dy**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_x = track.inner_x[i]
                    nearest_y = track.inner_y[i]

            self.x = nearest_x
            self.y = nearest_y
            self.speed *= -0.3

    def check_collision(self, track):
        car_pos = (self.x, self.y)
        inside_outer = track.outer_path.contains_point(car_pos)
        inside_inner = track.inner_path.contains_point(car_pos)
        if not inside_outer or inside_inner:
            return True
        return False

    def cast_ray(self, track, angle_offset, max_distance):
        ray_angle = math.radians(self.angle + angle_offset)

        for dist in range(1, max_distance):
            rx = self.x + dist * math.cos(ray_angle)
            ry = self.y + dist * math.sin(ray_angle)

            if not track.outer_path.contains_point(
                (rx, ry)
            ) or track.inner_path.contains_point((rx, ry)):
                return dist / max_distance

        return 1.0

    def get_observation(self, track):
        obs = np.zeros(13)

        obs[0] = self.x / self.screen_width
        obs[1] = self.y / self.screen_height
        obs[2] = self.speed / 5.0
        obs[3] = math.cos(math.radians(self.angle))
        obs[4] = math.sin(math.radians(self.angle))
        obs[5] = 0
        obs[6] = self.cast_ray(track, 0, 200)
        obs[7] = self.cast_ray(track, -90, 200)
        obs[8] = self.cast_ray(track, 90, 200)
        obs[9] = self.cast_ray(track, -45, 200)
        obs[10] = self.cast_ray(track, 45, 200)
        obs[11] = 0
        obs[12] = 0

        return obs
