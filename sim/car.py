import pygame
import math
from sim.track import Track
from matplotlib.path import Path
import numpy as np


class Car:
    def __init__(self, x, y, speed, angle, screen_width, screen_height):
        # Car positions
        self.x = x
        self.y = y
        # Car dimensions
        self.width = 10
        self.height = 20
        self.car_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        # Car movement
        self.speed = 0
        self.angle = 0
        # Screen dimensions
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.spawn_frames = 60

    def draw(self, surface):
        pygame.draw.rect(self.car_surface, (255, 0, 0), (0, 0, self.width, self.height))
        rotated = pygame.transform.rotate(self.car_surface, -self.angle)
        rect = rotated.get_rect(center=(self.x, self.y))
        surface.blit(rotated, rect)

    def update(self, keys, track, action=None):
        if action and not action.get("playback"):
            self.speed += action["throttle"] * 0.1
            self.speed -= action["brake"] * 0.2
            self.angle += action["steer"] * 4

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

        self.speed = max(min(self.speed, 2), -2)

        self.x -= self.speed * math.cos(math.radians(self.angle + 90))
        self.y -= self.speed * math.sin(math.radians(self.angle + 90))

        self.speed = self.speed * 0.95

        self.check_and_clamp(track)
        self.check_and_clamp(track)

    def check_and_clamp(self, track):
        if self.spawn_frames > 0:
            self.spawn_frames -= 1
            return
            
        if self.check_collision(track):
            min_dist = float('inf')
            nearest_x, nearest_y = self.x, self.y
            
            for i in range(len(track.x) - 1):
                dx = self.x - track.x[i]
                dy = self.y - track.y[i]
                dist = math.sqrt(dx**2 + dy**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_x = track.x[i]
                    nearest_y = track.y[i]
            
            self.x = nearest_x
            self.y = nearest_y
            self.speed *= -0.3

    def check_collision(self, track):
        t = ((self.x - track.x1) * track.dx_seg + (self.y - track.y1) * track.dy_seg) / (track.len_sq)
        t = np.clip(t, 0, 1)

        closest_x = track.x1 + t * track.dx_seg
        closest_y = track.y1 + t * track.dy_seg
        dist_sq = (self.x - closest_x)**2 + (self.y - closest_y)**2
        min_dist = np.sqrt(np.min(dist_sq))
        return min_dist > track.width

    def cast_ray(self, track, angle_offset, max_distance):
        ray_angle = math.radians(self.angle + angle_offset)
        dists = np.arange(1, max_distance)
        rx = self.x + dists * math.cos(ray_angle)
        ry = self.y + dists * math.sin(ray_angle)
        
        cx = track.x[:-1]
        cy = track.y[:-1]
        
        for i, (rpx, rpy) in enumerate(zip(rx, ry)):
            diff_x = rpx - cx
            diff_y = rpy - cy
            t = (diff_x * track.dx_seg + diff_y * track.dy_seg) / (track.len_sq)
            t = np.clip(t, 0, 1)

            closest_x = track.x1 + t * track.dx_seg
            closest_y = track.y1 + t * track.dy_seg
            dist_sq = (self.x - closest_x)**2 + (self.y - closest_y)**2
            min_dist = np.sqrt(np.min(dist_sq))
            if min_dist > track.width:
                return dists[i] / max_distance
        
        return 1.0

    def get_observation(self, track):
        obs = np.zeros(13)

        obs[0] = self.x / self.screen_width
        obs[1] = self.y / self.screen_height
        obs[2] = self.speed / 5.0
        obs[3] = math.cos(math.radians(self.angle))
        obs[4] = math.sin(math.radians(self.angle))
        obs[5] = track.get_curvature(self)
        obs[6] = self.cast_ray(track, 0, 50)
        obs[7] = self.cast_ray(track, -90, 50)
        obs[8] = self.cast_ray(track, 90, 50)
        obs[9] = self.cast_ray(track, -45, 50)
        obs[10] = self.cast_ray(track, 45, 50)
        obs[11] = track.get_progress(self) / len(track.x)
        obs[12] = 0

        return obs
