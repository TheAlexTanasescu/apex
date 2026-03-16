import pygame
import math
from sim.track import Track
from matplotlib.path import Path

class Car:
    def __init__(self, x, y, speed, angle):
        self.x = x
        self.y = y
        self.speed = 0
        self.angle = 0

    def draw(self, surface):
        car_surface = pygame.Surface((10, 20), pygame.SRCALPHA)
        pygame.draw.rect(car_surface, (255, 0, 0), (0, 0, 10, 20))
        rotated = pygame.transform.rotate(car_surface, -self.angle)
        rect = rotated.get_rect(center=(self.x, self.y))
        surface.blit(rotated, rect)

    def update(self, keys, track):
        if keys[pygame.K_UP]:
            self.speed += 0.1
        if keys[pygame.K_DOWN]:
            self.speed -= 0.1
        if keys[pygame.K_LEFT]:
            self.angle -= 1
        if keys[pygame.K_RIGHT]:
            self.angle += 1


        self.x -= self.speed * math.cos(math.radians(self.angle + 90))
        self.y -= self.speed * math.sin(math.radians(self.angle +90))   

        self.speed = min(self.speed, 3)
        self.speed = self.speed * 0.95

        self.check_and_clamp(track)

    def check_and_clamp(self, track):
        if self.check_collision(track):
            # find nearest point on outer wall
            min_dist = float('inf')
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
            
            # snap to just inside nearest wall point
            self.x = nearest_x
            self.y = nearest_y
            self.speed *= -0.3

    def check_collision(self, track):
        outer_points = list(zip(track.outer_x, track.outer_y))
        inner_points = list(zip(track.inner_x, track.inner_y))
        
        outer_path = Path(outer_points)
        inner_path = Path(inner_points)
        
        car_pos = (self.x, self.y)
        
        inside_outer = outer_path.contains_point(car_pos)
        inside_inner = inner_path.contains_point(car_pos)
        
        # valid track position = inside outer but outside inner
        if not inside_outer or inside_inner:
            return True
        
        return False
        """ for i in range(len(track.outer_x)):
            dx = self.x - track.outer_x[i]
            dy = self.y - track.outer_y[i]
            distance = math.sqrt(dx**2 + dy**2)
            if distance < 10:
                return True
        
        for i in range(len(track.inner_x)):
            dx = self.x - track.inner_x[i]
            dy = self.y - track.inner_y[i]
            distance = math.sqrt(dx**2 + dy**2)
            if distance < 10:
                return True

        return False """

        
        
