import pygame
import math
from sim.track import Track
from sim.car import Car

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800

def main():

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    font = pygame.font.SysFont("Arial", 24)

    pygame.display.set_caption("Apex")

    

    track = Track("Monza", False, 0, 0)
    track.load_track()
    track.transform(SCREEN_WIDTH, SCREEN_HEIGHT)

    dx = track.x[1] - track.x[0]
    dy = track.y[1] - track.y[0]
    car_angle = math.degrees(math.atan2(dy, dx))


    car = Car(track.x[0],track.y[0], 0, car_angle )

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))  # black background
        track.draw(screen)
        text = font.render(f"Lap: {track.last_lap_time / 1000:.2f}s", True, (255, 255, 255))
        screen.blit(text, (20, 20))
        keys = pygame.key.get_pressed()
        car.update(keys, track)
        car.draw(screen)
        track.check_lap(car)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
