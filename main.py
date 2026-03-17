import pygame
import math
from sim.track import Track
from sim.car import Car
from agents.heuristic_agent import HeuristicAgent


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

    spawn_index = 100  # fallback
    for i in range(len(track.x) - 1):
        if track.outer_path.contains_point((track.x[i], track.y[i])) and \
        not track.inner_path.contains_point((track.x[i], track.y[i])):
            spawn_index = i
            break

    print(f"Spawning at index {spawn_index}, valid: {track.outer_path.contains_point((track.x[spawn_index], track.y[spawn_index]))}")

    dx = track.x[spawn_index + 1] - track.x[spawn_index]
    dy = track.y[spawn_index + 1] - track.y[spawn_index]
    car_angle = math.degrees(math.atan2(dy, dx))
    car = Car(track.x[spawn_index], track.y[spawn_index], 0, car_angle, SCREEN_WIDTH, SCREEN_HEIGHT)
    print(track.outer_path.contains_point((car.x, car.y)))

    agent = HeuristicAgent()


    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))  # black background
        track.draw(screen)
        text = font.render(f"Lap: {track.last_lap_time / 1000:.2f}s", True, (255, 255, 255))
        screen.blit(text, (20, 20))
       # obs = car.get_observation(track)
        #action = agent.act(obs, None, 0)
        #car.update(None, track, action)
        keys = pygame.key.get_pressed()
        car.update(keys, track)
        car.draw(screen)
        #print(car.get_observation(track))
        track.check_lap(car)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
