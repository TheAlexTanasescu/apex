import pygame
from sim.track import Track

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800

def main():
    



    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Apex")

    track = Track("Monza")
    track.load_track()
    track.transform(SCREEN_WIDTH, SCREEN_HEIGHT)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))  # black background
        track.draw(screen)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
