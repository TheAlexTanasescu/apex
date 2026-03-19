import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import asyncio
import websockets
import json


from sim.track import Track
from sim.car import Car
from training.evolve import NeuralNetwork, EvoAgent
import numpy as np

import pygame
import math

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800



async def main(weights_path):
    track = Track("Monza")
    track.load_track()
    track.transform(SCREEN_WIDTH, SCREEN_HEIGHT)
    
    # load weights
    weights = np.load(weights_path)
    
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24)
    
    async with websockets.connect("ws://localhost:8765") as websocket:
        # send join message with weights
        weights_path = sys.argv[1] if len(sys.argv) > 1 else "weights/evo_best.npy"

        if weights_path.endswith(".zip"):
            # PPO model
            await websocket.send(json.dumps({
                "type": "join",
                "weights_type": "ppo",
                "weights_path": weights_path
            }))
        else:
            # Evo model
            weights = np.load(weights_path)
            await websocket.send(json.dumps({
                "type": "join",
                "weights_type": "evo",
                "weights": weights.tolist()
            }))
        
        # wait for joined confirmation
        response = json.loads(await websocket.recv())
        my_id = response["id"]
        print(f"Connected as player {my_id}")
        
        # main loop
        running = True
        cars_state = []
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # receive state from server
            try:
                message = json.loads(await asyncio.wait_for(websocket.recv(), timeout=0.1))
                if message["type"] == "state":
                    cars_state = message["cars"]
            except asyncio.TimeoutError:
                pass
            
            # render
            screen.fill((0, 0, 0))
            track.draw(screen)
            
            for car_data in cars_state:
                color = (0, 200, 255) if car_data["id"] == my_id else (255, 50, 50)
                car_surface = pygame.Surface((10, 20), pygame.SRCALPHA)
                pygame.draw.rect(car_surface, color, (0, 0, 10, 20))
                rotated = pygame.transform.rotate(car_surface, -car_data["angle"])
                rect = rotated.get_rect(center=(int(car_data["x"]), int(car_data["y"])))
                screen.blit(rotated, rect)
            
            text = font.render(f"Players: {len(cars_state)}", True, (255, 255, 255))
            screen.blit(text, (20, 20))
            
            pygame.display.flip()
            clock.tick(60)

if __name__ == "__main__":
    weights_path = sys.argv[1] if len(sys.argv) > 1 else "weights/evo_best.npy"
    asyncio.run(main(weights_path))