from training.gym_env import RacingEnv
from stable_baselines3 import PPO
import pygame
from sim.track import Track
from sim.car import Car
import math

def render_ppo(model_path, track_name):
    track = Track(track_name)
    track.load_track()
    track.transform(1200, 800)
    
    spawn_index = 0
    for i in range(len(track.x) - 1):
        dx = track.x[i] - track.x[0]
        dy = track.y[i] - track.y[0]
        dist = math.sqrt(dx**2 + dy**2)
        if dist > 50:
            spawn_index = i
            break
    
    dx = track.x[spawn_index + 1] - track.x[spawn_index]
    dy = track.y[spawn_index + 1] - track.y[spawn_index]
    car_angle = math.degrees(math.atan2(-dy, dx)) + 180
    car = Car(track.x[spawn_index], track.y[spawn_index], 0, car_angle, 1200, 800)
    
    model = PPO.load(model_path)
    env = RacingEnv(track_name)
    obs, _ = env.reset()
    
    pygame.init()
    screen = pygame.display.set_mode((1200, 800))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        screen.fill((0, 0, 0))
        track.draw(screen)
        env.car.draw(screen)
        
        progress = track.get_progress(env.car)
        text = font.render(f"Progress: {progress} | Reward: {reward:.2f}", True, (255, 255, 255))
        screen.blit(text, (20, 20))
        
        pygame.display.flip()
        clock.tick(60)
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    pygame.quit()

#render_ppo("weights/ppo_agent", "Monza")

race_env = RacingEnv("Monza")
model = PPO.load("weights/ppo_agent", env=race_env)
model.learn(total_timesteps=200000)
model.save("weights/ppo_agent")

#model = PPO("MlpPolicy", race_env, verbose=1)
#model.learn(total_timesteps=500000)
#model.save("weights/ppo_agent")