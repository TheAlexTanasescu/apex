import math
import os
import pygame
import numpy as np
from agents.base_agent import BaseAgent
from sim.car import Car
from sim.track import Track

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
MAX_STEPS = 2000

class NeuralNetwork:
    def __init__(self):
        self.W1 = np.random.randn(13, 13) * 0.1
        self.W2 = np.random.randn(26, 13) * 0.1
        self.W3 = np.random.randn(26, 26) * 0.1
        self.W4 = np.random.randn(52, 26) * 0.1
        self.W5 = np.random.randn(26, 52) * 0.1
        self.W6 = np.random.randn(26, 26) * 0.1
        self.W7 = np.random.randn(13, 26) * 0.1
        self.W8 = np.random.randn(3, 13) * 0.1
        self.b1 = np.zeros(13)
        self.b2 = np.zeros(26)
        self.b3 = np.zeros(26)
        self.b4 = np.zeros(52)
        self.b5 = np.zeros(26)
        self.b6 = np.zeros(26)
        self.b7 = np.zeros(13)
        self.b8 = np.zeros(3)

        self.weights = [self.W1, self.W2, self.W3, self.W4, self.W5, self.W6, self.W7, self.W8]
        self.biases = [self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.b7, self.b8]

    def basicLayer(self, weight, input, bias):
        result = np.tanh(weight @ input + bias)
        return result

    def output(self, input):
        output = np.tanh(self.W8 @ input + self.b8)
        return output

    def forward(self, observation):
        h1 = self.basicLayer(self.W1, observation, self.b1)
        h2 = self.basicLayer(self.W2, h1, self.b2)
        h3 = self.basicLayer(self.W3, h2, self.b3)
        h4 = self.basicLayer(self.W4, h3, self.b4)
        h5 = self.basicLayer(self.W5, h4, self.b5)
        h6 = self.basicLayer(self.W6, h5, self.b6)
        h7 = self.basicLayer(self.W7, h6, self.b7)
        output = self.output(h7)
        return output
    
    def get_weights(self):
        return np.concatenate([w.flatten() for w in self.weights] + [b.flatten() for b in self.biases])

    def set_weights(self, flat_weights):
        idx = 0
        for w in self.weights:
            size = w.size
            w[:] = flat_weights[idx:idx+size].reshape(w.shape)
            idx += size
        for b in self.biases:
            size = b.size
            b[:] = flat_weights[idx:idx+size].reshape(b.shape)
            idx += size

class SherriffAgent(BaseAgent):
    def __init__(self, name, color):
        super().__init__(name, color)
        self.network = NeuralNetwork()
    
    def sim(self, observation, car_state, sim_time):
        output = self.network.forward(observation)
        throttle = output[0]
        brake = output[1]
        steer = output[2]
        return {"throttle": float(throttle), "brake": float(brake), "steer": float(steer)}
    
    

    # def backward():
        
    
    # def update():

        
    # def train():

def render_agent(agent, track):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Best Agent")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24)
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
    car = Car(track.x[spawn_index], track.y[spawn_index], 0, car_angle, SCREEN_WIDTH, SCREEN_HEIGHT)

    for step in range(MAX_STEPS):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        obs = car.get_observation(track)
       
        action = agent.sim(obs, None, step)
        car.update(None, track, action)

        screen.fill((0, 0, 0))
        track.draw(screen)
        car.draw(screen)

        progress = track.get_progress(car)
        with open("log.txt", "a") as f:
            f.write(f"step:{step} progress:{progress} curvature:{obs[5]:.3f} speed:{car.speed:.2f}\n")
        text = font.render(f"Step: {step} | Progress: {progress}", True, (255, 255, 255))
        screen.blit(text, (20, 20))

        pygame.display.flip()
        clock.tick(60)
    pygame.quit()   

def evaluate_agent(agent, track):
        print("Evaluating agent...")
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
        car = Car(track.x[spawn_index], track.y[spawn_index], 0, car_angle, SCREEN_WIDTH, SCREEN_HEIGHT)

        spawn_progress = track.get_progress(car)
        lap_time = 0
        best_distance = 0
        wall_hits = 0
        lap_completed = False
        last_progress = spawn_progress
        last_best_step = 0

        for step in range(MAX_STEPS):
            obs = car.get_observation(track)
            action = agent.sim(obs, None, step)
            car.update(None, track, action)
            
            if car.check_collision(track):
                wall_hits += 1

            if car.speed < 0:
                wall_hits += 10

            distance = track.get_progress(car)
            
            if distance > last_progress and distance < last_progress + 20:
                best_distance = distance
                last_progress = distance
                last_best_step = step

            elif distance < last_progress - 30:
                last_progress = distance
            if step - last_best_step > 50:
                break
            if step > 100 and distance < spawn_progress + 10 and best_distance > len(track.x) * 0.9:
                lap_completed = True
                break
            
            if step % 100 == 0:
                print(f"  step {step}, distance: {distance}, Fitness: {best_distance - (wall_hits)}")

        if lap_completed:
            print(f"Done. Fitness: {-lap_time}, steps survived: {step}")
            return -lap_time   # negative because lower is better, evolution maximizes
        else:
            #print(f"Done. Fitness: {best_distance}, steps survived: {step}")
            return best_distance - (wall_hits)

def sevolve(track_name, generations, population_size):
        track = Track(track_name, SCREEN_WIDTH, SCREEN_HEIGHT)

        #print("Starting evolution...")
        # create initial population
        population = [SherriffAgent(name=f"Agent_{i}", color=(255, 255, 255)) for i in range(population_size)]

        if os.path.exists("weights/evo_best.npy"):
            print("Loading previous best weights...")
            best_weights = np.load("weights/evo_best.npy")
            population[0].network.set_weights(best_weights)

        with open("log.txt", "w") as f:
            f.write("=== New Evolution Run ===\n")
        for gen in range(generations):
            # evaluate all agents
            fitness_scores = [evaluate_agent(agent, track) for agent in population]

            # sort by fitness descending
            ranked = sorted(zip(fitness_scores, population), key=lambda x: x[0], reverse=True)
            fitness_scores, population = zip(*ranked)
            population = list(population)

            best_fitness = fitness_scores[0]
            print(f"Generation {gen} | Best fitness: {best_fitness:.2f}")
            with open("log.txt", "a") as f:
                f.write(f"Generation {gen} | Best fitness: {best_fitness:.2f}\n")
            render_agent(population[0], track)
            
            elite_count = 3
            elites = population[:elite_count]

            # breed next generation
            next_gen = list(elites)
            while len(next_gen) < population_size:
                parent = elites[np.random.randint(0, len(elites))]
                child = SherriffAgent(name="child", color=(255, 255, 255))
                weights = parent.network.get_weights()
                weights += np.random.randn(len(weights)) * 0.3 # mutate
                child.network.set_weights(weights)
                next_gen.append(child)
                best = population[0]
                np.save("weights/evo_best.npy", best.network.get_weights())
                print("Saved best agent to weights/evo_best.npy")

            population = next_gen
        return best

if __name__ == "__main__":
    sevolve("Monza", generations=10, population_size=10)