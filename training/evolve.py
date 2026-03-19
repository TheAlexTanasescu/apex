import math
import numpy as np
import pygame
import os


from agents.base_agent import BaseAgent
from sim.car import Car
from sim.track import Track

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
MAX_STEPS = 2000

class NeuralNetwork:
    def __init__(self):
        self.W1 = np.random.randn(32, 13) * 0.5
        self.b1 = np.zeros(32)
        self.W2 = np.random.randn(16, 32) * 0.5
        self.b2 = np.zeros(16)
        self.W3 = np.random.randn(3, 16) * 0.5
        self.b3 = np.zeros(3)

    def forward(self, observation):
        h1 = np.tanh(self.W1 @ observation + self.b1)
        h2 = np.tanh(self.W2 @ h1 + self.b2)
        output = np.tanh(self.W3 @ h2 + self.b3)
        return output

    def get_weights(self):
       return np.concatenate([self.W1.flatten(), self.b1, self.W2.flatten(), self.b2, self.W3.flatten(), self.b3])

    def set_weights(self, weights):
        self.W1 = weights[0:416].reshape(32, 13)
        self.b1 = weights[416:448]
        self.W2 = weights[448:960].reshape(16, 32)
        self.b2 = weights[960:976]
        self.W3 = weights[976:1024].reshape(3, 16)
        self.b3 = weights[1024:1027]

class EvoAgent(BaseAgent):

    def __init__(self, name, color):
        super().__init__(name, color)
        self.network = NeuralNetwork()

    def act(self, observation, car_state, sim_time):
        output = self.network.forward(observation)
        throttle = output[0]
        brake = output[1]
        steer = output[2]
        return {"throttle": float(throttle), "brake": float(brake), "steer": float(steer)}

   


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
            action = agent.act(obs, None, step)
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
                print(f"  step {step}, distance: {distance}")

        if lap_completed:
            print(f"Done. Fitness: {-lap_time}, steps survived: {step}")
            return -lap_time   # negative because lower is better, evolution maximizes
        else:
            #print(f"Done. Fitness: {best_distance}, steps survived: {step}")
            return best_distance - (wall_hits * 0.1)

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
       
        action = agent.act(obs, None, step)
        car.update(None, track, action)

        screen.fill((0, 0, 0))
        track.draw(screen)
        pygame.draw.circle(screen, (255, 255, 0), (int(track.x[279]), int(track.y[279])), 8)

        car.draw(screen)

        progress = track.get_progress(car)
        with open("log.txt", "a") as f:
            f.write(f"step:{step} progress:{progress} curvature:{obs[5]:.3f} speed:{car.speed:.2f}\n")
        text = font.render(f"Step: {step} | Progress: {progress}", True, (255, 255, 255))
        screen.blit(text, (20, 20))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()   

def evolve(track_name, generations, population_size):
        
        track = Track(track_name)
        track.load_track()
        track.transform(SCREEN_WIDTH, SCREEN_HEIGHT)


        #print("Starting evolution...")
        # create initial population
        population = [EvoAgent(name=f"Agent_{i}", color=(255, 255, 255)) for i in range(population_size)]

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
                child = EvoAgent(name="child", color=(255, 255, 255))
                weights = parent.network.get_weights()
                weights += np.random.randn(len(weights)) * 0.3 # mutate
                child.network.set_weights(weights)
                next_gen.append(child)
                best = population[0]
                np.save("weights/evo_best.npy", best.network.get_weights())
                print("Saved best agent to weights/evo_best.npy")

            population = next_gen

        # save best agent
        
        
        return best


if __name__ == "__main__":
    evolve("Monza", generations=30, population_size=50)