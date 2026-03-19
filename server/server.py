
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import asyncio
import websockets
import json
from sim.track import Track
from sim.car import Car
from training.evolve import NeuralNetwork, EvoAgent
from agents.ppo_agent import PPOAgent
import numpy as np

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60



track = None
cars = {}
connected_clients = {}

MIN_PLAYERS = 2
race_started = False

async def race_loop():
    global track, race_started
    track = Track("Monza")
    track.load_track()
    track.transform(SCREEN_WIDTH, SCREEN_HEIGHT)
    
    while True:
        ready_clients = {cid: data for cid, data in connected_clients.items() if data.get("car")}
        
        if len(ready_clients) >= MIN_PLAYERS and not race_started:
            race_started = True
            print("Race started!")
            for data in connected_clients.values():
                await data["websocket"].send(json.dumps({"type": "start"}))
        
        if race_started:
            for client_id, client_data in connected_clients.items():
                if client_data["agent"] and client_data["car"]:
                    obs = client_data["car"].get_observation(track)
                    action = client_data["agent"].act(obs, None, 0)
                    client_data["car"].update(None, track, action)
            
            if connected_clients:
                state = {
                    "type": "state",
                    "cars": [
                        {
                            "id": cid,
                            "x": float(data["car"].x),
                            "y": float(data["car"].y),
                            "angle": float(data["car"].angle),
                            "progress": int(track.get_progress(data["car"]))
                        }
                        for cid, data in connected_clients.items()
                        if data.get("car")
                    ]
                }
                message = json.dumps(state)
                for client_data in connected_clients.values():
                    try:
                        await client_data["websocket"].send(message)
                    except:
                        pass
        
        await asyncio.sleep(1/FPS)

async def handle_client(websocket):
    client_id = id(websocket)
    print(f"Client {client_id} connected")
    connected_clients[client_id] = {"websocket": websocket, "agent": None}
    
    try:
        async for message in websocket:
            data = json.loads(message)
            if data["type"] == "join":
                # load weights into an agent
            
                if data["weights_type"] == "evo":
                    weights = np.array(data["weights"])
                    agent = EvoAgent(name=f"Player_{client_id}", color=(255, 255, 255))
                    agent.network.set_weights(weights)
                else:
                    agent = PPOAgent(data["weights_path"])
                                
                # find spawn position
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
                offset = (len(connected_clients) - 1)*20
                car = Car(track.x[spawn_index] + offset * math.cos(math.radians(car_angle+90)), track.y[spawn_index] + offset * math.sin(math.radians(car_angle+90)), 0, car_angle, SCREEN_WIDTH, SCREEN_HEIGHT)
                
                connected_clients[client_id]["agent"] = agent
                connected_clients[client_id]["car"] = car
                print(f"Client {client_id} joined with agent")
                
                await websocket.send(json.dumps({"type": "joined", "id": client_id}))
    except websockets.exceptions.ConnectionClosed:
        print(f"Client {client_id} disconnected")
    finally:
        del connected_clients[client_id]

async def main():
    print("Starting server on ws://localhost:8765")
    async with websockets.serve(handle_client, "0.0.0.0", 8765):
        await race_loop()

if __name__ == "__main__":
    asyncio.run(main())