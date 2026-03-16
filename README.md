# 🏎️ F1 Racing Sim

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame-2.5%2B-green)
![FastF1](https://img.shields.io/badge/FastF1-3.3%2B-red)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A competitive AI racing sim built on real F1 telemetry. Two players train agents independently, then race them head-to-head over a network.

> *(Add a demo GIF here once the sim is running)*

---

## ✨ Features

- Real F1 circuits from FastF1 (Monza, Silverstone, Monaco, and more)
- Drive yourself or watch AI agents race
- Train agents via **evolutionary algorithms** or **PPO reinforcement learning**
- Ray-cast sensors — agents "see" the track like a real autonomous vehicle
- Head-to-head multiplayer over a central server

---

## 🏗️ Project Structure

```
f1-racing-sim/
├── sim/
│   ├── track.py            # FastF1 loader, track geometry, ray casting
│   ├── car.py              # Car physics, observation vector, collisions
│   └── race.py             # Race loop, HUD, lap timing
├── agents/
│   ├── base_agent.py       # Interface all agents implement
│   ├── human_agent.py      # Keyboard-controlled
│   └── heuristic_agent.py  # Rule-based baseline to beat
├── training/
│   ├── evolve.py           # Neuroevolution training loop
│   ├── gym_env.py          # Gymnasium wrapper
│   └── train_rl.py         # PPO trainer
├── server/                 # Multiplayer server (WIP)
└── weights/                # Saved agent weight files
```

---

## ⚡ Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/f1-racing-sim
cd f1-racing-sim
pip install -r requirements.txt

# Drive the track yourself
python main.py --mode human --track monza

# Watch the rule-based agent (your first target to beat)
python main.py --mode heuristic --track monza
```

---

## 🧬 Training an Agent

### Step 1 — Evolutionary Algorithm (start here)

Breeds and mutates a population of neural net controllers over generations. No ML background needed to get started.

```bash
python -m training.evolve --track monza --gens 200 --pop 50
```

Best weights saved to `weights/evo_best.npy` automatically.

### Step 2 — Reinforcement Learning

Fine-tune further with PPO once you have an evolutionary baseline.

```bash
python -m training.train_rl --track monza --steps 500000
```

Saved to `weights/ppo_agent.zip`.

---

## 🏁 Multiplayer

Both players train on the same track independently, then connect to a shared server to race. Weight files can also be swapped directly for offline head-to-head.

```bash
# WIP — server setup instructions coming soon
```

---

## 🤖 Writing Your Own Agent

Implement one method:

```python
from agents.base_agent import BaseAgent
import numpy as np

class MyAgent(BaseAgent):
    def act(self, observation: np.ndarray, car_state, sim_time: float) -> dict:
        ray_ahead = observation[6]  # 1.0 = clear, 0.0 = wall right here
        if ray_ahead < 0.3:
            return self._action(throttle=0.0, brake=1.0, steer=0.0)
        return self._action(throttle=1.0, brake=0.0, steer=0.0)
```

---

## 📡 Observation Vector (13 values)

| Index | Meaning | Range |
|-------|---------|-------|
| 0–1 | X/Y position (normalized) | 0–1 |
| 2 | Speed | 0–1 |
| 3–4 | cos/sin of heading | −1–1 |
| 5 | Steering angle | −1–1 |
| 6–10 | Ray sensors (ahead, left, right, ahead-left, ahead-right) | 0–1 |
| 11 | Track progress | 0–1 |
| 12 | Current lap | int |

---

## 📦 Requirements

```
pygame>=2.5.0
numpy>=1.24.0
fastf1>=3.3.0
gymnasium>=0.29.0
stable-baselines3>=2.0.0
```

Python 3.10+. Works on Windows and macOS.

---

## 📄 License

MIT — use it, fork it, race it.

---

## 🙏 Credits

[FastF1](https://github.com/theOehrly/Fast-F1) · [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) · [Gymnasium](https://gymnasium.farama.org/)
