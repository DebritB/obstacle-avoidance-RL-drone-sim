# Obstacle Avoidance RL Drone

A reinforcement learning project for training and evaluating a drone to navigate toward a goal while avoiding obstacles using PyBullet and PPO.

## Project Structure

```
obstacle_avoidance_rl_drone/
│
├── .github/workflows/         # CI/CD workflows
├── docs/                     # Documentation
├── src/                      # Source code (environment, agent, GUI, URDF, models)
├── tests/                    # Unit tests
├── main.py                   # Entry point for training/evaluation
├── requirements.txt          # Python dependencies
├── setup.py                  # Packaging script
├── README.md                 # Project overview
```

## Quick Start

```bash
pip install -r requirements.txt
python main.py train   # To train the agent
python main.py         # To evaluate the agent
```

## Features
- PyBullet-based drone simulation
- PPO reinforcement learning agent
- Tkinter GUI for evaluation
- Modular and extensible codebase 