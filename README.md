# Obstacle Avoidance RL Drone

An advanced reinforcement learning (RL) project for training and evaluating a drone to autonomously navigate toward a goal while avoiding obstacles in a simulated 3D environment using PyBullet and Proximal Policy Optimization (PPO).

https://github.com/user-attachments/assets/1c755f87-170b-4b60-8146-144024dc33dd

https://github.com/user-attachments/assets/3bbcbe77-49a9-4fe8-b8e5-2f3bee86bbf1

![env1](https://github.com/user-attachments/assets/5026291e-9f56-4f82-bedf-81d7c4a6db40)

![env2](https://github.com/user-attachments/assets/ec6661b9-6fa3-4396-80af-486849425447)


## Features

- **PyBullet-based 3D Simulation**: Realistic drone physics and obstacle environments
- **Continuous Action Space**: Fine-grained drone control for smooth navigation
- **PPO Agent**: Robust RL algorithm for efficient training
- **Reward Shaping**: Custom rewards for goal-reaching, obstacle avoidance, and efficient movement
- **Tkinter GUI**: User-friendly interface for agent evaluation and visualization
- **Modular Codebase**: Easily extensible for new environments, agents, or reward functions
- **Checkpoints & Visualization**: Model saving and reward plotting during training

## Requirements

- Python 3.8+
- [PyBullet](https://pybullet.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [PyTorch](https://pytorch.org/)
- [Tkinter](https://wiki.python.org/moin/TkInter) (usually included with Python)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd obstacle_avoidance_rl_drone
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Agent

Train the PPO agent in the simulated environment:
```bash
python main.py train
```

### Evaluating the Agent

Evaluate a trained agent in the simulation:
```bash
python main.py
```

### GUI Evaluation

Launch the Tkinter GUI for interactive evaluation and visualization:
```bash
python -m src.gui
```

## Project Structure

```
obstacle_avoidance_rl_drone/
│
├── docs/                # Documentation and guides
├── src/                 # Source code (environment, agent, GUI, URDF, models)
│   ├── agent.py         # PPO agent implementation
│   ├── environment.py   # PyBullet drone environment
│   ├── gui.py           # Tkinter GUI for evaluation
│   ├── cf2.urdf         # Drone model (URDF)
│   └── ...
├── tests/               # Unit tests
├── main.py              # Entry point for training/evaluation
├── requirements.txt     # Python dependencies
├── setup.py             # Packaging script
├── LICENSE              # MIT License
└── README.md            # Project overview
```

## Core Components

- **`src/environment.py`**: Defines the PyBullet-based drone environment, obstacles, boundaries, and reward logic.
- **`src/agent.py`**: Implements the PPO agent with training, evaluation, and model checkpointing.
- **`src/gui.py`**: Tkinter-based GUI for running and visualizing agent evaluation.
- **`src/cf2.urdf`**: Custom quadcopter model for simulation.
- **`main.py`**: Command-line entry point for training and evaluation.

## Training & Evaluation Flow

1. **Environment Initialization**: Loads the drone, obstacles, and goal in PyBullet.
2. **Agent Training**: PPO agent interacts with the environment, learning to reach the goal while avoiding obstacles.
3. **Reward Calculation**: Custom rewards for progress, goal achievement, and collision avoidance.
4. **Checkpointing**: Model weights are saved periodically during training.
5. **Evaluation**: Trained agent is tested for success rate and average reward.
6. **GUI Visualization**: Optional Tkinter GUI for interactive evaluation and results display.

## Configuration

- **Environment & Agent Parameters**: Modify `src/environment.py` and `src/agent.py` for custom settings (e.g., goal position, reward shaping, PPO hyperparameters).
- **Model Paths**: Change model save/load paths in `main.py` or GUI as needed.

## Performance

- **Simulation Speed**: Adjustable via PyBullet settings
- **Training Duration**: Depends on hardware and max_timesteps
- **Evaluation Metrics**: Average reward and success rate per episode

## Testing

- **Unit Tests**: Add tests in the `tests/` directory for new features or bug fixes.
- **Headless Mode**: Run with `use_gui=False` for faster training without visualization.

## Documentation

- See `docs/user_guide.md` for detailed setup, training, and evaluation instructions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyBullet for physics simulation
- OpenAI Baselines for PPO inspiration
- Community contributors

## Pretrained Models

Pretrained PPO agent models are available for quick evaluation and benchmarking.

- **Download Link:** [Google Drive - PPO Agent Models](<your-gdrive-link-here>)

After downloading, place the model files (e.g., `ppo_agent.pth`, `ppo_agent_step_new2_load2600000_4.pth`) in the `src/` directory or specify the path when loading in your scripts or GUI.

---

**Note**: This project is for research and educational purposes. For questions or contributions, please open an issue or pull request on GitHub. 
