# User Guide

## Setup

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd obstacle_avoidance_rl_drone
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training

To train the PPO agent:
```bash
python main.py train
```

## Evaluation

To evaluate the trained agent:
```bash
python main.py
```

## GUI Evaluation

To use the Tkinter GUI for evaluation, run:
```bash
python -m src.gui
``` 