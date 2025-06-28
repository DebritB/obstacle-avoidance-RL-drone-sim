import sys
from src.environment import ContinuousActionDroneEnv
from src.agent import PPOAgent

def main():
    env = ContinuousActionDroneEnv(use_gui=False)
    agent = PPOAgent(env)
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        agent.train(max_timesteps=10000000)
        agent.save_model('ppo_agent.pth')
    else:
        agent.load_model('ppo_agent.pth')
        agent.evaluate(episodes=5)

if __name__ == '__main__':
    main() 