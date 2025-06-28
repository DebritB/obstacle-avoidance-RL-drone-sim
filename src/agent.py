import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # Log standard deviation for continuous actions

    def forward(self, x):
        value = self.critic(x)
        mean = self.actor(x)
        std = torch.exp(self.log_std)  # Ensure std is positive
        return mean, std, value

class PPOAgent:
    def __init__(self, env, gamma=0.99, lr=1e-4, clip_epsilon=0.1, epochs=10, batch_size=64):
        self.env = env
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        obs_dim = env.get_observation().shape[0]
        action_dim = 3  
        self.episode_rewards_history=[]
        self.timesteps_history=[]

        # Initialize actor-critic model
        self.model = ActorCritic(input_dim=obs_dim, action_dim=action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, std, _ = self.model(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        return action.squeeze().detach().numpy(), log_prob.item()

    def compute_advantages(self, rewards, values, dones, lam=0.95):
        returns = []
        G = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * (1 - done)
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)

        # GAE
        advantages = []
        advantage = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + (self.gamma * values[i + 1] * (1 - dones[i])) - values[i]
            advantage = delta + (self.gamma * lam * advantage * (1 - dones[i]))
            advantages.insert(0, advantage)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        return returns, advantages


    def update_policy(self, states, actions, log_probs, returns, advantages):
        # Convert lists to NumPy arrays for indexing with batch_indices
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        states = np.array(states)
        actions = np.array(actions)
        log_probs = np.array(log_probs)
        returns = np.array(returns)
        advantages = np.array(advantages)

        for _ in range(self.epochs):
            indices = np.random.permutation(len(states))
            for i in range(0, len(states), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                
                # Select mini-batch using batch_indices
                batch_states = torch.FloatTensor(states[batch_indices])
                batch_actions = torch.FloatTensor(actions[batch_indices])
                batch_log_probs = torch.FloatTensor(log_probs[batch_indices])
                batch_returns = torch.FloatTensor(returns[batch_indices])
                batch_advantages = torch.FloatTensor(advantages[batch_indices])


                # Calculate new log probabilities
                mean, std, values = self.model(batch_states)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=1)
                entropy = dist.entropy().mean()

                # Calculate ratios
                ratios = torch.exp(new_log_probs - batch_log_probs)

                # Clipped objective
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss
                critic_loss = nn.MSELoss()(values.view(-1), batch_returns.view(-1))

                # Combined loss
                loss = actor_loss + 0.5 * critic_loss - 0.005 * entropy

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


    def train(self, max_timesteps=10000):
        state = self.env.reset()
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []


        step_count = 0
        episode_reward = 0
        for t in range(max_timesteps):
            step_count += 1
            action, log_prob = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action, state[-1])  # Use last distance to goal

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            _, _, value = self.model(torch.FloatTensor(state).unsqueeze(0))
            values.append(value.item())

            state = next_state
            episode_reward += reward



            if done or (t+1)%5000==0:
                _, _, next_value = self.model(torch.FloatTensor(next_state).unsqueeze(0))
                values.append(next_value.item())

                returns, advantages = self.compute_advantages(rewards, values, dones)
                self.update_policy(states, actions, log_probs, returns, advantages)

                self.episode_rewards_history.append(episode_reward)
                self.timesteps_history.append(t)

                print(f"Episode Reward: {episode_reward}")
                state, episode_reward = self.env.reset(), 0
                print(f'steps:{step_count},is_done:{done}')
                states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

            if (t + 1) % (max_timesteps//50) == 0:
                checkpoint_path = f"ppo_agent_step_new2_load5_{t + 1}_4.pth"
                self.save_model(filepath=checkpoint_path)
                print(f"Checkpoint saved at step {t + 1} to {checkpoint_path}")

        self.plot_rewards(self.timesteps_history, list(np.log2(self.episode_rewards_history)-10*np.log2(10)))
    

    def evaluate(self, episodes=5, max_timesteps=1000):
        
        self.env = ContinuousActionDroneEnv(use_gui=True)  # Reinitialize environment in GUI mode
        episode_rewards = []
        success_count=0

        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            step_count = 0

            while not done and step_count < max_timesteps:
                action, _ = self.select_action(state)  # Select action using trained policy
                next_state, reward, done, _ = self.env.step(action, state[-1])
                
                episode_reward += reward
                state = next_state
                step_count += 1
                
                # Optional: Slow down the simulation for better visualization
                time.sleep(1./30.)  # Adjust the sleep time as needed

            if episode_reward>30000000000:
                success_count+=1
            episode_rewards.append(episode_reward)
            print(f"Episode {episode + 1} Reward: {episode_reward} | Steps: {step_count}")
        success_rate=round(success_count/episodes,4)
        # Plotting rewards over evaluation episodes
        plt.figure(figsize=(10, 5))
        plt.plot(episode_rewards, label="Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Evaluation Rewards Over Episodes")
        plt.legend()
        plt.grid(True)
        plt.show()
        print("Evaluation completed.")
        return episode_rewards, success_rate

    def save_model(self, filepath="ppo_agent.pth"):
        """
        Save the model parameters.
        """
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath="ppo_agent.pth"):
        """
        Load the model parameters.
        """
        if os.path.exists(filepath):
            self.model.load_state_dict(torch.load(filepath))
            print(f"Model loaded from {filepath}")
        else:
            print(f"No model found at {filepath}. Starting from scratch.")

    def plot_rewards(self, timesteps, rewards):
        plt.figure(figsize=(10, 5))
        plt.plot(timesteps, rewards, label="Total Reward per Episode")
        plt.xlabel("Timesteps")
        plt.ylabel("Total Reward")
        plt.title("Reward per Episode Over Timesteps")
        plt.legend()
        plt.grid(True)
        plt.show()