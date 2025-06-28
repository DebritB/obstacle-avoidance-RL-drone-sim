# ... existing code ...
# Paste the DroneGUI class and main block from GUI.py here
# ... existing code ... 

import tkinter as tk
from tkinter import messagebox, Toplevel
import pybullet as p
from src.environment import ContinuousActionDroneEnv as Env1
from src.agent import PPOAgent as PPOAgent1
# If you want to support Env2/PPOAgent2, import or implement them as needed

class DroneGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Drone Evaluation GUI")
        self.episodes_var = tk.IntVar(value=10)
        self.env_var = tk.StringVar(value="Environment 1")
        self.avg_rewards = []
        self.success_rates = []
        self.create_widgets()
    def create_widgets(self):
        font_style = ("Arial", 12)
        self.main_title = tk.Label(self.root, text="Main Configuration Interface", font=("Arial", 12, "bold"))
        self.main_title.grid(row=0, column=0, columnspan=2, padx=10, pady=15)
        tk.Label(self.root, text="Number of Episodes:", font=font_style).grid(row=1, column=0, padx=15, pady=15)
        self.episodes_entry = tk.Entry(self.root, textvariable=self.episodes_var, font=font_style, width=15)
        self.episodes_entry.grid(row=1, column=1, padx=15, pady=15)
        tk.Label(self.root, text="Select Environment:", font=font_style).grid(row=2, column=0, padx=15, pady=15)
        self.env_menu = tk.OptionMenu(self.root, self.env_var, "Environment 1")
        self.env_menu.config(font=font_style, width=12)
        self.env_menu.grid(row=2, column=1, padx=15, pady=15)
        self.start_button = tk.Button(self.root, text="Start Evaluation", font=font_style, command=self.start_evaluation)
        self.start_button.grid(row=3, column=0, columnspan=2, padx=15, pady=15)
        self.exit_button = tk.Button(self.root, text="Exit", font=font_style, command=self.exit_application)
        self.exit_button.grid(row=4, column=0, columnspan=2, padx=15, pady=15)
    def start_evaluation(self):
        num_episodes = self.episodes_var.get()
        if num_episodes <= 0:
            messagebox.showerror("Invalid Input", "Number of episodes must be positive.")
            return
        self.avg_rewards, self.success_rates = self.run_evaluation(num_episodes)
        self.root.withdraw()
        self.show_results_window()
    def run_evaluation(self, episodes):
        avg_rewards = []
        success_rates = []
        env_class = Env1
        agent_class = PPOAgent1
        model_path = r"ppo_agent_step_new2_load2600000_4.pth"
        for _ in range(1):
            env = env_class(use_gui=False)
            env.reset()
            agent = agent_class(env)
            agent.load_model(model_path)
            p.disconnect()
            episode_rewards, success_rate = agent.evaluate(episodes=episodes)
            avg_episode_reward = sum(episode_rewards) / len(episode_rewards)
            avg_rewards.append(avg_episode_reward)
            success_rates.append(success_rate)
        return avg_rewards, success_rates
    def show_results_window(self):
        result_window = Toplevel(self.root)
        result_window.title("Evaluation Results")
        font_style = ("Arial", 12)
        avg_reward = sum(self.avg_rewards) / len(self.avg_rewards)
        avg_success_rate = sum(self.success_rates) / len(self.success_rates)
        tk.Label(result_window, text=f"Average Reward: {avg_reward:.2f}", font=font_style).pack(pady=10)
        tk.Label(result_window, text=f"Success Rate: {avg_success_rate:.2%}", font=font_style).pack(pady=10)
        back_button = tk.Button(result_window, text="Back", font=font_style, command=lambda: self.close_results_window(result_window))
        back_button.pack(pady=10)
    def close_results_window(self, window):
        window.destroy()
        self.root.deiconify()
    def exit_application(self):
        self.root.quit()
        self.root.destroy()

        

if __name__ == "__main__":
    root = tk.Tk()
    app = DroneGUI(root)
    root.mainloop() 