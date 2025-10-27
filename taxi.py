"""
Dans ce TP, nous allons implémenter un agent qui apprend à jouer au jeu Taxi-v3
de OpenAI Gym. Le but du jeu est de déposer un passager à une destination
spécifique en un minimum de temps. Le jeu est composé d'une grille de 5x5 cases
et le taxi peut se déplacer dans les 4 directions (haut, bas, gauche, droite).
Le taxi peut prendre un passager sur une case spécifique et le déposer à une
destination spécifique. Le jeu est terminé lorsque le passager est déposé à la
destination. Le jeu est aussi terminé si le taxi prend plus de 200 actions.

Vous devez implémenter un agent qui apprend à jouer à ce jeu en utilisant
les algorithmes Q-Learning et SARSA.

Pour chaque algorithme, vous devez réaliser une vidéo pour montrer que votre modèle fonctionne.
Vous devez aussi comparer l'efficacité des deux algorithmes en termes de temps
d'apprentissage et de performance.

A la fin, vous devez rendre un rapport qui explique vos choix d'implémentation
et vos résultats (max 1 page).
"""

import typing as t
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from qlearning import QLearningAgent
from qlearning_eps_scheduling import QLearningAgentEpsScheduling
from sarsa import SarsaAgent


env = gym.make("Taxi-v3", render_mode="rgb_array")
n_actions = env.action_space.n


#################################################
# 1. Play with QLearningAgent
#################################################

agent = QLearningAgent(
    learning_rate=0.5, epsilon=0.1, gamma=0.99, legal_actions=list(range(n_actions))
)


def play_and_train(env: gym.Env, agent: QLearningAgent, t_max=int(1e4)) -> float:
    """
    This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total rewardb
    """
    total_reward: t.SupportsFloat = 0.0
    s, _ = env.reset()

    for _ in range(t_max):
        a = agent.get_action(s)
        next_s, r, done, truncated, _ = env.step(a)

        # Train agent for state s
        # BEGIN SOLUTION
        agent.update(s, a, r, next_s)
        total_reward += r
        s = next_s
        if done or truncated:
            break
        # END SOLUTION

    return total_reward


def play_and_train_sarsa(env: gym.Env, agent, t_max=int(1e4)) -> float:
    """
    SARSA-specific training function.
    """
    total_reward: t.SupportsFloat = 0.0
    s, _ = env.reset()
    a = agent.get_action(s)
    
    for _ in range(t_max):
        next_s, r, done, truncated, _ = env.step(a)
        next_a = agent.get_action(next_s)
        agent.update_sarsa(s, a, r, next_s, next_a)
        total_reward += r
        s = next_s
        a = next_a
        if done or truncated:
            break
    
    return total_reward


def record_video_gym(env_name, agent, filename, n_episodes=3, use_sarsa=False):
    """
    Enregistre une vidéo en utilisant gym.wrappers.RecordVideo
    """
    print(f"Recording video: {filename}")
    video_env = gym.make(env_name, render_mode="rgb_array")
    video_env = gym.wrappers.RecordVideo(
        video_env, 
        video_folder="videos",
        name_prefix=filename.replace(".mp4", ""),
        episode_trigger=lambda x: True
    )
    
    for episode in range(n_episodes):
        s, _ = video_env.reset()
        done = False
        
        if use_sarsa:
            a = agent.get_best_action(s)
        
        while not done:
            if use_sarsa:
                next_s, r, done, truncated, _ = video_env.step(a)
                a = agent.get_best_action(next_s)
                s = next_s
                done = done or truncated
            else:
                a = agent.get_best_action(s)
                next_s, r, done, truncated, _ = video_env.step(a)
                s = next_s
                done = done or truncated
    
    video_env.close()
    print(f"Video saved in videos/ folder")


def plot_learning_curves(rewards_dict, filename="training_comparison.png"):
    """
    Trace les courbes d'apprentissage
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    colors = ['blue', 'green', 'red']
    
    for idx, (name, rewards) in enumerate(rewards_dict.items()):
        moving_avg = [np.mean(rewards[max(0, i-100):i+1]) for i in range(len(rewards))]
        ax1.plot(moving_avg, color=colors[idx], linewidth=2, label=name)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward (avg 100)')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    final_means = []
    final_stds = []
    names = []
    
    for name, rewards in rewards_dict.items():
        final_100 = rewards[-100:]
        final_means.append(np.mean(final_100))
        final_stds.append(np.std(final_100))
        names.append(name)
    
    x_pos = np.arange(len(names))
    ax2.bar(x_pos, final_means, yerr=final_stds, alpha=0.7, 
            color=colors[:len(names)], capsize=10)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=15, ha='right')
    ax2.set_ylabel('Mean Reward (last 100 episodes)')
    ax2.set_title('Final Performance')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    for i, (mean, std) in enumerate(zip(final_means, final_stds)):
        ax2.text(i, mean + std + 0.5, f'{mean:.2f}±{std:.2f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved: {filename}")


print("="*70)
print("TRAINING: Q-Learning Standard")
print("="*70)

rewards = []
for i in range(1000):
    rewards.append(play_and_train(env, agent))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))

assert np.mean(rewards[-100:]) > 0.0
rewards_qlearning = rewards
# TODO: créer des vidéos de l'agent en action
record_video_gym("Taxi-v3", agent, "qlearning.mp4", n_episodes=3)

#################################################
# 2. Play with QLearningAgentEpsScheduling
#################################################

print("\n" + "="*70)
print("TRAINING: Q-Learning with Epsilon Scheduling")
print("="*70)

agent = QLearningAgentEpsScheduling(
    learning_rate=0.5, epsilon=0.25, gamma=0.99, legal_actions=list(range(n_actions))
)
agent.reset()

rewards = []
for i in range(1000):
    rewards.append(play_and_train(env, agent))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))

assert np.mean(rewards[-100:]) > 0.0
rewards_qlearning_scheduling = rewards
# TODO: créer des vidéos de l'agent en action
record_video_gym("Taxi-v3", agent, "qlearning_scheduling.mp4", n_episodes=3)

####################
# 3. Play with SARSA
####################

print("\n" + "="*70)
print("TRAINING: SARSA")
print("="*70)

agent = SarsaAgent(learning_rate=0.5, gamma=0.99, legal_actions=list(range(n_actions)), epsilon=0.05)

rewards = []
for i in range(1000):
    rewards.append(play_and_train_sarsa(env, agent))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))

rewards_sarsa = rewards
record_video_gym("Taxi-v3", agent, "sarsa.mp4", n_episodes=3, use_sarsa=True)

# Comparaison
print("\n" + "="*70)
print("COMPARISON")
print("="*70)

rewards_dict = {
    'Q-Learning (ε=0.1)': rewards_qlearning,
    'Q-Learning Scheduling': rewards_qlearning_scheduling,
    'SARSA (ε=0.05)': rewards_sarsa
}

plot_learning_curves(rewards_dict, "training_comparison.png")

for name, rewards in rewards_dict.items():
    final_mean = np.mean(rewards[-100:])
    final_std = np.std(rewards[-100:])
    print(f"{name}: {final_mean:.2f} ± {final_std:.2f}")

print("\n" + "="*70)
print("DONE! Check videos/ folder for recordings")
print("="*70)