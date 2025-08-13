from boolean_env_mlp import BooleanSimplificationEnv
from agent_mlp import DQNAgent
import numpy as np
import torch
from collections import deque
import time
import matplotlib.pyplot as plt

EPISODES = 2000
MAX_STEPS_PER_EPISODE = 200
SOLVE_SCORE = 60.0

MAX_EXPRESSION_DEPTH = 7
MAX_LITERALS = 7

AGENT_SEED = 0
HIDDEN_SIZE = 64
LEARNING_RATE = 5e-4
GAMMA = 0.99
TAU = 1e-3
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
UPDATE_EVERY = 4

def train_DQN_MLP():
    env = BooleanSimplificationEnv(max_expression_depth=MAX_EXPRESSION_DEPTH,
                                   max_literals=MAX_LITERALS,
                                   max_steps=MAX_STEPS_PER_EPISODE)

    agent = DQNAgent(
        state_size=env.get_state_size(),
        action_size=env.get_action_size(),
        seed=AGENT_SEED,
        hidden_size=HIDDEN_SIZE,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        tau=TAU,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        update_every=UPDATE_EVERY
    )

    scores = deque(maxlen=100)
    all_scores = []
    start_time = time.time()

    for episode in range(1, EPISODES + 1):
        state = env.reset()
        initial_expr = env.current_expression
        episode_reward = 0.0
        for step in range(MAX_STEPS_PER_EPISODE):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            if done:
                break

        scores.append(episode_reward)
        all_scores.append(episode_reward)
        avg_score = np.mean(scores)

        COLOR_LAVENDER = "\u001b[95m"
        COLOR_BLUE = "\u001b[94m"
        COLOR_GREEN = "\u001b[92m"
        COLOR_RESET = "\u001b[0m"

        print(f"{COLOR_LAVENDER}--- Episode {episode}/{EPISODES} ---{COLOR_RESET}")
        print(f"Initial Expression: {COLOR_BLUE}{initial_expr}{COLOR_RESET}")
        print(f"Final Expression:   {COLOR_GREEN}{env.current_expression}{COLOR_RESET}")
        print(f"Episode Reward:     {episode_reward:.2f}")
        print(f"Average Score:      {avg_score:.2f}")
        print(f"Epsilon:            {agent.epsilon:.2f}")
        print(f"Initial Complexity: {env.initial_complexity}")
        print(f"Final Complexity:   {env._get_complexity(env.current_expression)}")
        print("-" * (23 + len(str(episode)) + len(str(EPISODES))))

        if episode >= 100 and avg_score >= SOLVE_SCORE:
            print(f"Environment solved in {episode} episodes!\tAverage Score: {avg_score:.2f}")
            torch.save(agent.qnet_policy.state_dict(), 'checkpoint_mlp.pth')
            break

    torch.save(agent.qnet_policy.state_dict(), 'checkpoint_mlp.pth')

    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")
    plot_scores(all_scores)


def plot_scores(scores):
    avg_scores = [np.mean(scores[max(0, i - 100):i + 1]) for i in range(len(scores))]

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(scores, label='Episode Score', color='#3498db', alpha=0.6)
    ax.plot(avg_scores, label='100-Episode Average', color='#e74c3c', linewidth=2)

    ax.set_title('Training Progress (MLP)', fontsize=18, fontweight='bold')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)

    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(fontsize=12)

    plt.savefig('training_plot_mlp.png')
    plt.show()


if __name__ == "__main__":
    train_DQN_MLP()
