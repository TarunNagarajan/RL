import torch
from boolean_env_mlp import BooleanSimplificationEnv
from agent_mlp import DQNAgent
import numpy as np
import time

# --- Configuration from main_mlp.py ---
MAX_EXPRESSION_DEPTH = 7
MAX_LITERALS = 7
MAX_STEPS_PER_EPISODE = 200

AGENT_SEED = 0
HIDDEN_SIZE = 64
LEARNING_RATE = 5e-4
GAMMA = 0.99
TAU = 1e-3
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
UPDATE_EVERY = 4
# --- End Configuration ---

def test_DQN_MLP(num_test_episodes=10):
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

    # Load the trained policy network weights
    try:
        agent.qnet_policy.load_state_dict(torch.load('checkpoint_mlp.pth', weights_only = True))
        agent.qnet_policy.eval() # Set to evaluation mode
        print(f"{COLOR_GREEN}Successfully loaded checkpoint_mlp.pth{COLOR_RESET}")
    except FileNotFoundError:
        print(f"{COLOR_RED}Error: checkpoint_mlp.pth not found. Please ensure the training script has been run and the file exists.{COLOR_RESET}")
        return
    except Exception as e:
        print(f"{COLOR_RED}Error loading model: {e}{COLOR_RESET}")
        return

    print(f"\n{COLOR_LAVENDER}--- Starting MLP Model Testing ---{COLOR_RESET}")

    optimal_count = 0
    for episode in range(1, num_test_episodes + 1):
        state = env.reset()
        initial_expr = env.current_expression
        initial_complexity = env._get_complexity(initial_expr)
        known_best_complexity = env.known_best_complexity
        
        episode_reward = 0.0
        done = False
        steps_taken = 0

        print(f"\n{COLOR_LAVENDER}--- Test Episode {episode}/{num_test_episodes} ---{COLOR_RESET}")
        print(f"Initial Expression: {COLOR_BLUE}{initial_expr}{COLOR_RESET}")
        print(f"Initial Complexity: {initial_complexity}")
        print(f"Known Best Complexity: {known_best_complexity}")

        while not done and steps_taken < MAX_STEPS_PER_EPISODE:
            action = agent.act(state) # Use the policy network to act
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward
            steps_taken += 1
            # Optional: print intermediate steps
            # print(f"  Step {steps_taken}: Applied rule {action}, New Complexity: {env._get_complexity(env.current_expression)}")

        final_expr = env.current_expression
        final_complexity = env._get_complexity(final_expr)

        print(f"Final Expression:   {COLOR_GREEN}{final_expr}{COLOR_RESET}")
        print(f"Final Complexity:   {final_complexity}")
        print(f"Episode Reward:     {episode_reward:.2f}")
        print(f"Steps Taken:        {steps_taken}")
        if final_complexity <= known_best_complexity:
            print(f"Result:             {COLOR_GREEN}Simplified to optimal or better!{COLOR_RESET}")
            optimal_count += 1
        else:
            print(f"Result:             {COLOR_YELLOW}Did not reach optimal complexity.{COLOR_RESET}")

    accuracy = (optimal_count / num_test_episodes) * 100 if num_test_episodes > 0 else 0
    print(f"\n{COLOR_LAVENDER}--- Test Summary ---{COLOR_RESET}")
    print(f"Total Episodes:     {num_test_episodes}")
    print(f"Successful Simplifications: {optimal_count}")
    print(f"Accuracy:           {accuracy:.2f}%{COLOR_RESET}")

COLOR_LAVENDER = "\u001b[95m"
COLOR_BLUE = "\u001b[94m"
COLOR_GREEN = "\u001b[92m"
COLOR_YELLOW = "\u001b[93m"
COLOR_RED = "\u001b[91m"
COLOR_RESET = "\u001b[0m"

if __name__ == "__main__":
    test_DQN_MLP(num_test_episodes = 100) # You can change the number of test episodes here
