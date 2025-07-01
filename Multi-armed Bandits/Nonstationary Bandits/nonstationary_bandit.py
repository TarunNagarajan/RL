import numpy as np
import matplotlib.pyplot as plt

# Demonstrate how sample-average methods struggle in non-stationary envs
# and how constant step size methods work better. 

# Parameters
k = 10 
runs = 2000
steps = 10000

epsilon = 0.1
alpha = 0.1

q_star_walk_std = 0.01

def run_bandit(non_stationary_environment = True, use_sample_average = True):
    optimal_action_counts = np.zeros(steps)
    average_reward = np.zeros(steps)

    for run in range(runs):
        q_true = np.zeros(k)
        Q = np.zeros(k)
        N = np.zeros(k)

        for step in range(steps):
            # epsilon greedy selection
            if np.random.rand() < epsilon:
                action = np.random.randint(k) # select a random level to pull on. 
            else: 
                action = np.argmax(Q) # select the maximum among the action value estimates.
            
            reward = np.random.normal(q_true[action], 1) # reward from the selected action

            best_action = np.argmax(q_true) # actual ground reality action value

            if action == best_action: 
                optimal_action_counts[step] += 1 

            N[action] += 1

            # update estimates depending upon the environment.
            if use_sample_average:
                Q[action] += (1 / N[action]) * (reward - Q[action])
            else:
                Q[action] += (alpha) * (reward - Q[action]) 

            if non_stationary_environment:
                q_true += np.random.normal(0, q_star_walk_std, k) # each arm of the true action-value gets updated
                # to simulate the non-stationary environment.
            
            average_reward[step] += reward 

    return (optimal_action_counts / runs * 100, average_reward / runs)

opt_sample_avg, rewards_sample_avg = run_bandit(use_sample_average = True)
opt_const_alpha, rewards_const_alpha = run_bandit(use_sample_average = False)


# Visualize both runs
plt.figure(figsize=(14, 5))
plt.get_current_fig_manager().set_window_title("The Non-Stationary Bandit")

# Plot I: % Optimal Action
plt.subplot(1, 2, 1)
plt.plot(opt_sample_avg, label = "Sample Average", color = 'teal')
plt.plot(opt_const_alpha, label = "Constant Alpha = 0.1", color = 'navy')

plt.xlabel('Steps')
plt.ylabel('% Optimal Actions')
plt.legend()
plt.grid(True) 

# Plot II: Average Rewards
plt.subplot(1, 2, 2)
plt.plot(rewards_sample_avg, label = "Sample Average", color = 'teal')
plt.plot(rewards_const_alpha, label = "Constant Alpha = 0.1", color = 'navy')

plt.xlabel('Steps')
plt.ylabel('Average Rewards')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

