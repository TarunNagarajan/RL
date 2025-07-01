import numpy as np
import matplotlib.pyplot as plt

def run_bandit(steps, runs, epsilon):
    k = 10
    avg_rewards = np.zeros(steps)
    optimal_action_counts = np.zeros(steps)

    for run in range(runs):
        q_true = np.random.normal(0, 1, k)
        best_action = np.argmax(q_true) # q*(a) = E[R_t | A_t = a]

        Q = np.zeros(k) # Initial Estimates
        N = np.zeros(k) # Action frequency array

        for t in range(steps): # for each time step
            if np.random.rand() < epsilon:
                # epsilon - greedy action selection
                action = np.random.randint(k)
            else: 
                # exploitation
                action = np.argmax(Q) 

            reward = np.random.normal(q_true[action], 1) 
            # rewards sampled from a normal distribution with mean q_true and std dev 1
            
            N[action] += 1 
            Q[action] += (reward - Q[action]) / N[action]

            avg_rewards[t] += reward
            if action == best_action:
                optimal_action_counts[t] += 1

    avg_rewards /= runs
    optimal_action_percentage = (optimal_action_counts / runs) * 100

    return avg_rewards, optimal_action_percentage

# Parameters
steps = 1000
runs = 2000

rewards_00, optimal_00 = run_bandit(steps, runs, epsilon = 0)
rewards_01, optimal_01 = run_bandit(steps, runs, epsilon = 0.1)
rewards_001, optimal_001 = run_bandit(steps, runs, epsilon = 0.01)

plt.figure(figsize=(14, 5))

# Plot I: Average Reward over Time
plt.subplot(1, 2, 1)
plt.plot(rewards_00, label = 'epsilon = 0', color = 'green')
plt.plot(rewards_001, label = 'epsilon = 0.01', color = 'red')
plt.plot(rewards_01, label = 'epsilon = 0.1', color = 'blue')

plt.title('Average Reward over Time')
plt.xlabel('Steps')
plt.ylabel('Avg Rwd')
plt.legend()
plt.grid(True)


# Plot II: % Optimal Action Selected over Time
plt.subplot(1, 2, 2)
plt.plot(optimal_00, label = 'epsilon = 0', color = 'green')
plt.plot(optimal_001, label = 'epsilon = 0.01', color = 'red')
plt.plot(optimal_01, label = 'epsilon = 0.1', color = 'blue')

plt.title('% Optimal Action Selected over Time')
plt.xlabel('Steps')
plt.ylabel('% Optimal Action')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
