import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, k):
        self.k = k
        self.q_true = np.random.normal(0, 1, k) # true reward for each arm
        self.best_action = np.argmax(self.q_true) # best reward among the estimates

    def pull(self, action):
        return np.random.normal(self.q_true[action], 1) # reward with significant noise
    
class EpsilonGreedy:
    def __init__(self, k, epsilon):
        self.k = k
        self.epsilon = epsilon
        self.counts = np.zeros(k)
        self.values = np.zeros(k)

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        return np.argmax(self.values)
    
    def update(self, action, reward):
        self.counts[action] += 1
        alpha = 1 / self.counts[action]

        self.values[action] += alpha * (reward - self.values[action])

class UCB: 
    def __init__(self, k, c):
        self.k = k
        self.c = c
        self.action_count = 0
        self.values = np.zeros(k)
        self.counts = np.zeros(k)

    def select_action(self):
        self.action_count += 1

        for i in range (self.k):
            if self.counts[i] == 0:
                return i
        
        ucb_values = self.values + self.c * np.sqrt(np.log(self.action_count) / self.counts)

        return np.argmax(ucb_values)
    
    def update(self, action, reward):
        self.counts[action] += 1
        alpha = 1 / self.counts[action]

        self.values[action] += alpha * (reward - self.values[action])
    
def simulate(bandit_class, k = 10, runs = 2000, steps = 1000, **kwargs):
    rewards = np.zeros(steps)

    for i in range(runs):
        bandit = Bandit(k)
        method = bandit_class(k, **kwargs)

        for t in range(steps):
            action = method.select_action()
            reward = bandit.pull(action)
            method.update(action, reward)

            rewards[t] += reward

    return rewards / runs

k = 10
steps = 1000
runs = 2000

avg_rewards_greedy = simulate(EpsilonGreedy, k = k, runs = runs, steps = steps, epsilon = 0.1)
avg_rewards_ucb = simulate(UCB, k = k, runs = runs, steps = steps, c = 2)

# Plot 
plt.figure(figsize=(14, 5))
plt.get_current_fig_manager().set_window_title("UCB vs. Greedy Epsilon = 0.1")

plt.plot(avg_rewards_greedy, label = "Epsilon Greedy (Epsilon = 0.1)", color = "mediumvioletred")
plt.plot(avg_rewards_ucb, label = "Upper Confidence Bound (c = 2)", color = "darkblue")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("UCB vs. Epsilon Greedy")
plt.legend()

plt.grid(True)
plt.show()
