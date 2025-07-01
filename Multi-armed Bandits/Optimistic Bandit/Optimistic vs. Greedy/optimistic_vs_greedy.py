import numpy as np
import matplotlib.pyplot as plt

# Constants
k = 10 
n_steps = 1000
n_runs = 2000

def run_bandit(Q1, epsilon, alpha = None): 
    optimal_action_counts = np.zeros(n_steps)

    for _ in range(n_runs):
        q_true = np.random.normal(0, 1, k)
        best_action = np.argmax(q_true)

        Q = np.ones(k) * Q1 # this marks the initial point, from which the agent starts
        N = np.zeros(k) # intialize the count array for each arm

        action_counts = [] 

        for t in range(n_steps): # for each time step,
            # epsilon greedy approach (epsilon = 0 counts as well)
            if np.random.rand() < epsilon:
                action = np.random.randint(k)
            else: 
                action = np.argmax(Q)

            reward = np.random.normal(q_true[action], 1)

            N[action] += 1

            if alpha is None:
                Q[action] += (reward - Q[action]) / N[action] # Sample Average Method
            else: 
                Q[action] += (alpha) * (reward - Q[action]) # Constant step size Method

            action_counts.append(int(action == best_action))

        optimal_action_counts += np.array(action_counts)
    
    return optimal_action_counts / n_runs * 100

# run both agents
opt_greedy = run_bandit(Q1 = 1, epsilon=0.0)
eps_greedy = run_bandit(Q1 = 0, epsilon=0.1)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(opt_greedy, label=r'Optimistic, greedy $Q_1=5,\ \epsilon=0$', color='deepskyblue')
plt.plot(eps_greedy, label=r'Realistic, $\epsilon$-greedy $Q_1=0,\ \epsilon=0.1$', color='gray')
plt.xlabel("Steps")
plt.ylabel("% Optimal action")
plt.title("Optimistic Initialization vs ε-Greedy")
plt.legend()
plt.grid()
plt.show()