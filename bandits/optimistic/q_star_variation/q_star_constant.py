import numpy as np 
import matplotlib.pyplot as plt

# Parameters
q_star = 1.0 # true reward value
Q_knot = 0.0
n_steps = 100
alphas = [0.05, 0.1, 0.2, 0.5]

expectations = {}
biases = {} 

for alpha in alphas: 
    exp_Qn = [] 
    bias_Qn = [] 

    for n in range(1, n_steps + 1):
        expected_Q = (1 - alpha)**n * Q_knot + (1 - (1 - alpha)**n) * q_star
        bias = expected_Q - q_star

        exp_Qn.append(expected_Q) 
        bias_Qn.append(bias)

    expectations[alpha] = exp_Qn
    biases[alpha] = bias_Qn

# Plot
plt.figure(figsize = (14, 5))
plt.get_current_fig_manager().set_window_title("q* evals")

# Plot I
plt.subplot(1, 2, 1)
for alpha in alphas: 
    plt.plot(expectations[alpha], label=f"Alpha = {alpha}")
plt.axhline(q_star, linestyle = '--', color = 'black', label = 'True q*')
plt.title("Expected Value of Qn")
plt.xlabel("Steps")
plt.ylabel("E[Qn]")
plt.legend()
plt.grid(True)

# Plot II
plt.subplot(1, 2, 2)
for alpha in alphas: 
    plt.plot(biases[alpha], label=f"Alpha = {alpha}")
plt.axhline(0, linestyle = '--', color = 'black', label = 'Zero Bias Line')
plt.title("Bias of Qn")
plt.xlabel("Steps")
plt.ylabel("Bias = E[Qn] - q*")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
