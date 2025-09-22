import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import gym

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen = capacity)
    
    def push(self, state, action, reward, next_action, done):
        self.buffer.append((state, action, reward, next_action, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_action, done = map(np.stack, zip(*batch))
        return state, action, reward, next_action, done
    
    def __len__(self):
        return len(self.buffer)
    
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size = 120):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):   
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, lr = 1e-3, gamma = 0.99, epsilon = 1, min_epsilon = 0.01, epsilon_decay = 0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr = lr)

        self.memory = ReplayBuffer(1000)
        self.batch_size = 64
        self.update_target_freq = 100 
        self.step_count = 0

    def act(self, state):
        if (random.random() < self.epsilon):
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)

        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
