import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

class LearningPathEnv(gym.Env):
    def __init__(self):
        super(LearningPathEnv, self).__init__()
        self.num_topics = 5
        self.observation_space = spaces.Box(low=0, high=100, shape=(self.num_topics,), dtype=np.float32)
        self.num_activities = 5
        self.action_space = spaces.Discrete(self.num_activities)
        self.state_difficulty = np.array([50, 30, 60, 20, 5], dtype=np.float32)
        self.state = np.zeros(self.num_topics, dtype=np.float32)
        self.activity_impact = np.array([
            [5, 2, 1, 0, 0], 
            [0, 3, 0, 4, 1], 
            [2, 0, 5, 1, 0], 
            [1, 1, 2, 3, 0], 
            [0, 0, 3, 2, 5],
        ], dtype=np.float32)
        self.current_step = 0

    def reset(self):
        self.state = np.zeros(self.num_topics, dtype=np.float32)
        self.current_step = 0
        return self.state

    def step(self, action):
        base_improvements = self.activity_impact[action]
        skill_improvements = base_improvements * (1 - (self.state_difficulty / 100))
        self.state = np.clip(self.state + skill_improvements, 0, 100)
        self.current_step += 1
        if np.all(self.state >= 100):
            reward = 1000 - self.current_step
            done = True
        else:
            reward = -1
            done = False
        return self.state, reward, done, {}

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(dqn(state_tensor)).item()

# Hyperparameters
batch_size = 128
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
target_update = 10
num_episodes = 1000
memory_size = 10000

env = LearningPathEnv()
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

dqn = DQN(input_dim, output_dim)
target_dqn = DQN(input_dim, output_dim)
target_dqn.load_state_dict(dqn.state_dict())
optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
memory = deque(maxlen=memory_size)

def update_model():
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(np.array(states))
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    q_values = dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target_dqn(next_states).max(1)[0]
    expected_q_values = rewards + gamma * next_q_values * (1 - dones)

    loss = nn.MSELoss()(q_values, expected_q_values.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Entrenamiento
all_rewards = []
all_actions = []
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    episode_actions = []
    for t in range(1, 10000):
        action = select_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        memory.append((state, action, reward, next_state, done))
        update_model()
        
        state = next_state
        episode_reward += reward
        episode_actions.append(action)  # Almacena las acciones del episodio

        if done:
            break
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    all_rewards.append(episode_reward)
    all_actions.append(episode_actions)
    
    if episode % target_update == 0:
        target_dqn.load_state_dict(dqn.state_dict())
        
    if episode % 100 == 0:
        print(f"Episode {episode}, Reward: {episode_reward}, Epsilon: {epsilon:.3f}")

# Visualización del progreso de recompensas
plt.plot(all_rewards)
plt.xlabel('Episodios')
plt.ylabel('Recompensa acumulada')
plt.title('Progreso de la recompensa por episodio')
plt.grid(True)
plt.show()

# Visualización de las acciones tomadas en el último episodio
plt.plot(all_actions[-1], marker='o', linestyle='-')
plt.xlabel('Paso')
plt.ylabel('Acción')
plt.title('Acciones tomadas en el último episodio')
plt.grid(True)
plt.show()
