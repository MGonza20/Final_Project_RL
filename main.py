
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class LearningPathEnv(gym.Env):
    def __init__(self):
        super(LearningPathEnv, self).__init__()

        # Espacio de observación (estado) 
        # Indice 0: Programacion, Indice 1: Matemáticas, Indice 2: Ciencias, Indice 3: Ingles, ...
        self.num_topics = 5
        self.observation_space = spaces.Box(low=0, high=10, shape=(self.num_topics,), dtype=np.float32)

        # Espacio de acción: Acciones posibles (Seleccionar módulos o tareas específicas)
        self.action_space = spaces.Discrete(10)

        # Inicialización del estado del estudiante (habilidades en cada tema) [8, 6, 9, 3, 2]
        self.state = np.zeros(self.num_topics, dtype=np.float32)
        self.current_step = 0

    def reset(self):
        # Reinicio del estado del estudiante a 0 en todos los temas
        self.state = np.zeros(self.num_topics, dtype=np.float32)
        self.current_step = 0
        return self.state

    def step(self, action):
        # Simulación del efecto de la acción en el nivel de habilidad

        # Incremento en las habilidades por tema tras tomar la acción
        skill_improvement = np.random.uniform(0, 1, size=self.num_topics)
        self.state = np.clip(self.state + skill_improvement, 0, 10)

        # Calculo de la recompensa sumando las mejoras en habilidades
        reward = np.sum(skill_improvement)

        self.current_step += 1
        done = self.current_step >= 50

        return self.state, reward, done, {}

    def render(self, mode='human'):
        print(f"Estado actual (habilidades en cada tema): {self.state}")


env = LearningPathEnv()

best_actions = []
best_states = []
max_reward = -float('inf')

for episode in range(100):
    state = env.reset()
    episode_actions = []
    episode_states = [state]
    total_reward = 0

    for t in range(50):
        # Seleccionando acciones aleatorias (por el momento)
        action = env.action_space.sample()
        
        next_state, reward, done, _ = env.step(action)
        
        episode_actions.append(action)
        episode_states.append(next_state)
        
        total_reward += reward
        state = next_state
        
        if done: break

    # Verificación de mejor ruta de aprendizaje
    if total_reward > max_reward:
        max_reward = total_reward
        best_actions = episode_actions
        best_states = episode_states

print("Mejor Ruta de Aprendizaje Encontrada:")
print("\nSecuencia de Acciones:", best_actions)
print("\nSecuencia de Estados:", best_states)
print("\nRecompensa Total de la Mejor Ruta:", max_reward)
