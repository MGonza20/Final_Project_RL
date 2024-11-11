
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class LearningPathEnv(gym.Env):
    def __init__(self):
        super(LearningPathEnv, self).__init__()

        # Espacio de observación (estado)
        # Indice 0: Programacion, Indice 1: Matemáticas, Indice 2: Ciencias, Indice 3: Ingles, Indice 4: Historia
        self.num_topics = 5
        self.observation_space = spaces.Box(low=0, high=100, shape=(self.num_topics,), dtype=np.float32)

        # Espacio de acción: Acciones posibles (Seleccionar módulos o tareas específicas)
        self.action_space = spaces.Discrete(10)

        # Difulcultad de cada curso
        self.state_difficulty = np.array([75, 50, 80, 30, 10], dtype=np.float32)

        # Inicialización del estado del estudiante (habilidades en cada tema)
        self.state = np.zeros(self.num_topics, dtype=np.float32)

        # Filas: Actividades, Columnas: temas
        self.activity_impact = np.array([
            [5, 2, 1, 0, 0],  # Actividad 0: Impacta Programación, Matemáticas y Ciencias
            [0, 3, 0, 4, 1],  # Actividad 1: Impacta Matemáticas, Ingles e Historia
            [2, 0, 5, 1, 0],  # Actividad 2: Impacta Programación, Ciencias e Ingles
            [1, 1, 2, 3, 0],  # Actividad 3: Impacta Programación, Matemáticas, Ciencias e Ingles
            [0, 0, 3, 2, 5],  # Actividad 4: Impacta Ciencias, Ingles e Historia
        ], dtype=np.float32)

        self.current_step = 0


    def reset(self):
        # Reinicio del estado del estudiante a 0 en todos los temas
        self.state = np.zeros(self.num_topics, dtype=np.float32)
        self.current_step = 0
        return self.state

    def step(self, action):
        # Simulación del efecto de la acción en el nivel de habilidad

        # Incremento en las habilidades por tema tras tomar la acción
        base_improvements = self.activity_impact[action]

        skill_improvements = base_improvements * (1 - (self.state_difficulty / 100))

        self.state = np.clip(self.state + skill_improvements, 0, 100)

        # Calculo de la recompensa sumando las mejoras en habilidades
        reward = np.sum(skill_improvements)

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
