import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class LearningPathEnv(gym.Env):
    def __init__(self):
        super(LearningPathEnv, self).__init__()

        # Espacio de observación (estado)
        self.num_topics = 5
        self.observation_space = spaces.Box(low=0, high=100, shape=(self.num_topics,), dtype=np.float32)

        # Espacio de acción: Acciones posibles (Seleccionar tareas específicas)
        self.num_activities = 5
        self.action_space = spaces.Discrete(self.num_activities)

        # Dificultad de cada curso
        self.state_difficulty = np.array([50, 30, 60, 20, 5], dtype=np.float32)

        # Inicialización del estado del estudiante (habilidades en cada tema)
        self.state = np.zeros(self.num_topics, dtype=np.float32)

        # Filas: Actividades, Columnas: temas
        self.activity_impact = np.array([
            [5, 2, 1, 0, 0],  # Actividad 0
            [0, 3, 0, 4, 1],  # Actividad 1
            [2, 0, 5, 1, 0],  # Actividad 2
            [1, 1, 2, 3, 0],  # Actividad 3
            [0, 0, 3, 2, 5],  # Actividad 4
        ], dtype=np.float32)

        self.current_step = 0

        # Parámetros para discretización del estado
        self.num_bins = 10
        self.bins = [np.linspace(0, 100, self.num_bins + 1) for _ in range(self.num_topics)]

    def reset(self):
        self.state = np.zeros(self.num_topics, dtype=np.float32)
        self.current_step = 0
        return self.state

    def step(self, action):
        base_improvements = self.activity_impact[action]
        skill_improvements = base_improvements * (1 - (self.state_difficulty / 100))
        self.state = np.clip(self.state + skill_improvements, 0, 100)
        self.current_step += 1

        # Recompensa y condición de finalización
        if np.all(self.state >= 100):
            reward = 1000 - self.current_step  # Mayor recompensa por alcanzar el objetivo más rápido
            done = True
        else:
            reward = -1  # Penalización por cada paso hasta alcanzar el objetivo
            done = False

        return self.state, reward, done, {}

    def render(self, mode='human'):
        print(f"Paso {self.current_step}: Estado actual (habilidades en cada tema): {self.state}")

    def discretize_state(self, state):
        discrete_state = []
        for i in range(self.num_topics):
            discrete_value = np.digitize(state[i], self.bins[i]) - 1
            if discrete_value == self.num_bins:
                discrete_value = self.num_bins - 1
            discrete_state.append(discrete_value)
        return tuple(discrete_state)

# Parámetros de Q-Learning
alpha = 0.1  # Tasa de aprendizaje
gamma = 0.99  # Factor de descuento
epsilon = 1.0  # Tasa de exploración inicial
epsilon_min = 0.01
epsilon_decay = 0.999
num_episodes = 1000

env = LearningPathEnv()

# Inicializar la tabla Q
Q_table = np.zeros((env.num_bins,) * env.num_topics + (env.action_space.n,))

best_actions = []
best_states = []
min_steps = float('inf')

state_progress = []  # Para guardar el número de pasos en los que se alcanzó el objetivo

for episode in range(num_episodes):
    state = env.reset()
    discrete_state = env.discretize_state(state)
    episode_actions = []
    episode_states = [state.copy()]
    total_reward = 0

    for t in range(1, 10000):  # Usamos un número grande para permitir suficientes pasos
        # Selección de acción epsilon-greedy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_table[discrete_state])

        next_state, reward, done, _ = env.step(action)
        discrete_next_state = env.discretize_state(next_state)

        # Actualización Q-Learning
        best_next_action = np.argmax(Q_table[discrete_next_state])
        td_target = reward + gamma * Q_table[discrete_next_state + (best_next_action,)]
        td_error = td_target - Q_table[discrete_state + (action,)]
        Q_table[discrete_state + (action,)] += alpha * td_error

        episode_actions.append(action)
        episode_states.append(next_state.copy())
        total_reward += reward

        state = next_state
        discrete_state = discrete_next_state

        if done:
            # Actualizar la mejor ruta si se alcanzó en menos pasos
            if env.current_step < min_steps:
                min_steps = env.current_step
                best_actions = episode_actions
                best_states = episode_states
            break

    # Decaimiento de epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

print("Mejor Ruta de Aprendizaje Encontrada:")

if best_actions:
    print(f"\nNúmero mínimo de pasos para alcanzar el objetivo: {min_steps}")
    print("\nSecuencia de Acciones:")
    print(best_actions)
    print("\nSecuencia de Estados:")
    for idx, s in enumerate(best_states):
        print(f"Paso {idx}: {s}")
else:
    print("\nNo se encontró una ruta que alcance el objetivo.")

# Generar la gráfica de progreso de habilidades en el mejor episodio
if best_states:
    # Nombres de los temas
    topic_names = ['Programación', 'Matemáticas', 'Ciencias', 'Inglés', 'Historia']

    # Convertir best_states a un array de NumPy
    best_states_array = np.array(best_states)  # shape: (num_steps, num_topics)
    steps = np.arange(len(best_states_array))

    plt.figure(figsize=(12, 6))

    for i in range(env.num_topics):
        plt.plot(steps, best_states_array[:, i], label=topic_names[i])

    plt.xlabel('Paso')
    plt.ylabel('Nivel de Habilidad')
    plt.title('Progreso de Habilidades por Paso en el Mejor Episodio')
    plt.legend()
    plt.grid(True)
    plt.show()
