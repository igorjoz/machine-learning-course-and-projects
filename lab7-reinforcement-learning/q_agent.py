import numpy as np
from rl_base import Agent, Action, State
import os

class QAgent(Agent):
    def __init__(self, n_states: int, n_actions: int, learning_rate=0.05, gamma=0.985, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        super().__init__(name="Q-Learning_Agent")
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros((n_states, n_actions))

    def update_action_policy(self) -> None:
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def choose_action(self, state: State) -> Action:
        assert 0 <= state < self.n_states, f"Bad state_idx. Has to be int between 0 and {self.n_states}"

        if np.random.rand() < self.epsilon:
            # Exploration: choose a random action
            return Action(np.random.randint(0, self.n_actions))
        else:
            # Exploitation: choose the best known action
            return Action(np.argmax(self.q_table[state]))

    def learn(self, state: State, action: Action, reward: float, new_state: State, done: bool) -> None:
        old_value = self.q_table[state, action]
        future_optimal_value = np.max(self.q_table[new_state]) if not done else 0
        new_value = old_value + self.learning_rate * (reward + self.gamma * future_optimal_value - old_value)
        self.q_table[state, action] = new_value

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)

    def get_instruction_string(self):
        return [f"Linearly decreasing eps-greedy: eps={self.epsilon:0.4f}"]
