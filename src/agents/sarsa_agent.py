# SARSA agent (on-policy TD learning).

import json
import os
from collections import defaultdict
from typing import Any, Dict

import numpy as np


class SARSAAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.9995,
    ):
        self.n_states = n_states
        self.n_actions = n_actions

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.episode_count = 0
        self.total_steps = 0

    def choose_action(self, state: int, training: bool = True) -> int:
        if not training or np.random.random() > self.epsilon:
            return int(np.argmax(self.q_table[state]))
        return int(np.random.randint(0, self.n_actions))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: int,
    ) -> None:
        q_current = self.q_table[state][action]
        q_next = self.q_table[next_state][next_action]
        td_target = reward + self.gamma * q_next
        td_error = td_target - q_current
        self.q_table[state][action] = q_current + self.alpha * td_error
        self.total_steps += 1

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode_count += 1

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        q_table_serializable = {str(k): v.tolist() for k, v in self.q_table.items()}

        agent_data = {
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'q_table': q_table_serializable,
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
        }

        with open(filepath, 'w') as f:
            json.dump(agent_data, f, indent=2)

    def load(self, filepath: str) -> None:
        with open(filepath, 'r') as f:
            agent_data = json.load(f)

        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        for state_str, q_values in agent_data['q_table'].items():
            self.q_table[int(state_str)] = np.array(q_values)

        self.epsilon = agent_data['epsilon']
        self.episode_count = agent_data['episode_count']
        self.total_steps = agent_data['total_steps']

    def get_stats(self) -> Dict[str, Any]:
        return {
            'n_states_visited': len(self.q_table),
            'n_states_total': self.n_states,
            'coverage': len(self.q_table) / self.n_states * 100,
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'alpha': self.alpha,
            'gamma': self.gamma,
        }

    def get_q_values(self, state: int) -> np.ndarray:
        return self.q_table[state].copy()

    def reset_epsilon(self, epsilon: float = 1.0) -> None:
        self.epsilon = epsilon
