import numpy as np
import math


class MCTS:
    def __init__(self, config):
        self.config = config
        self.Q = {}
        self.N = {}
        self.children = {}
        self.exploration_constant = config.MCTS_EXPLORATION_CONSTANT

    def search(self, state, num_simulations=None):
        if num_simulations is None:
            num_simulations = self.config.MCTS_SIMULATIONS

        root_key = self._get_state_key(state)
        self.children[root_key] = state.get_valid_actions()

        for _ in range(num_simulations):
            self._simulate(state.clone())

        return self._best_action(state)

    def _simulate(self, state):
        path = []
        current_state = state
        done = False

        while not done:
            state_key = self._get_state_key(current_state)
            if state_key not in self.children:
                return self._expand(current_state, path)

            action = self._select_action(current_state)
            path.append((state_key, action))
            current_state, rewards, done = current_state.apply_action(action)

        return self._backpropagate(path, rewards[current_state.current_player])

    def _expand(self, state, path):
        state_key = self._get_state_key(state)
        self.children[state_key] = state.get_valid_actions()
        action = np.random.choice(self.children[state_key])
        path.append((state_key, action))
        next_state, rewards, done = state.apply_action(action)
        return self._backpropagate(path, rewards[next_state.current_player])

    def _select_action(self, state):
        state_key = self._get_state_key(state)
        return max(self.children[state_key], key=lambda a: self._ucb_score(state_key, a))

    def _ucb_score(self, state_key, action):
        q = self.Q.get((state_key, action), 0)
        n = self.N.get((state_key, action), 0)
        total_n = sum(self.N.get((state_key, a), 0) for a in self.children[state_key])
        return q + self.exploration_constant * math.sqrt(2 * math.log(total_n + 1) / (n + 1))

    def _backpropagate(self, path, value):
        for state_key, action in reversed(path):
            if (state_key, action) not in self.Q:
                self.Q[(state_key, action)] = 0
                self.N[(state_key, action)] = 0
            self.N[(state_key, action)] += 1
            self.Q[(state_key, action)] += (value - self.Q[(state_key, action)]) / self.N[(state_key, action)]

    def _best_action(self, state):
        state_key = self._get_state_key(state)
        visits = [self.N.get((state_key, a), 0) for a in self.children[state_key]]
        return self.children[state_key][np.argmax(visits)]

    def _get_state_key(self, state):
        return state.to_string()
