import numpy as np
from game.actions import Action
from game.state import GameState
from collections import defaultdict
import logging


class GTOSolver:
    def __init__(self, config):
        self.config = config
        self.regret_sum = defaultdict(self._create_zero_array)
        self.strategy_sum = defaultdict(self._create_zero_array)

    def _create_zero_array(self):
        return np.zeros(len(Action), dtype=np.float32)

    def get_action(self, state: GameState):
        info_set = self._get_info_set(state)
        strategy = self._get_strategy(info_set)
        valid_actions = state.get_valid_actions()
        valid_action_indices = np.array([action.value for action in valid_actions])
        valid_strategy = strategy[valid_action_indices]
        valid_strategy /= np.sum(valid_strategy)
        action = np.random.choice(valid_actions, p=valid_strategy)
        return action, strategy

    def update(self, state: GameState, iterations: int = None):
        if iterations is None:
            iterations = self.config.GTO_ITERATIONS
        for _ in range(iterations):
            self._cfr_iteration(state)
        self._prune_strategies()

    def _cfr_iteration(self, initial_state: GameState):
        stack = [(initial_state, 1.0, 1.0, [])]
        while stack:
            current_item = stack.pop()
            if isinstance(current_item[0], GameState):
                state, p0, p1, history = current_item
            else:
                logging.error(f"Invalid state: {current_item[0]}")
                continue

            if state.is_terminal():
                payoffs = state.get_payoff()
                self._backpropagate(history, payoffs)
                continue

            player = state.current_player
            info_set = self._get_info_set(state)
            strategy = self._get_strategy(info_set)
            valid_actions = state.get_valid_actions()

            for action in valid_actions:
                next_state, _, _ = state.apply_action(action)
                if player == 0:
                    new_p0 = p0 * strategy[action.value]
                    stack.append((next_state, new_p0, p1, history + [(info_set, action, player, strategy)]))
                else:
                    new_p1 = p1 * strategy[action.value]
                    stack.append((next_state, p0, new_p1, history + [(info_set, action, player, strategy)]))

    def _backpropagate(self, history, payoffs):
        for info_set, action, player, strategy in reversed(history):
            util = payoffs[player]
            regret = np.zeros(len(Action), dtype=np.float32)
            regret[action.value] = util - np.dot(strategy, np.full(len(Action), util))
            self.regret_sum[info_set] += regret
            self.strategy_sum[info_set] += strategy

    def _get_strategy(self, info_set):
        regrets = self.regret_sum[info_set]
        positive_regrets = np.maximum(regrets, 0)
        sum_positive_regrets = np.sum(positive_regrets)
        return positive_regrets / sum_positive_regrets if sum_positive_regrets > 0 else np.ones(len(Action), dtype=np.float32) / len(Action)

    def _get_info_set(self, state: GameState):
        return f"{state.stage}|{state.pot}|{state.community_cards}|{state.players[state.current_player].hand}"

    def _prune_strategies(self):
        for info_set in list(self.regret_sum.keys()):
            if np.sum(np.abs(self.regret_sum[info_set])) < self.config.GTO_PRUNE_THRESHOLD:
                del self.regret_sum[info_set]
                del self.strategy_sum[info_set]

    def get_state(self):
        return {
            'regret_sum': dict(self.regret_sum),
            'strategy_sum': dict(self.strategy_sum)
        }

    def set_state(self, state):
        self.regret_sum = defaultdict(self._create_zero_array, state['regret_sum'])
        self.strategy_sum = defaultdict(self._create_zero_array, state['strategy_sum'])

    def get_average_strategy(self, info_set):
        strategy_sum = self.strategy_sum[info_set]
        total = np.sum(strategy_sum)
        return strategy_sum / total if total > 0 else np.ones(len(Action), dtype=np.float32) / len(Action)
