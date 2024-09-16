import random
import numpy as np
import logging
from .opponent_modeling import OpponentModel


class SelfPlay:
    def __init__(self, config, agent):
        if config.USE_OPPONENT_MODELING:
            self.opponent_model = OpponentModel(config)
            if self.opponent_model is None:
                logging.warning("Opponent modeling is enabled in config but failed to initialize.")
        else:
            self.opponent_model = None
        self.config = config
        self.agent = agent
        self.epsilon = config.INITIAL_EPSILON
        self.epsilon_decay = config.EPSILON_DECAY
        self.epsilon_min = config.EPSILON_MIN
        self.previous_agents = []

    def run_episode(self, engine, agent):
        logging.debug("Starting run_episode")
        state = engine.reset()
        done = False
        episode_memory = []
        total_reward = np.zeros(self.config.NUM_PLAYERS, dtype=np.float32)

        opponent = self._select_opponent()
        logging.debug("Opponent selected")

        step_count = 0
        while not done:
            logging.debug(f"Step {step_count}")
            if state.current_player == 0:
                action = self._get_action(agent, state)
            else:
                action = self._get_action(opponent, state)
            next_state, rewards, done = engine.step(action)
            episode_memory.append((state.vectorize(), action.value, rewards[0], next_state.vectorize(), done))
            state = next_state
            total_reward += rewards
            step_count += 1

        win_rate = float(total_reward[0] > 0)
        self._update_epsilon()
        if self.config.USE_OPPONENT_MODELING and engine.current_hand % self.config.OPPONENT_MODEL_UPDATE_FREQUENCY == 0:
            agent.opponent_model.update(engine.get_game_history())

        logging.debug(f"Episode finished after {step_count} steps")
        return {
            "epsilon": self.epsilon,
            "avg_reward": float(total_reward[0]),
            "win_rate": win_rate,
            "memory": episode_memory
        }

    def _get_action(self, agent, state):
        return np.random.choice(state.get_valid_actions()) if np.random.random() < self.epsilon else agent.get_action(state)

    def _update_epsilon(self):
        self.epsilon = max(self.config.EPSILON_MIN, self.epsilon * self.config.EPSILON_DECAY)

    def _select_opponent(self):
        if not self.previous_agents or np.random.random() > self.config.PREVIOUS_OPPONENT_PROB:
            return self.agent
        return np.random.choice(self.previous_agents)

    def update_previous_agents(self, agent):
        self.previous_agents.append(agent.clone())
        if len(self.previous_agents) > self.config.MAX_PREVIOUS_AGENTS:
            self.previous_agents.pop(0)
