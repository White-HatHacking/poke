import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import threading
import multiprocessing as mp
from functools import partial
from game.engine import PokerEngine
from game.state import GameState
from game.actions import Action
from .search import MCTS
from .self_play import SelfPlay
from .opponent_modeling import OpponentModel
from .prioritized_replay_buffer import PrioritizedReplayBuffer
from models.neural_net import NeuralNetwork
from models.gto_solver import GTOSolver
import logging
import traceback
from collections import deque
import time
import os


class PokerAgent:
    def __init__(self, config, engine):
        self.config = config
        self.engine = engine
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nn = NeuralNetwork(config).to(self.device)
        self.target_nn = NeuralNetwork(config).to(self.device)
        self.target_nn.load_state_dict(self.nn.state_dict())
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA)
        state_dim = config.INPUT_SIZE
        action_dim = config.OUTPUT_SIZE
        self.gto_solver = GTOSolver(config)
        self.search = MCTS(config)
        self.opponent_model = OpponentModel(config) if config.USE_OPPONENT_MODELING else None
        self.memory = PrioritizedReplayBuffer(config.MEMORY_SIZE, config.INPUT_SIZE, config.OUTPUT_SIZE)
        self.elo_rating = config.INITIAL_ELO

    def _get_latest_model(self):
        model_dir = self.config.MODEL_SAVE_DIR
        if not os.path.exists(model_dir):
            return None
        models = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        if not models:
            return None
        return os.path.join(model_dir, max(models, key=lambda x: os.path.getmtime(os.path.join(model_dir, x))))

    def _evaluate(self, engine):
        logging.info("Starting evaluation...")
        if self.config.DETAILED_REPORTING:
            logging.debug("Evaluation configuration: "
                          f"NUM_PROCESSES={self.config.NUM_PROCESSES}, "
                          f"EVALUATION_EPISODES={self.config.EVALUATION_EPISODES}")

        with mp.Pool(processes=self.config.NUM_PROCESSES) as pool:
            results = pool.map(self._evaluate_episode, [engine] * self.config.EVALUATION_EPISODES)

        total_reward = sum(r for r, _ in results)
        wins = sum(w for _, w in results)

        avg_reward = total_reward / self.config.EVALUATION_EPISODES
        win_rate = wins / self.config.EVALUATION_EPISODES
        elo_change = self.config.ELO_K_FACTOR * (win_rate - 0.5)

        logging.info(f"Evaluation completed: Avg Reward: {avg_reward:.2f}, Win Rate: {win_rate:.2%}, ELO Change: {elo_change:.2f}")
        if self.config.DETAILED_REPORTING:
            logging.debug(f"Total Reward: {total_reward}, Wins: {wins}, ELO Change: {elo_change:.2f}")

        return avg_reward, win_rate, elo_change

    def _evaluate_episode(self, engine):
        logging.debug("Starting evaluation episode")
        state = engine.reset()
        done = False
        total_reward = 0
        step_count = 0

        if self.config.DETAILED_REPORTING:
            logging.debug(f"Initial state: {state}")

        while not done:
            action = self.get_action(state)
            state, reward, done = engine.step(action)
            total_reward += reward[0]
            step_count += 1

            if self.config.DETAILED_REPORTING:
                logging.debug(f"Step {step_count}: Action: {action}, Reward: {reward[0]}, Done: {done}, State: {state}")

        logging.debug(f"Evaluation episode completed: Reward: {total_reward}, Win: {int(reward[0] > 0)}")
        if self.config.DETAILED_REPORTING:
            logging.debug(f"Final State: {state}, Total Reward: {total_reward}, Steps: {step_count}")

        return total_reward, int(reward[0] > 0)

    @staticmethod
    def _init_worker(config, epsilon, completed_episodes):
        global shared_config, shared_epsilon, shared_completed_episodes
        shared_config = config
        shared_epsilon = epsilon
        shared_completed_episodes = completed_episodes

    def train(self):
        # Load the latest saved model if it exists
        if self.config.DETAILED_REPORTING:
            logging.info("Detailed reporting enabled. Logging detailed training information.")
        latest_model = self._get_latest_model()
        if latest_model:
            self.load(latest_model)
            logging.info(f"Loaded latest model: {latest_model}")

        self_play = SelfPlay(self.config, self)
        evaluation_results = []
        best_elo = self.elo_rating

        num_processes = min(mp.cpu_count(), self.config.MAX_PROCESSES)
        shared_epsilon = mp.Value('d', self.config.INITIAL_EPSILON)
        completed_episodes = mp.Value('i', 0)

        with mp.Pool(processes=num_processes, initializer=self._init_worker,
                     initargs=(self.config, shared_epsilon, completed_episodes)) as pool:

            train_episode_partial = partial(self._train_episode_worker, engine=self.engine, agent=self, self_play=self_play)

            progress_bar = tqdm(total=self.config.NUM_TRAINING_ITERATIONS, desc="Training Progress")

            try:
                chunk_size = max(1, self.config.NUM_TRAINING_ITERATIONS // (num_processes * 10))
                async_result = pool.map_async(train_episode_partial, range(self.config.NUM_TRAINING_ITERATIONS), chunksize=chunk_size)

                while not async_result.ready():
                    completed = completed_episodes.value
                    progress_bar.n = completed
                    progress_bar.set_postfix({"Episodes": completed, "ELO": f"{self.elo_rating:.2f}"})
                    progress_bar.refresh()

                    # if self.config.DETAILED_REPORTING:
                    #     logging.debug(f"Completed episodes: {completed}, Current ELO: {self.elo_rating:.2f}")

                    if completed >= self.config.NUM_TRAINING_ITERATIONS:
                        logging.info("Reached target number of training iterations.")
                        break

                    if completed % self.config.EVALUATION_INTERVAL == 0 and completed > 0:
                        avg_reward, win_rate, elo_change = self._evaluate(self.engine)
                        self.elo_rating += elo_change
                        evaluation_results.append((completed, avg_reward, win_rate, self.elo_rating))
                        logging.info(f"Episode {completed}: Avg Reward: {avg_reward:.2f}, Win Rate: {win_rate:.2%}, ELO: {self.elo_rating:.2f}")

                        if self.elo_rating > best_elo:
                            best_elo = self.elo_rating
                            model_filename = f"best_model_elo_{int(best_elo)}.pth"
                            model_path = os.path.join(self.config.MODEL_SAVE_DIR, model_filename)
                            self.save(model_path)
                            logging.info(f"New best model saved: {model_path}")

                        progress_bar.set_description(f"Training Progress - ELO: {self.elo_rating:.2f}")

                    if completed % self.config.SAVE_INTERVAL == 0 and completed > 0:
                        model_path = f"model_episode_{completed}.pth"
                        self.save(model_path)
                        logging.info(f"Model saved at episode {completed}: {model_path}")

                    time.sleep(0.1)  # Avoid busy waiting

                # Process the results
                logging.info("Waiting for async_result to complete...")
                results = async_result.get(timeout=10)  # Add a timeout to prevent hanging
                logging.info("async_result completed")
                self._update_shared_resources_batch(results)

            except TimeoutError:
                logging.error("Timeout occurred while getting results from async_result.")
            except Exception as e:
                logging.error(f"An error occurred during training: {str(e)}")
                logging.exception("Exception details:")
            finally:
                progress_bar.close()

        if not evaluation_results:
            logging.warning("No evaluation results were produced during training.")
        else:
            logging.info(f"Training completed with {len(evaluation_results)} evaluations.")

        return evaluation_results

    def _update_shared_resources_batch(self, results):
        logging.info(f"Updating shared resources with {len(results)} results")
        all_memory = []
        for stats, _ in results:
            if stats is not None:
                all_memory.extend(stats['memory'])

        # Batch update memory
        self.memory.push_batch(all_memory)

        # Perform network updates
        if len(self.memory) >= self.config.BATCH_SIZE:
            self._update_network_batch()

        # Update target network
        if np.random.rand() < self.config.TARGET_UPDATE_FREQUENCY:
            self.target_nn.load_state_dict(self.nn.state_dict())

        # Update GTO solver asynchronously
        if np.random.rand() < self.config.GTO_UPDATE_FREQUENCY:
            self._update_gto_solver_async()

    def _update_gto_solver_async(self):
        logging.info("Updating GTO solver asynchronously")

        def update_gto():
            initial_state = self.engine.reset()
            if isinstance(initial_state, GameState):
                self.gto_solver.update(initial_state, iterations=self.config.GTO_UPDATE_ITERATIONS)

        threading.Thread(target=update_gto).start()

    @staticmethod
    def _train_episode_worker(episode, engine, agent, self_play):
        try:
            global shared_config, shared_epsilon, shared_completed_episodes
            logging.debug(f"Starting episode {episode}")
            stats = self_play.run_episode(engine, agent)
            logging.debug(f"Finished episode {episode}")
            with shared_completed_episodes.get_lock():
                shared_completed_episodes.value += 1
            logging.debug(f"Updated shared_completed_episodes to {shared_completed_episodes.value}")
            return stats, None  # Return None for episode_loss to reduce data transfer
        except Exception as e:
            logging.error(f"Error in worker process: {str(e)}")
            traceback.print_exc()
            return None, None

    def _update_model(self):
        if len(self.memory) < self.config.BATCH_SIZE:
            return 0

        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(
            self.config.BATCH_SIZE, self.config.PER_ALPHA, self.config.PER_BETA
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        current_q_values = self.nn(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_nn(next_states).max(1)[0].detach()
        expected_q_values = rewards + (self.config.GAMMA * next_q_values * (1 - dones))

        loss = (current_q_values.squeeze() - expected_q_values).pow(2) * weights
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.nn.parameters(), self.config.MAX_GRAD_NORM)
        self.optimizer.step()

        td_errors = (current_q_values.squeeze() - expected_q_values).abs().detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        return loss.item()

    def _update_shared_resources(self, stats):
        logging.info(f"Updating shared resources with stats")
        # Batch update memory
        self.memory.push_batch(stats['memory'])

        # Perform network updates less frequently
        if len(self.memory) >= self.config.BATCH_SIZE * self.config.UPDATE_FREQUENCY:
            self._update_network_batch()

        # Update target network less frequently
        if np.random.rand() < self.config.TARGET_UPDATE_FREQUENCY:
            self.target_nn.load_state_dict(self.nn.state_dict())

        # Update GTO solver asynchronously
        if np.random.rand() < self.config.GTO_UPDATE_FREQUENCY:
            self._update_gto_solver_async()

    def _scale_rewards(self, rewards):
        return np.clip(rewards, -1, 1)

    def _pad_and_convert(self, states):
        max_len = max(len(s) for s in states)
        padded_states = [np.pad(s, (0, max_len - len(s)), 'constant') for s in states]
        return torch.FloatTensor(np.array(padded_states)).to(self.device)

    def _update_network_batch(self):
        if len(self.memory) < self.config.BATCH_SIZE:
            return 0

        total_loss = 0
        updates = min(self.config.UPDATES_PER_EPISODE, len(self.memory) // self.config.BATCH_SIZE)

        self.optimizer.zero_grad()
        for _ in range(updates):
            loss = self._update_model()
            if loss is not None:
                total_loss += loss
                if (_ + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

        self.scheduler.step()

        avg_loss = total_loss / updates if updates > 0 else 0
        return avg_loss

    def get_action(self, state):
        valid_actions = state.get_valid_actions()
        if len(valid_actions) == 1:
            return valid_actions[0]

        state_vector = state.vectorize()
        nn_action, nn_confidence = self._get_nn_action(state_vector)
        gto_action, gto_confidence = self._get_gto_action(state)
        search_action, search_confidence = self._get_search_action(state)

        if self.config.USE_OPPONENT_MODELING and self.opponent_model is not None:
            opponent_model_action, opponent_model_confidence = self._get_opponent_model_action(state)
            action_confidences = {
                nn_action: nn_confidence * self.config.NN_WEIGHT,
                gto_action: gto_confidence * self.config.GTO_WEIGHT,
                search_action: search_confidence * self.config.SEARCH_WEIGHT,
                opponent_model_action: opponent_model_confidence * self.config.OPPONENT_MODEL_WEIGHT
            }
        else:
            action_confidences = {
                nn_action: nn_confidence * (self.config.NN_WEIGHT + self.config.OPPONENT_MODEL_WEIGHT),
                gto_action: gto_confidence * self.config.GTO_WEIGHT,
                search_action: search_confidence * self.config.SEARCH_WEIGHT,
            }

        best_action = max(action_confidences, key=lambda a: action_confidences.get(a, 0) if a in valid_actions else float('-inf'))
        return best_action if best_action in valid_actions else max(valid_actions, key=lambda a: action_confidences.get(a, 0))

    def _get_nn_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_values = self.nn(state_tensor)
        action = Action(action_values.squeeze().argmax().item())
        confidence = F.softmax(action_values.squeeze(), dim=-1).max().item()
        return action, confidence

    def _get_gto_action(self, state):
        action, strategy = self.gto_solver.get_action(state)
        confidence = np.max(strategy)
        return action, confidence

    def _get_search_action(self, state):
        action = self.search.search(state, num_simulations=self.config.MCTS_SIMULATIONS)
        confidence = 1.0  # You might want to adjust this based on the search results
        return action, confidence

    def _get_opponent_model_action(self, state):
        tendencies = self.opponent_model.get_opponent_tendencies(state.current_player, state)
        valid_actions = state.get_valid_actions()
        valid_tendencies = [tendencies[action.value] for action in valid_actions]
        action = valid_actions[np.argmax(valid_tendencies)]
        confidence = np.max(valid_tendencies)
        return action, confidence

    def save(self, path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save({
            'nn_state_dict': self.nn.state_dict(),
            'target_nn_state_dict': self.target_nn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'gto_solver_state': self.gto_solver.get_state(),
            'opponent_model_state': self.opponent_model.get_state(),
            'elo_rating': self.elo_rating,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.nn.load_state_dict(checkpoint['nn_state_dict'])
        self.target_nn.load_state_dict(checkpoint['target_nn_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.gto_solver.set_state(checkpoint['gto_solver_state'])

        # Check if opponent_model exists before trying to set its state
        if self.opponent_model is not None and 'opponent_model_state' in checkpoint:
            self.opponent_model.set_state(checkpoint['opponent_model_state'])
        elif 'opponent_model_state' in checkpoint:
            logging.warning("Opponent model state found in checkpoint, but opponent modeling is disabled in current configuration.")

        self.elo_rating = checkpoint.get('elo_rating', self.config.INITIAL_ELO)

    def clone(self):
        new_agent = PokerAgent(self.config)
        new_agent.nn.load_state_dict(self.nn.state_dict())
        new_agent.target_nn.load_state_dict(self.target_nn.state_dict())
        new_agent.gto_solver.set_state(self.gto_solver.get_state())
        new_agent.opponent_model.set_state(self.opponent_model.get_state())
        return new_agent
