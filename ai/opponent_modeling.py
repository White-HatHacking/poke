import numpy as np
from collections import defaultdict
from game.actions import Action
from game.state import GameState
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
import logging


class OpponentModel:
    def __init__(self, config):
        self.config = config
        self._validate_config()
        self.action_counts = defaultdict(self._create_zero_array)
        self.stage_action_counts = defaultdict(self._create_zero_array)
        self.stack_action_counts = defaultdict(self._create_zero_array)
        self.decay_factor = config.OPPONENT_MODEL_DECAY_FACTOR
        self.feature_extractors = [
            self._extract_basic_features,
            self._extract_hand_strength_features,
            self._extract_pot_features,
            self._extract_position_features
        ]
        self.classifiers = defaultdict(self._create_classifier)
        self.scalers = defaultdict(StandardScaler)

    def _create_zero_array(self):
        return np.zeros(len(Action))

    def _validate_config(self):
        required_attrs = ['OVERALL_WEIGHT', 'STAGE_WEIGHT', 'STACK_WEIGHT', 'ML_WEIGHT', 'STACK_BUCKET_SIZE', 'RF_N_ESTIMATORS', 'RF_MAX_DEPTH']
        for attr in required_attrs:
            if not hasattr(self.config, attr):
                raise AttributeError(f"Config object is missing required attribute: {attr}")

    def update(self, opponent_data):
        for player in set(state.current_player for state, _, _, _, _ in opponent_data):
            player_data = [data for data in opponent_data if data[0].current_player == player]
            states, actions, _, _, _ = zip(*player_data)

            features = np.array([self._extract_features(state) for state in states])
            self.scalers[player].partial_fit(features)
            scaled_features = self.scalers[player].transform(features)

            action_values = np.array([action.value for action in actions])
            self.classifiers[player].partial_fit(scaled_features, action_values, classes=range(len(Action)))

            # Vectorized operations for updating counts
            unique_actions, action_counts = np.unique(action_values, return_counts=True)
            self.action_counts[player][unique_actions] += action_counts

            for stage in set(state.stage for state in states):
                stage_mask = np.array([state.stage == stage for state in states])
                stage_actions = action_values[stage_mask]
                unique_stage_actions, stage_action_counts = np.unique(stage_actions, return_counts=True)
                self.stage_action_counts[(player, stage)][unique_stage_actions] += stage_action_counts

            stack_buckets = np.array([self._get_stack_bucket(state.players[player].stack) for state in states])
            for bucket in np.unique(stack_buckets):
                bucket_mask = stack_buckets == bucket
                bucket_actions = action_values[bucket_mask]
                unique_bucket_actions, bucket_action_counts = np.unique(bucket_actions, return_counts=True)
                self.stack_action_counts[(player, bucket)][unique_bucket_actions] += bucket_action_counts

            # Apply decay factor
            self.action_counts[player] *= self.decay_factor
            for stage in set(state.stage for state in states):
                self.stage_action_counts[(player, stage)] *= self.decay_factor
            for bucket in np.unique(stack_buckets):
                self.stack_action_counts[(player, bucket)] *= self.decay_factor

    def _update_counts(self, counts, action):
        counts *= self.decay_factor
        counts[action.value] += 1

    def get_opponent_tendencies(self, player, state):
        features = self._extract_features(state)
        features = np.array(features, dtype=np.float64).reshape(1, -1)

        # Check for non-numeric values and replace them
        non_numeric_mask = ~np.isfinite(features)
        if non_numeric_mask.any():
            print(f"Warning: Non-numeric values detected in features: {features[non_numeric_mask]}")
            features[non_numeric_mask] = 0.0

        if player not in self.scalers:
            self.scalers[player] = StandardScaler()
            self.scalers[player].fit(features)

        scaled_features = self.scalers[player].transform(features)

        if player not in self.classifiers:
            self.classifiers[player] = self._create_classifier()
            self.classifiers[player].partial_fit(scaled_features, [0], classes=range(len(Action)))

        if isinstance(self.classifiers[player], SGDClassifier) and not hasattr(self.classifiers[player], 'n_iter_'):
            # Not enough data, return uniform distribution
            return np.ones(len(Action)) / len(Action)

        ml_prediction = self.classifiers[player].predict_proba(scaled_features)[0]

        overall_tendency = self._normalize(self.action_counts.get(player, np.zeros(len(Action))))
        stage_tendency = self._normalize(self.stage_action_counts.get((player, state.stage), np.zeros(len(Action))))
        stack_bucket = self._get_stack_bucket(state.players[player].stack)
        stack_tendency = self._normalize(self.stack_action_counts.get((player, stack_bucket), np.zeros(len(Action))))

        combined_tendency = (
            self.config.OVERALL_WEIGHT * overall_tendency +
            self.config.STAGE_WEIGHT * stage_tendency +
            self.config.STACK_WEIGHT * stack_tendency +
            self.config.ML_WEIGHT * ml_prediction
        )
        return self._normalize(combined_tendency)

    def _normalize(self, array):
        total = np.sum(array)
        return array / total if total > 0 else np.ones_like(array) / len(array)

    def _get_stack_bucket(self, stack):
        return min(int(stack / (self.config.STARTING_STACK / 10)), 9)

    def _create_classifier(self):
        return SGDClassifier(loss='log_loss', alpha=0.01, max_iter=1000, tol=1e-3, warm_start=True)

    def _extract_features(self, state):
        features = []
        for i, extractor in enumerate(self.feature_extractors):
            try:
                feature = extractor(state)
                logging.debug(f"Feature {i} ({extractor.__name__}): {feature}")
                if isinstance(feature, list):
                    features.extend([float(f) if f is not None else 0.0 for f in feature])
                elif feature is not None:
                    features.append(float(feature))
                else:
                    features.append(0.0)
            except Exception as e:
                logging.error(f"Error in feature extraction {i} ({extractor.__name__}): {e}")
                features.append(0.0)
        return features

    def _extract_basic_features(self, state):
        return [
            state.pot / state.config.STARTING_STACK,
            len(state.community_cards) / 5,
            state.players[state.current_player].stack / state.config.STARTING_STACK
        ]

    def _extract_hand_strength_features(self, state):
        try:
            hand_strength = state._estimate_hand_strength()
            logging.debug(f"Hand strength: {hand_strength}")
            return [float(hand_strength) if hand_strength is not None else 0.5]
        except Exception as e:
            logging.error(f"Error in hand strength estimation: {e}")
            return [0.5]  # Use a default value of 0.5 (50% win probability) if estimation fails or returns None

    def _extract_pot_features(self, state):
        return [state.pot / (state.config.STARTING_STACK * state.config.NUM_PLAYERS)]

    def _extract_position_features(self, state):
        return [state.current_player / state.config.NUM_PLAYERS]

    def get_state(self):
        return {
            'action_counts': dict(self.action_counts),
            'stage_action_counts': dict(self.stage_action_counts),
            'stack_action_counts': dict(self.stack_action_counts)
        }

    def set_state(self, state):
        self.action_counts = defaultdict(self._create_zero_array)
        self.stage_action_counts = defaultdict(self._create_zero_array)
        self.stack_action_counts = defaultdict(self._create_zero_array)

        for key, value in state['action_counts'].items():
            self.action_counts[key] = np.array(value)

        for key, value in state['stage_action_counts'].items():
            self.stage_action_counts[key] = np.array(value)

        for key, value in state['stack_action_counts'].items():
            self.stack_action_counts[key] = np.array(value)
