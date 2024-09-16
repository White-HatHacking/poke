import random
import itertools
from itertools import chain
from game.actions import Action
from collections import Counter
import numpy as np
from numba import jit
import multiprocessing as mp


class Player:
    def __init__(self, stack):
        self.stack = stack
        self.hand = []
        self.is_active = True  # Changed from a method to an attribute
        self.has_acted = False
        self.episodes_completed = 0
        self.engine = None
        self.current_bet = 0
        self.total_bet = 0
        self.history = []
        self.action_counts = np.zeros(len(Action))

    def get_history(self):
        return self.history

    def is_Active(self):
        return bool(self.hand)

    def post_blind(self, amount):
        self.stack -= amount

    def post_ante(self, amount):
        self.stack -= amount

    def clone(self):
        new_player = Player(self.stack)
        new_player.hand = self.hand.copy()
        new_player.is_active = self.is_active
        new_player.has_acted = self.has_acted
        new_player.current_bet = self.current_bet
        new_player.total_bet = self.total_bet
        return new_player

    def reset(self):
        self.hand = []
        self.is_active = True
        self.has_acted = False
        self.current_bet = 0
        self.total_bet = 0


class GameState:
    def __init__(self, config):
        self.config = config
        self.players = [Player(config.STARTING_STACK) for _ in range(config.NUM_PLAYERS)]
        self.community_cards = []
        self.pot = 0
        self.current_player = 0
        self.small_blind_position = 0
        self.stage = 'preflop'
        self.deck = self._create_deck()
        self._deal_initial_hands()
        self.history = []

    def get_history(self):
        return self.history

    def __getstate__(self):
        state = self.__dict__.copy()
        state['players'] = [player.__dict__.copy() for player in self.players]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.players = [Player(p['stack']) for p in state['players']]
        for i, p in enumerate(state['players']):
            self.players[i].__dict__.update(p)

    def to_string(self):
        return f"Player: {self.current_player}, Pot: {self.pot}, Stage: {self.stage}, Cards: {self.community_cards}, Hands: {[p.hand for p in self.players]}"

    def reset(self):
        self.__init__(self.config)
        self.small_blind_position = (self.small_blind_position + 1) % self.config.NUM_PLAYERS
        self._post_blinds()
        self.history = []
        return self

    def clone(self):
        new_state = GameState(self.config)
        new_state.players = [player.clone() for player in self.players]
        new_state.community_cards = self.community_cards.copy()
        new_state.pot = self.pot
        new_state.current_player = self.current_player
        new_state.small_blind_position = self.small_blind_position
        new_state.stage = self.stage
        new_state.deck = self.deck.copy()
        return new_state

    def advance_stage(self):
        stages = ['preflop', 'flop', 'turn', 'river', 'showdown']
        current_index = stages.index(self.stage)
        self.stage = stages[current_index + 1]
        if self.stage == 'flop':
            self.community_cards.extend([self.deck.pop() for _ in range(3)])
        elif self.stage in ['turn', 'river']:
            self.community_cards.append(self.deck.pop())
        for player in self.players:
            player.has_acted = False

    def _create_deck(self):
        ranks = '23456789TJQKA'
        suits = 'hdcs'
        return [f'{r}{s}' for r in ranks for s in suits]

    def _deal_initial_hands(self):
        random.shuffle(self.deck)
        for player in self.players:
            player.hand = [self.deck.pop() for _ in range(2)]

    def get_valid_actions(self):
        valid_actions = [Action.FOLD, Action.CALL]
        if self.players[self.current_player].stack > self.config.BIG_BLIND:
            valid_actions.append(Action.RAISE)
        return valid_actions

    def apply_action(self, action):
        current_player = self.players[self.current_player]
        if action == Action.FOLD:
            current_player.is_active = False
        elif action == Action.CALL:
            call_amount = max(player.current_bet for player in self.players) - current_player.current_bet
            current_player.stack -= call_amount
            current_player.current_bet += call_amount
            current_player.total_bet += call_amount
            self.pot += call_amount
        elif action == Action.RAISE:
            raise_amount = self.get_min_raise()
            current_player.stack -= raise_amount
            current_player.current_bet += raise_amount
            current_player.total_bet += raise_amount
            self.pot += raise_amount

        current_player.has_acted = True
        self._move_to_next_player()

        if self._should_advance_stage():
            self._advance_stage()

        done = self.is_terminal()
        rewards = self.get_payoff() if done else [0] * len(self.players)

        # Add the current state, action, reward, next state, and done to the history
        next_state = self.clone()
        self.history.append((self.vectorize(), action, rewards[self.current_player], next_state.vectorize(), done))
        return self, rewards, done

    def _next_player(self):
        self.current_player = (self.current_player + 1) % self.config.NUM_PLAYERS
        while len(self.players[self.current_player].hand) == 0 or self.players[self.current_player].stack == 0:
            self.current_player = (self.current_player + 1) % self.config.NUM_PLAYERS

    def _next_stage(self):
        stages = ['preflop', 'flop', 'turn', 'river', 'showdown']
        self.stage = stages[stages.index(self.stage) + 1]
        if self.stage == 'flop':
            self.community_cards.extend([self.deck.pop() for _ in range(3)])
        elif self.stage in ['turn', 'river']:
            self.community_cards.append(self.deck.pop())

    def vectorize(self):
        player_data = np.array([(p.stack / self.config.STARTING_STACK, int(p.is_active)) for p in self.players]).flatten()
        community_cards = np.zeros(10)
        for i, card in enumerate(self.community_cards):
            rank, suit = card[0], card[1]
            community_cards[i * 2] = "23456789TJQKA".index(rank) / 12
            community_cards[i * 2 + 1] = "hdcs".index(suit) / 3
        pot = np.array([self.pot / self.config.STARTING_STACK])
        stage = np.zeros(5)
        stage[['preflop', 'flop', 'turn', 'river', 'showdown'].index(self.stage)] = 1
        return np.concatenate([player_data, community_cards, pot, stage])

    def _encode_stage(self):
        stages = ['preflop', 'flop', 'turn', 'river', 'showdown']
        return [1 if self.stage == stage else 0 for stage in stages]

    def is_terminal(self):
        return self.stage == 'showdown' or len([p for p in self.players if p.stack > 0]) <= 1

    def _post_blinds(self):
        sb_pos = self.small_blind_position
        bb_pos = (sb_pos + 1) % self.config.NUM_PLAYERS
        self.players[sb_pos].stack -= self.config.SMALL_BLIND
        self.players[bb_pos].stack -= self.config.BIG_BLIND
        self.pot = self.config.SMALL_BLIND + self.config.BIG_BLIND
        self.current_player = (bb_pos + 1) % self.config.NUM_PLAYERS

    def _end_betting_round(self):
        if self.stage != 'showdown':
            self._next_stage()
        self.current_player = 0

    def determine_winner(self):
        active_players = [i for i, player in enumerate(self.players) if player.is_active]
        if len(active_players) == 1:
            return [active_players[0]]

        player_hands = [(i, self._evaluate_hand(self.players[i].hand + self.community_cards)) for i in active_players]
        best_hand = max(player_hands, key=lambda x: x[1])
        winners = [i for i, hand in player_hands if hand == best_hand[1]]
        return winners

    def _evaluate_hand(self, cards):
        ranks = '23456789TJQKA'
        suits = 'hdcs'

        hand_ranks = [ranks.index(card[0]) for card in cards]
        hand_suits = [suits.index(card[1]) for card in cards]

        is_flush = len(set(hand_suits)) == 1
        is_straight = len(set(hand_ranks)) == 5 and max(hand_ranks) - min(hand_ranks) == 4

        rank_counts = Counter(hand_ranks)
        count_values = sorted(rank_counts.values(), reverse=True)

        hand_strength = (
            (8, max(hand_ranks)) if is_straight and is_flush else
            (7, rank_counts.most_common(1)[0][0]) if count_values == [4, 1] else
            (6, rank_counts.most_common(1)[0][0]) if count_values == [3, 2] else
            (5, max(hand_ranks)) if is_flush else
            (4, max(hand_ranks)) if is_straight else
            (3, rank_counts.most_common(1)[0][0]) if count_values[0] == 3 else
            (2, max(r for r, c in rank_counts.items() if c == 2)) if count_values[:2] == [2, 2] else
            (1, rank_counts.most_common(1)[0][0]) if count_values[0] == 2 else
            (0, max(hand_ranks))
        )
        return hand_strength

    def _estimate_hand_strength(self):
        if not self.community_cards:
            return self._estimate_preflop_strength()

        my_hand = self.players[self.current_player].hand
        all_cards = my_hand + self.community_cards
        my_score = self._evaluate_hand(all_cards)

        other_cards = [card for card in self.deck if card not in all_cards]
        possible_hands = list(itertools.combinations(other_cards, 5 - len(self.community_cards)))

        wins = 0
        total = 0
        for hand in possible_hands:
            opponent_score = self._evaluate_hand(list(hand) + self.community_cards)
            if my_score > opponent_score:
                wins += 1
            total += 1

        return wins / total if total > 0 else 0.5

    def _estimate_preflop_strength(self):
        hand = self.players[self.current_player].hand
        if not hand or len(hand) != 2:
            return 0.5  # Return a default value if the hand is not valid

        ranks = '23456789TJQKA'
        rank1, rank2 = ranks.index(hand[0][0]), ranks.index(hand[1][0])
        suited = hand[0][1] == hand[1][1]

        if rank1 == rank2:
            return 0.5 + rank1 * 0.03
        elif suited:
            return 0.4 + (rank1 + rank2) * 0.01
        else:
            return 0.3 + (rank1 + rank2) * 0.01

    def determine_winner(self):
        active_players = [i for i, player in enumerate(self.players) if player.is_Active()]
        if len(active_players) == 1:
            return [active_players[0]]

        player_hands = [(i, self._evaluate_hand(self.players[i].hand + self.community_cards)) for i in active_players]
        best_hand = max(player_hands, key=lambda x: x[1])
        winners = [i for i, hand in player_hands if hand == best_hand[1]]
        return winners

    def get_payoff(self):
        if not self.is_terminal():
            raise ValueError("Game is not over yet")

        winners = self.determine_winner()
        payoffs = np.zeros(len(self.players), dtype=np.float32)

        for i, player in enumerate(self.players):
            if i in winners:
                payoffs[i] = (self.pot / len(winners) - player.total_bet) / self.config.STARTING_STACK
            else:
                payoffs[i] = -player.total_bet / self.config.STARTING_STACK

        return payoffs

    def _move_to_next_player(self):
        self.current_player = (self.current_player + 1) % self.config.NUM_PLAYERS
        while not self.players[self.current_player].is_Active():
            self.current_player = (self.current_player + 1) % self.config.NUM_PLAYERS

    def get_min_raise(self):
        return max(self.config.BIG_BLIND, max(player.current_bet for player in self.players) * 2)

    def _should_advance_stage(self):
        return all(player.has_acted or not player.is_Active() for player in self.players)

    def _advance_stage(self):
        stages = ['preflop', 'flop', 'turn', 'river', 'showdown']
        current_index = stages.index(self.stage)
        if current_index < len(stages) - 1:
            self.stage = stages[current_index + 1]
            if self.stage == 'flop':
                self.community_cards.extend([self.deck.pop() for _ in range(3)])
            elif self.stage in ['turn', 'river']:
                self.community_cards.append(self.deck.pop())
            for player in self.players:
                player.has_acted = False
            self.current_player = self.small_blind_position
        else:
            self.stage = 'showdown'
