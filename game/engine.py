import numpy as np
from game.state import GameState
from game.actions import Action
from typing import List, Tuple
import logging


class PokerEngine:
    def __init__(self, config):
        self.config = config
        self.current_game_state = GameState(config)
        self.num_players = config.NUM_PLAYERS
        self.starting_stack = config.STARTING_STACK
        self.small_blind = config.SMALL_BLIND
        self.big_blind = config.BIG_BLIND
        self.ante = config.ANTE
        self.current_hand = 0

    def get_game_history(self):
        return self.current_game_state.get_history()

    def __getstate__(self):
        state = self.__dict__.copy()
        # del state['_render']  # Remove the rendering function as it's not picklable
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._render = None  # Set rendering function to None when unpickling

    def reset(self) -> GameState:
        self.current_game_state = self.current_game_state.reset()
        self.current_hand += 1
        return self.current_game_state

    def step(self, action):
        self.current_game_state.apply_action(action)
        rewards = self._calculate_rewards()
        done = self.current_game_state.is_terminal()
        return self.current_game_state, rewards, done

    def _calculate_rewards(self):
        rewards = np.zeros(self.config.NUM_PLAYERS, dtype=np.float32)
        if self.current_game_state.is_terminal():
            payoffs = self.current_game_state.get_payoff()
            for i, payoff in enumerate(payoffs):
                rewards[i] = payoff
        else:
            # Intermediate rewards based on pot equity
            for i, player in enumerate(self.current_game_state.players):
                if player.is_Active():
                    hand_strength = self.current_game_state._estimate_hand_strength()
                    rewards[i] = (hand_strength * self.current_game_state.pot - player.total_bet) / self.config.STARTING_STACK

        logging.debug(f"Debug: Calculated rewards: {rewards}")
        return rewards

    def _should_advance_stage(self) -> bool:
        return all(player.has_acted or not player.is_Active() for player in self.current_game_state.players)

    def _advance_stage(self):
        self.current_game_state.advance_stage()

    def get_valid_actions(self) -> List[Action]:
        return self.current_game_state.get_valid_actions()

    def is_terminal(self) -> bool:
        return self.current_game_state.is_terminal()

    def get_current_player(self) -> int:
        return self.current_game_state.current_player

    def get_state(self) -> GameState:
        return self.current_game_state

    def render(self):
        print("\n" + "=" * 50)
        print("Poker Game State")
        print("=" * 50)

        # Display community cards
        community_cards = self.current_game_state.community_cards
        print(f"Community Cards: {' '.join(str(card) for card in community_cards)}")

        # Display pot size
        print(f"Pot: ${self.current_game_state.pot}")

        # Display current stage
        print(f"Current Stage: {self.current_game_state.stage}")

        # Display player information
        for i, player in enumerate(self.current_game_state.players):
            print(f"\nPlayer {i + 1}:")
            if player.is_Active():
                print(f"  Hand: {' '.join(str(card) for card in player.hand)}")
            else:
                print("  Hand: [Folded]")
            print(f"  Stack: ${player.stack}")
            print(f"  Bet: ${player.current_bet}")

        # Display current player's turn
        current_player = self.current_game_state.current_player
        print(f"\nCurrent Player: Player {current_player + 1}")

        # Display valid actions for the current player
        valid_actions = self.get_valid_actions()
        print("Valid Actions:", ", ".join(str(action) for action in valid_actions))

        print("=" * 50 + "\n")

    def play_game(self, agents):
        state = self.reset()
        done = False
        while not done:
            self.render()  # Render the game state before each action
            current_player = self.get_current_player()
            action = agents[current_player].get_action(state)
            state, rewards, done = self.step(action)
        self.render()  # Render the final game state
        print("Game Over!")
        return rewards

    def play_vs_human(self, agent):
        state = self.reset()
        while not state.is_terminal():
            self.render()
            if state.current_player == 0:  # Human player
                action = self._get_human_action(state)
            else:  # AI player
                action, _ = agent.get_action(state)
            state, _, _ = self.step(action)
        self.render()
        print("Game over!")

    def _get_human_action(self, state):
        valid_actions = state.get_valid_actions()
        while True:
            action_input = input("Enter your action (f: fold, c: call, r: raise): ").lower()
            if action_input == 'f' and Action.FOLD in valid_actions:
                return Action.FOLD
            elif action_input == 'c' and Action.CALL in valid_actions:
                return Action.CALL
            elif action_input == 'r' and Action.RAISE in valid_actions:
                raise_amount = int(input("Enter raise amount: "))
                return (Action.RAISE, raise_amount)
            else:
                print("Invalid action. Try again.")
