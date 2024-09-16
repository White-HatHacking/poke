import os
import torch
import multiprocessing as mp
from game.actions import Action


class Config:
    # System settings
    NUM_PROCESSES = mp.cpu_count()  # Typically 8 or 10 on M1 Pro
    MAX_PROCESSES = mp.cpu_count() - 1  # Leave one core for system processes
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Reporting settings
    DETAILED_REPORTING = False
    VERBOSE = False
    MODEL_SAVE_DIR = os.path.join(os.getcwd(), 'saved_models')

    # Performance settings
    RENDER_TRAINING = False

    # Game settings
    NUM_PLAYERS = 2
    STARTING_STACK = 10000
    SMALL_BLIND = 50
    BIG_BLIND = 100
    ANTE = 10

    # Training settings
    NUM_TRAINING_ITERATIONS = 5000000
    SELF_PLAY_ITERATIONS = 10000
    BATCH_SIZE = 2048  # Increased for M1 Pro
    MEMORY_SIZE = 1000000  # Increased for 32GB RAM
    PRIORITIZED_MEMORY_SIZE = 500000
    SAVE_INTERVAL = 20000
    EVALUATION_EPISODES = 100
    EVALUATION_INTERVAL = 1000
    ELO_K_FACTOR = 32
    INITIAL_ELO = 1500

    # Prioritized Experience Replay
    USE_PRIORITIZED_REPLAY = True
    PER_ALPHA = 0.6
    PER_BETA = 0.4
    PER_BETA_INCREMENT = 0.0001

    # Neural Network settings
    INPUT_SIZE = (NUM_PLAYERS * 2) + 10 + 1 + 5  # Player data + community cards + pot + stage
    HIDDEN_SIZE = 512  # Increased for better model capacity
    NUM_LAYERS = 6  # Increased for deeper network
    OUTPUT_SIZE = len(Action)
    LEARNING_RATE = 0.0005
    DROPOUT_RATE = 0.2
    TARGET_UPDATE_FREQUENCY = 0.01
    MAX_GRAD_NORM = 1.0
    ATTENTION_HEADS = 8  # Increased for better attention mechanism

    # Optimizer settings
    LR_PATIENCE = 5
    LR_FACTOR = 0.5
    GAMMA = 0.99

    # Exploration settings
    INITIAL_EPSILON = 1.0
    EPSILON_DECAY = 0.99995
    EPSILON_MIN = 0.01

    # MCTS settings
    MCTS_SIMULATIONS = 2000  # Increased for better search
    MCTS_EXPLORATION_CONSTANT = 1.5

    # GTO solver settings
    GTO_EPSILON = 0.05
    GTO_DISCOUNT_FACTOR = 0.99
    GTO_UCB_CONSTANT = 1.0
    GTO_ITERATIONS = 10000  # Increased for better convergence
    GTO_UPDATE_FREQUENCY = 0.001
    GTO_UPDATE_ITERATIONS = 100
    GTO_PRUNE_THRESHOLD = 0.01

    # Opponent modeling settings
    USE_OPPONENT_MODELING = True
    OPPONENT_MODEL_UPDATE_FREQUENCY = 5000
    OPPONENT_MODEL_DECAY_FACTOR = 0.99

    # Self-play settings
    PREVIOUS_OPPONENT_PROB = 0.2
    MAX_PREVIOUS_AGENTS = 20  # Increased for more diverse opponents

    # Rendering settings
    RENDER = False

    # Priority Replay settings
    PRIORITY_EPSILON = 0.01
    PRIORITY_ALPHA = 0.6
    PRIORITY_BETA = 0.4
    PRIORITY_BETA_INCREMENT = 0.0001

    # Neural Network weights
    NN_WEIGHT = 0.4
    GTO_WEIGHT = 0.3
    SEARCH_WEIGHT = 0.2
    OPPONENT_MODEL_WEIGHT = 0.1

    # Training settings
    UPDATES_PER_EPISODE = 20  # Increased for more frequent updates
    PRUNE_INTERVAL = 5000

    # Adaptive learning parameters
    ADAPTIVE_LEARNING_RATE = True
    ADAPTIVE_EPSILON = True
    ADAPTIVE_BATCH_SIZE = True
    ADAPTIVE_UPDATE_FREQUENCY = True

    ADAPTIVE_LR_FACTOR = 0.5
    ADAPTIVE_LR_PATIENCE = 3
    MIN_LEARNING_RATE = 1e-6

    # Learning rate adaptation
    LR_DECAY = 0.9995
    LR_MIN = 1e-6

    # Epsilon adaptation
    EPSILON_DECAY_MIN = 0.99995
    EPSILON_DECAY_MAX = 0.99999

    # Batch size adaptation
    BATCH_SIZE_MIN = 1024
    BATCH_SIZE_MAX = 4096
    BATCH_SIZE_INCREASE_RATE = 1.005

    # Update frequency adaptation
    UPDATE_FREQUENCY = 20
    UPDATE_FREQUENCY_MIN = 10
    UPDATE_FREQUENCY_MAX = 40
    UPDATE_FREQUENCY_INCREASE_RATE = 1.001

    # Efficiency settings
    LR_STEP_SIZE = 1000  # Increased for less frequent LR updates
    LR_GAMMA = 0.95
    GRADIENT_ACCUMULATION_STEPS = 4  # Increased for larger effective batch size

    # Opponent modeling weights
    OVERALL_WEIGHT = 0.3
    STAGE_WEIGHT = 0.2
    STACK_WEIGHT = 0.2
    ML_WEIGHT = 0.3
    STACK_BUCKET_SIZE = 1000
    RF_N_ESTIMATORS = 200  # Increased for better random forest model
    RF_MAX_DEPTH = 15  # Increased for deeper trees

    @classmethod
    def update_adaptive_parameters(cls, agent_performance):
        if cls.ADAPTIVE_LEARNING_RATE:
            cls.LEARNING_RATE = max(cls.LEARNING_RATE * cls.LR_DECAY, cls.LR_MIN)

        if cls.ADAPTIVE_EPSILON:
            performance_factor = min(max(agent_performance, 0), 1)
            cls.EPSILON_DECAY = cls.EPSILON_DECAY_MIN + (cls.EPSILON_DECAY_MAX - cls.EPSILON_DECAY_MIN) * performance_factor

        if cls.ADAPTIVE_BATCH_SIZE:
            cls.BATCH_SIZE = min(int(cls.BATCH_SIZE * cls.BATCH_SIZE_INCREASE_RATE), cls.BATCH_SIZE_MAX)

        if cls.ADAPTIVE_UPDATE_FREQUENCY:
            cls.UPDATE_FREQUENCY = min(int(cls.UPDATE_FREQUENCY * cls.UPDATE_FREQUENCY_INCREASE_RATE), cls.UPDATE_FREQUENCY_MAX)

    @classmethod
    def update_for_quick_testing(cls):
        if not os.path.exists(cls.MODEL_SAVE_DIR):
            os.makedirs(cls.MODEL_SAVE_DIR)
        cls.NUM_TRAINING_ITERATIONS = 10000
        cls.SELF_PLAY_ITERATIONS = 1000
        cls.BATCH_SIZE = 512
        cls.MEMORY_SIZE = 100000
        cls.PRIORITIZED_MEMORY_SIZE = 50000
        cls.SAVE_INTERVAL = 1000
        cls.EVALUATION_INTERVAL = 500
        cls.HIDDEN_SIZE = 256
        cls.NUM_LAYERS = 4
        cls.LEARNING_RATE = 0.001
        cls.TARGET_UPDATE_FREQUENCY = 0.1
        cls.EPSILON_DECAY = 0.999
        cls.MCTS_SIMULATIONS = 500
        cls.GTO_ITERATIONS = 1000
        cls.GTO_UPDATE_FREQUENCY = 0.01
        cls.OPPONENT_MODEL_UPDATE_FREQUENCY = 500
        cls.MAX_PREVIOUS_AGENTS = 10
        cls.NUM_PROCESSES = min(4, mp.cpu_count())
        cls.PRIORITY_BETA_INCREMENT = 0.001
        cls.UPDATES_PER_EPISODE = 10
        cls.GTO_UPDATE_ITERATIONS = 50
        cls.RENDER_TRAINING = True
        cls.UPDATE_FREQUENCY = 10
        cls.LR_STEP_SIZE = 200
        cls.GRADIENT_ACCUMULATION_STEPS = 2

    @classmethod
    def optimize_for_speed(cls):
        if not cls.DETAILED_REPORTING and not cls.VERBOSE:
            cls.EVALUATION_INTERVAL = 2000
            cls.SAVE_INTERVAL = 50000
            cls.MCTS_SIMULATIONS = 1000
            cls.GTO_ITERATIONS = 5000
            cls.OPPONENT_MODEL_UPDATE_FREQUENCY = 10000
            cls.USE_OPPONENT_MODELING = False
            cls.GRADIENT_ACCUMULATION_STEPS = 1
            cls.UPDATE_FREQUENCY = 40
            cls.PRUNE_INTERVAL = 10000
            cls.BATCH_SIZE = 4096
            cls.NUM_PROCESSES = mp.cpu_count()
            cls.USE_PRIORITIZED_REPLAY = False
            cls.ADAPTIVE_LEARNING_RATE = False
            cls.ADAPTIVE_EPSILON = False
            cls.ADAPTIVE_BATCH_SIZE = False
            cls.ADAPTIVE_UPDATE_FREQUENCY = False
            cls.RENDER = False
            cls.RENDER_TRAINING = False
