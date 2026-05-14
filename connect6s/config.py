import torch


class Config:
    # Game
    BOARD_SIZE = 15
    WIN_LENGTH = 6

    # Network — bigger for GPU
    NUM_RES_BLOCKS = 6
    NUM_CHANNELS   = 128

    # MCTS
    NUM_SIMULATIONS  = 150
    C_PUCT           = 1.5
    DIRICHLET_ALPHA  = 0.15
    DIRICHLET_EPS    = 0.5
    LEAF_BATCH_SIZE  = 16   # leaves evaluated per NN call; set to 1 to disable batching
    HEURISTIC_WEIGHT = 1.5  # threat-prior boost for cold-start; 0 to disable

    # Self-play
    NUM_SELF_PLAY_GAMES = 100
    TEMP_THRESHOLD      = 40    # use temp=1 for first N moves, then greedy
    MAX_GAME_MOVES      = 500

    # Replay buffer
    REPLAY_BUFFER_SIZE = 50_000

    # Training
    BATCH_SIZE = 512
    LR         = 5e-4          # lower LR prevents catastrophic forgetting
    LR_DECAY   = 0.1
    LR_DECAY_ITERS = 100
    L2_REG     = 1e-4
    NUM_EPOCHS = 5             # fewer epochs per iter = less overfit

    # Evaluation
    NUM_EVAL_GAMES        = 20
    WIN_RATIO_THRESHOLD   = 0.55
    FORCE_ACCEPT_ITERS    = 10   # always accept new model for first N iterations

    # Parallelism
    # Workers run on CPU; set to 1 to disable multiprocessing.
    NUM_WORKERS = 6
    # Threads each worker uses for torch/numpy ops.
    # Formula: physical_cores // NUM_WORKERS  (machine has 12 cores → 2)
    THREADS_PER_WORKER = 2

    # Device (GPU for training; workers always use CPU)
    DEVICE = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Save full checkpoint every N iterations (best.pt saved every accepted iter)
    CHECKPOINT_EVERY = 50

    # Paths
    MODEL_DIR          = "models"
    BEST_MODEL_PATH    = "models/best.pt"
    CHECKPOINT_FMT     = "models/iter_{:04d}.pt"
    LOG_PATH           = "logs/train.log"
