import torch


class Config:
    # Game
    BOARD_SIZE = 15
    WIN_LENGTH = 6

    # Network — bigger for GPU
    NUM_RES_BLOCKS = 6
    NUM_CHANNELS   = 128

    # MCTS
    NUM_SIMULATIONS  = 400
    C_PUCT           = 1.5
    DIRICHLET_ALPHA  = 0.15
    DIRICHLET_EPS    = 0.25
    LEAF_BATCH_SIZE  = 16   # leaves evaluated per NN call; set to 1 to disable batching
    HEURISTIC_WEIGHT = 0.3  # threat-prior boost for cold-start; 0 to disable

    # Self-play
    NUM_SELF_PLAY_GAMES = 100
    TEMP_THRESHOLD      = 40    # use temp=1 for first N moves, then greedy
    MAX_GAME_MOVES      = 500
    CAPTURE_REWARD_ALPHA = 0.15  # blend weight for strong-capture reward shaping

    # Playout cap randomization (KataGo): fast search for 75% of positions,
    # full search for 25%; only full-search positions become policy targets.
    PLAYOUT_CAP_RATIO  = 0.25   # fraction of positions using full sims
    FAST_SIMULATIONS   = 50     # sims for non-target positions

    # Opponent pool: fraction of games played vs a past checkpoint.
    OPPONENT_POOL_RATIO = 0.50  # fraction of games vs heuristic/old model
    OPPONENT_POOL_SIZE  = 8     # max past checkpoints to keep in pool

    # Heuristic opponent: replace checkpoint pool with alpha-beta bot.
    # Requires connect6s_heuristic_cpp (bash scripts/build_heuristic.sh).
    HEURISTIC_OPP       = True  # use heuristic bot instead of checkpoint pool
    HEURISTIC_OPP_DEPTH = 4     # alpha-beta depth (4 ≈ 50-100ms/move)

    # Replay buffer
    REPLAY_BUFFER_SIZE = 300_000   # ≥ 4 iters of data at 400 sims × 100 games × 8 aug

    # Training
    BATCH_SIZE       = 512
    VALUE_LOSS_WEIGHT = 2.0
    LR               = 5e-4          # lower LR prevents catastrophic forgetting
    LR_DECAY   = 0.1
    LR_DECAY_ITERS = 100
    L2_REG     = 1e-4
    NUM_EPOCHS = 15            # more epochs to close KL gap vs MCTS targets
    # Cap batches per epoch to ~1 pass over newly collected data, not full replay buffer.
    # 100 games × ~100 moves × 8 aug / BATCH_SIZE ≈ 150 (small) / 40 (large)
    BATCHES_PER_EPOCH = 150

    # Evaluation
    NUM_EVAL_GAMES        = 20
    WIN_RATIO_THRESHOLD   = 0.55
    FORCE_ACCEPT_ITERS    = 0    # always accept new model for first N iterations

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
