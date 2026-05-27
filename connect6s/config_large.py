from .config import Config


class ConfigLarge(Config):
    # Bigger network
    NUM_RES_BLOCKS = 8
    NUM_CHANNELS   = 192

    NUM_SIMULATIONS   = 200
    FAST_SIMULATIONS  = 80    # 80:200 = 2.5x ratio; 20 was below search-collapse
    PLAYOUT_CAP_RATIO = 0.5   # 50% full-sim positions → more policy targets
    NUM_EPOCHS        = 10
    BATCHES_PER_EPOCH = 40
    HEURISTIC_WEIGHT  = 0.3   # low enough to not override NN; model must actually learn
    BATCH_SIZE        = 2048
    VALUE_LOSS_WEIGHT = 2.0   # restore value gradient (was 1.0, collapsed to v=0.005)

    # No force-accept: let eval filter degraded models
    FORCE_ACCEPT_ITERS = 0

    NUM_WORKERS        = 12
    THREADS_PER_WORKER = 1

    WIN_RATIO_THRESHOLD  = 0.52
    NUM_EVAL_GAMES       = 40

    # 30% vs heuristic; 2x more blocking signal, v loss healthy enough (0.078)
    OPPONENT_POOL_RATIO  = 0.30
    HEURISTIC_OPP_DEPTH  = 4

    # Separate model paths so both runs coexist
    BEST_MODEL_PATH = "models/best_large.pt"
    CHECKPOINT_FMT  = "models/large_iter_{:04d}.pt"
    LOG_PATH        = "logs/train_large.log"
