import torch

# Data
TEXT_FILE_PATH = "models/training_data/goethe.txt"

# Device
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# =============================================================================
# BIGRAM MODEL HYPERPARAMETERS
# =============================================================================
BIGRAM_BATCH_SIZE = 32
BIGRAM_BLOCK_SIZE = 8
BIGRAM_MAX_ITERS = 5000
BIGRAM_EVAL_INTERVAL = 300
BIGRAM_LEARNING_RATE = 1e-3
BIGRAM_EVAL_ITERS = 200

# =============================================================================
# GPT MODEL HYPERPARAMETERS
# =============================================================================
GPT_BATCH_SIZE = 32
GPT_BLOCK_SIZE = 128
GPT_MAX_ITERS = 5000
GPT_EVAL_INTERVAL = 300
GPT_LEARNING_RATE = 3e-4
GPT_EVAL_ITERS = 200

# GPT Architecture
GPT_N_EMBED = 192
GPT_N_HEADS = 4
GPT_N_LAYER = 4
GPT_DROPOUT = 0.5

# =============================================================================
# BACKWARDS COMPATIBILITY (used by bigram/)
# =============================================================================
BATCH_SIZE = BIGRAM_BATCH_SIZE
BLOCK_SIZE = BIGRAM_BLOCK_SIZE
MAX_ITERS = BIGRAM_MAX_ITERS
EVAL_INTERVAL = BIGRAM_EVAL_INTERVAL
LEARNING_RATE = BIGRAM_LEARNING_RATE
EVAL_ITERS = BIGRAM_EVAL_ITERS
