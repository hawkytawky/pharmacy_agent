import torch

# Data
TEXT_FILE_PATH = "models/training_data/goethe.txt"

# Hyperparameters
BATCH_SIZE = 32
BLOCK_SIZE = 8
MAX_ITERS = 10000
EVAL_INTERVAL = 300
LEARNING_RATE = 1e-3
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EVAL_ITERS = 200
