# config.py

import torch

# Paths
DATA_ROOT = "data"
TRAIN_DIR = "data/Training"
TEST_DIR = "data/Testing"
MODEL_PATH = "models/best_model.pth"
CLASS_NAMES = ["class0", "class1", "class2", "class3"]  # replace with actual names

# Training hyperparameters
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEMORY = True

NUM_CLASSES = 4
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Reproducibility
SEED = 42

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
