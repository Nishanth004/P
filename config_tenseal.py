# FEDMED_TenSEAL/config_tenseal.py

import logging

# --- General ---
SEED = 42
DEVICE = "cuda"  # "cuda" or "cpu" (Use "cuda" if available and PyTorch is CUDA-enabled)

# --- Data ---
DATA_PATH = "data/smoking.csv"
TEST_SPLIT_RATIO = 0.2
LABEL_COLUMN = "smoking"
CATEGORICAL_COLS = ['gender', 'oral', 'tartar'] # Confirm if 'hearing(left)', 'hearing(right)', 'Urine protein' are categorical

# --- Federated Learning ---
NUM_CLIENTS = 10
NUM_ROUNDS = 100  # Increased for better convergence observation
FRACTION_CLIENTS_PER_ROUND = 1.0 # All clients participate each round
LOCAL_EPOCHS = 5
BATCH_SIZE = 32

# --- Model (MLP) ---
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.2
HIDDEN_SIZES = [64, 32]

# --- Homomorphic Encryption (TenSEAL CKKS) ---
POLY_MODULUS_DEGREE = 8192
COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]
GLOBAL_SCALE = 2**40

# --- Quality Assessment & Adaptive Aggregation ---
QUALITY_SCORE_EPSILON = 1e-6
USE_ROBUST_QUALITY_AGGREGATION = False # Keep this enabled
QUALITY_SCORE_CLIP_PERCENTILE_LOWER = 5.0
QUALITY_SCORE_CLIP_PERCENTILE_UPPER = 95.0

# --- Robustness Evaluation (Enable these for this run) ---
FRACTION_NOISY_CLIENTS = 0.5  # e.g., 20% noisy clients
NOISE_LEVEL = 0.2            # 20% label flipping for noisy clients

FRACTION_ADVERSARIAL_CLIENTS = 0.2 # e.g., 20% adversarial clients (distinct from noisy)
ATTACK_TYPE = "model_poisoning_opposite"
ATTACK_SCALE = -1.5 # Standard scale for this attack

# --- Class Imbalance ---
CALCULATE_POS_WEIGHT = True

# --- Logging ---
DETAILED_SIM_LOG_LEVEL = logging.DEBUG
LOG_LEVEL = logging.INFO # Keep as INFO for full runs to avoid overly verbose logs
LOG_FILE = "fedmed_tenseal_NO_CLIP_HIGH_NOISY_FRAC.log" # Specific log file name for this run
WEBAPP_LOG_FILE = "fedmed_webapp.log"

# --- Plotting ---
PLOT_RESULTS = True
PLOT_FILENAME = "fedmed_tenseal_NO_CLIP_HIGH_NOISY_FRAC.png" # Specific plot name