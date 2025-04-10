# server/config.py
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 5000
NUM_ROUNDS = 5
CLIENTS_PER_ROUND = 2 # Number of clients expected (informative, not strictly enforced here)
# Minimum clients needed to proceed with aggregation/update.
# Set to 1 to allow updates even with a single client.
MIN_CLIENTS_FOR_AGGREGATION = 1
HE_KEY_SIZE = 1024 # Bits for Paillier key pair (use >=2048 in production)
MODEL_FEATURE_COUNT = 10 # Number of features in the security data
PRECISION_FACTOR = 10**6 # Factor to scale floats to integers for HE

# --- Simulated Autonomous Action ---
THREAT_THRESHOLD = 0.8 # If avg threat probability > threshold, trigger action