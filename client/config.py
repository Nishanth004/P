# client/config.py
import random # For CLIENT_ID generation

SERVER_URL = 'http://127.0.0.1:5000'
CLIENT_ID = f"cloud_env_{random.randint(1000, 9999)}" # Simulate unique client ID per instance

# Data Simulation Config
NUM_SAMPLES = 500
FEATURE_COUNT = 10 # MUST MATCH SERVER's model_manager.MODEL_FEATURE_COUNT
THREAT_RATIO = 0.1

# Training Config
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.01

PRECISION_FACTOR = 10**6 # MUST MATCH SERVER'S config.PRECISION_FACTOR

# --- Quantum-Inspired Enhancements (Client Side) ---
# Corresponds to server config, used during registration
ENABLE_QKD_SIMULATION = True # Should match server setting ideally
QKD_SIM_LENGTH = 512 # Initial number of bits/bases to generate (needs to be > server's QKD_KEY_LENGTH)