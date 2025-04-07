# client/data_simulator.py - improved version
import numpy as np
import logging
from .config import NUM_SAMPLES, FEATURE_COUNT, THREAT_RATIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_data(client_id):
    """
    Generates synthetic security log data for a client with clearer patterns.
    Each client might have slightly different data characteristics.
    """
    # Simulate slight variations based on client ID hash
    np.random.seed(hash(client_id) % (2**32 - 1)) # Seed based on client ID for consistency per client

    num_threats = int(NUM_SAMPLES * THREAT_RATIO)
    num_normal = NUM_SAMPLES - num_threats

    # Generate normal data (e.g., lower values for certain features)
    normal_data = np.random.rand(num_normal, FEATURE_COUNT) * 0.4  # Scale normal data features
    
    # Generate threat data with stronger signal
    threat_data = np.random.rand(num_threats, FEATURE_COUNT) * 0.3
    # Make threat features much more distinctive (last 3 features)
    threat_data[:, -3:] = 0.7 + np.random.rand(num_threats, 3) * 0.3
    
    # Add client-specific bias (example: shift one feature slightly)
    bias_feature_index = hash(client_id + "bias") % FEATURE_COUNT
    bias_amount = (hash(client_id + "amount") % 100 / 500.0) - 0.1 # Small bias +/-
    normal_data[:, bias_feature_index] += bias_amount / 2
    threat_data[:, bias_feature_index] += bias_amount

    # Combine data
    X = np.vstack((normal_data, threat_data))
    y = np.concatenate((np.zeros(num_normal, dtype=int), np.ones(num_threats, dtype=int)))

    # Shuffle data
    indices = np.arange(NUM_SAMPLES)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    logger.info(f"Client {client_id}: Generated {NUM_SAMPLES} samples ({num_threats} threats, {num_normal} normal).")
    logger.info(f"Sample threat features mean: {np.mean(threat_data[:, -3:]):.4f}, normal features mean: {np.mean(normal_data[:, -3:]):.4f}")
    
    return X, y
