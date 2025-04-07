# server/model_manager.py
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import os
import logging
from .config import MODEL_FEATURE_COUNT, PRECISION_FACTOR, ENABLE_QIFA_MOMENTUM, QIFA_MOMENTUM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_FILE = 'server/global_model.pkl'
qifa_velocity = None # State variable for QIFA momentum

def initialize_global_model():
    """Initializes a new global Logistic Regression model with better initialization."""
    global qifa_velocity
    # Initialize with small random weights to break symmetry
    model = LogisticRegression(warm_start=True, solver='liblinear', max_iter=100, C=1.0)
    
    # Fitting with dummy data helps set the 'classes_' attribute
    dummy_X = np.zeros((2, MODEL_FEATURE_COUNT))
    dummy_y = np.array([0, 1]) # Ensure both classes are seen
    model.fit(dummy_X, dummy_y)
    
    # Ensure non-zero initialization of weights
    initial_weights = np.random.normal(0, 0.01, MODEL_FEATURE_COUNT)
    model.coef_ = initial_weights.reshape(1, -1)
    model.intercept_ = np.array([0.0])
    
    logger.info(f"Initialized global model with {MODEL_FEATURE_COUNT} features.")
    logger.info(f"Initial weights (random): {model.coef_}")
    
    save_model(model)
    # Initialize QIFA velocity state
    qifa_velocity = np.zeros_like(get_model_weights(model))
    logger.info("Initialized QIFA momentum velocity.")
    return model

def load_model():
    """Loads the global model from disk."""
    global qifa_velocity
    if os.path.exists(MODEL_FILE):
        try:
            model = joblib.load(MODEL_FILE)
            if hasattr(model, 'coef_') and model.coef_.shape[1] == MODEL_FEATURE_COUNT:
                 logger.info(f"Loaded global model from {MODEL_FILE}")
                 # Initialize QIFA velocity if loading existing model
                 if qifa_velocity is None:
                     qifa_velocity = np.zeros_like(get_model_weights(model))
                     logger.info("Initialized QIFA momentum velocity upon loading model.")
                 return model
            else:
                 logger.warning(f"Model structure in {MODEL_FILE} doesn't match config. Reinitializing.")
                 return initialize_global_model()
        except Exception as e:
            logger.error(f"Error loading model from {MODEL_FILE}: {e}. Reinitializing.")
            return initialize_global_model()
    else:
        logger.info(f"No existing model file found at {MODEL_FILE}. Initializing a new one.")
        return initialize_global_model()

def save_model(model):
    """Saves the global model to disk."""
    try:
        os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
        joblib.dump(model, MODEL_FILE)
        logger.info(f"Saved global model to {MODEL_FILE}")
    except Exception as e:
        logger.error(f"Error saving model to {MODEL_FILE}: {e}")


def get_model_weights(model):
    """Extracts model weights (coefficients and intercept)."""
    weights = np.concatenate([model.coef_.flatten(), model.intercept_])
    return weights

def set_model_weights(model, weights):
    """Sets model weights from a flat array."""
    coef_shape = model.coef_.shape
    intercept_shape = model.intercept_.shape
    coef_size = np.prod(coef_shape)
    intercept_size = np.prod(intercept_shape)
    if len(weights) != coef_size + intercept_size:
        raise ValueError(f"Incorrect number of weights provided. Expected {coef_size + intercept_size}, got {len(weights)}")
    model.coef_ = weights[:coef_size].reshape(coef_shape)
    model.intercept_ = weights[coef_size:].reshape(intercept_shape)
    return model

# --- QIFA Momentum Application ---
def apply_qifa_momentum(average_update):
    """
    Applies the momentum component of the Quantum-Inspired Federated Averaging.
    Operates on the already averaged update due to HE constraints.
    Updates the global qifa_velocity state.

    Args:
        average_update (np.array): The averaged weight difference vector.

    Returns:
        np.array: The momentum-adjusted update vector.
    """
    global qifa_velocity
    if not ENABLE_QIFA_MOMENTUM:
        return average_update # Return original average if QIFA is disabled

    if qifa_velocity is None:
         logger.warning("QIFA velocity not initialized. Skipping momentum.")
         return average_update

    if qifa_velocity.shape != average_update.shape:
        logger.error(f"QIFA velocity shape {qifa_velocity.shape} mismatch with average update shape {average_update.shape}. Reinitializing velocity.")
        qifa_velocity = np.zeros_like(average_update)
        # Fallback to standard average update for this round
        return average_update

    logger.info(f"Applying QIFA momentum (factor: {QIFA_MOMENTUM})...")
    # Update velocity: v = momentum * v + (1 - momentum) * average_update
    qifa_velocity = QIFA_MOMENTUM * qifa_velocity + (1 - QIFA_MOMENTUM) * average_update

    logger.info("QIFA momentum applied.")
    return qifa_velocity # Return the velocity as the adjusted update


def update_global_model(aggregated_decrypted_updates, num_clients):
    """
    Updates the global model using the averaged decrypted updates,
    optionally applying QIFA momentum.
    """
    global qifa_velocity # Ensure we can access/update the state
    if num_clients == 0:
        logger.warning("Cannot update model with zero clients.")
        return

    global_model = load_model()
    current_weights = get_model_weights(global_model)

    # Convert aggregated integer updates back to scaled float updates
    average_scaled_update = np.array(aggregated_decrypted_updates) / num_clients
    average_update = average_scaled_update / PRECISION_FACTOR

    # --- Apply QIFA Momentum (if enabled) ---
    if ENABLE_QIFA_MOMENTUM:
        final_update = apply_qifa_momentum(average_update)
    else:
        final_update = average_update # Use standard average if QIFA disabled

    # Apply the final update to the current weights
    new_weights = current_weights + final_update # Add the (potentially momentum-adjusted) average update

    # Set the updated weights back to the model
    updated_model = set_model_weights(global_model, new_weights)

    save_model(updated_model)
    logger.info(f"Global model updated using {'QIFA-momentum adjusted' if ENABLE_QIFA_MOMENTUM else 'standard'} average from {num_clients} clients.")

    return updated_model


def evaluate_model_and_trigger_action(model):
    """Simulates evaluation and autonomous action based on the model's predictions."""
    from .config import THREAT_THRESHOLD, MODEL_FEATURE_COUNT
    try:
        num_samples = 10
        # Create more distinctive potential threat data
        potential_threat_data = np.random.rand(num_samples, MODEL_FEATURE_COUNT) * 0.3
        # Last 3 features have higher values for threats
        potential_threat_data[:, -3:] *= 5  # Make threat features more pronounced
        
        probabilities = model.predict_proba(potential_threat_data)
        threat_probabilities = probabilities[:, 1]
        avg_threat_prob = np.mean(threat_probabilities)
        logger.info(f"Simulated evaluation: Average threat probability on sample data: {avg_threat_prob:.4f}")
        
        if avg_threat_prob > THREAT_THRESHOLD:
            logger.warning(f"AUTONOMOUS ACTION TRIGGERED: Average threat probability {avg_threat_prob:.4f} exceeds threshold {THREAT_THRESHOLD}.")
            print("\n *** SIMULATING AUTONOMOUS ACTION: Blocking potentially malicious source (simulated) *** \n")
    except Exception as e:
        logger.error(f"Error during model evaluation or action trigger: {e}")
