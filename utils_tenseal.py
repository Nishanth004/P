# FEDMED_TenSEAL/utils_tenseal.py

import torch
import numpy as np
import tenseal as ts
import matplotlib.pyplot as plt
import random
import logging
import base64 # For serializing/deserializing TenSEAL objects

from config_tenseal import SEED, POLY_MODULUS_DEGREE, COEFF_MOD_BIT_SIZES, GLOBAL_SCALE

def set_seeds(seed=SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # TenSEAL doesn't have a global seed, context creation is deterministic if parameters are fixed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- TenSEAL Homomorphic Encryption Context ---
def create_tenseal_context():
    """Creates and returns a TenSEAL CKKS context."""
    try:
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=POLY_MODULUS_DEGREE,
            coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES
        )
        context.global_scale = GLOBAL_SCALE
        # Generate keys needed for various operations.
        # Relin keys are important if multiplications between ciphertexts occur,
        # or if noise from plaintext multiplication needs management via relinearization.
        # For weighted sum (Enc(v_i) * plain_w_i), relin keys might not be strictly needed
        # for the operation itself but good practice for context setup.
        context.generate_relin_keys()
        # Galois keys are for rotations/permutations on batched ciphertexts.
        # Not directly used in simple federated averaging of full vectors, but
        # having them doesn't hurt and makes the context more versatile.
        context.generate_galois_keys()
        logging.debug(f"TenSEAL CKKS context created. Poly_mod_degree: {POLY_MODULUS_DEGREE}, Scale: 2^{np.log2(GLOBAL_SCALE):.0f}")
        return context
    except Exception as e:
        logging.error(f"Error creating TenSEAL context: {e}")
        logging.error(f"  Parameters: POLY_MODULUS_DEGREE={POLY_MODULUS_DEGREE}, COEFF_MOD_BIT_SIZES={COEFF_MOD_BIT_SIZES}")
        logging.error("  Please ensure TenSEAL is installed correctly and parameters are valid for CKKS.")
        logging.error("  A common issue is the first prime in COEFF_MOD_BIT_SIZES not being congruent to 1 mod (2 * POLY_MODULUS_DEGREE).")
        logging.error("  TenSEAL (>=0.3.0) usually helps find suitable primes if you set plain_modulus appropriately and let it pick coeff_mod.")
        raise

# --- TenSEAL Encryption/Decryption & Serialization ---
def encrypt_vector_tenseal(context: ts.Context, vector: np.ndarray) -> str:
    """Encrypts a numpy vector using TenSEAL CKKSVector and serializes it to a base64 string."""
    if not isinstance(vector, np.ndarray):
        vector_np = np.array(vector, dtype=np.float64) # CKKS typically uses float64
    else:
        vector_np = vector.astype(np.float64) # Ensure float64

    encrypted_vector = ts.ckks_vector(context, vector_np)
    serialized_vector_bytes = encrypted_vector.serialize()
    return base64.b64encode(serialized_vector_bytes).decode('utf-8')

def _deserialize_from_bytes_to_ckksvector(context: ts.Context, serialized_bytes: bytes) -> ts.CKKSVector:
    """
    Helper to deserialize bytes to a CKKSVector.
    This assumes ts.CKKSVector.load(context, serialized_bytes) is the correct API.
    If this fails, the TenSEAL version might use a different API name.
    """
    try:
        # For TenSEAL versions (e.g., >= 0.3.7), .load() is a class method.
        return ts.CKKSVector.load(context=context, data=serialized_bytes)
    except Exception as e:
        logging.error(f"Failed to deserialize CKKSVector using ts.CKKSVector.load(context, bytes): {e}")
        logging.error("Ensure your TenSEAL version supports this API. Other possible APIs include factory functions like ts.ckks_vector_from_bytes(...) or different load signatures.")
        # As a last resort, one might try creating an empty vector and loading into it:
        # vec = ts.CKKSVector(context, [])
        # vec.load(serialized_bytes) # instance method
        # But this is less direct.
        raise # Re-raise to indicate deserialization failure clearly.

def decrypt_vector_tenseal(context: ts.Context, serialized_b64_vector: str) -> np.ndarray:
    """Deserializes and decrypts a TenSEAL CKKSVector from a base64 string."""
    serialized_vector_bytes = base64.b64decode(serialized_b64_vector.encode('utf-8'))
    encrypted_vector = _deserialize_from_bytes_to_ckksvector(context, serialized_vector_bytes)
    decrypted_vector_float64 = encrypted_vector.decrypt() # Decrypts to list of floats
    return np.array(decrypted_vector_float64, dtype=np.float32) # Cast to float32 for PyTorch model

def deserialize_ckks_vector(context: ts.Context, serialized_b64_vector: str) -> ts.CKKSVector:
    """Deserializes a TenSEAL CKKSVector from base64 string without immediate decryption."""
    serialized_vector_bytes = base64.b64decode(serialized_b64_vector.encode('utf-8'))
    return _deserialize_from_bytes_to_ckksvector(context, serialized_vector_bytes)


# --- Model Parameter Handling (Identical to Paillier version's utils.py) ---
def get_model_params_vector(model: torch.nn.Module) -> np.ndarray:
    """Flattens model parameters into a single numpy vector (float32)."""
    params = []
    for param in model.parameters():
        params.append(param.data.cpu().numpy().flatten())
    return np.concatenate(params).astype(np.float32)

def set_model_params_vector(model: torch.nn.Module, params_vector: np.ndarray):
    """Sets model parameters from a flat numpy vector."""
    if not isinstance(params_vector, np.ndarray):
        params_vector_np = np.array(params_vector, dtype=np.float32)
    else:
        params_vector_np = params_vector.astype(np.float32) # Ensure float32

    offset = 0
    for param in model.parameters():
        param_shape = param.data.shape
        param_size = torch.numel(param.data) # More robust way to get num elements
        
        param_slice_flat = params_vector_np[offset : offset + param_size]
        
        # Ensure the slice has the correct number of elements before reshaping
        if param_slice_flat.size != param_size:
            logging.error(f"Slice size {param_slice_flat.size} does not match parameter size {param_size} for shape {param_shape}")
            raise ValueError("Parameter slice size mismatch during model update.")
            
        param_slice_reshaped = param_slice_flat.reshape(param_shape)
        
        # Ensure dtype matches the model's parameter dtype, typically float32 for NNs
        param.data = torch.from_numpy(param_slice_reshaped).to(param.device).type(param.dtype)
        offset += param_size
        
    if offset != len(params_vector_np):
        logging.error(f"Size mismatch: params_vector has {len(params_vector_np)} elements, model requires {offset}.")
        raise ValueError("Size of params_vector does not match model structure.")

# --- Plotting (Identical to Paillier version's utils.py, with minor robustness check) ---
def plot_metrics(metrics_history: list, filename="performance.png"):
    logging.info(f"Plotting metrics to {filename}...")
    if not metrics_history or not isinstance(metrics_history, list) or not metrics_history[0]:
        logging.warning("No valid metrics data provided to plot.")
        return

    # Check if keys exist and exclude 'round'
    sample_keys = metrics_history[0].keys()
    metric_names = [key for key in sample_keys if key != 'round']
    
    if not metric_names:
        logging.warning("No metrics (other than 'round') to plot.")
        return
        
    num_metrics = len(metric_names)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, num_metrics * 4), squeeze=False)
    # squeeze=False ensures axes is always 2D, access with axes[i,0]
    
    rounds = [m['round'] for m in metrics_history]
    
    for idx, key in enumerate(metric_names):
        values = [m.get(key, float('nan')) for m in metrics_history] # Use .get for safety
        ax = axes[idx, 0]
        ax.plot(rounds, values, marker='o', linestyle='-')
        ax.set_xlabel("Communication Round")
        ax.set_ylabel(key.replace('_', ' ').title())
        ax.set_title(f"{key.replace('_', ' ').title()} vs. Round")
        ax.grid(True)
        
    plt.tight_layout()
    try:
        plt.savefig(filename)
        logging.info(f"Metrics plot saved to {filename}")
    except Exception as e:
        logging.error(f"Failed to save plot: {e}")
    # plt.show() # Optionally show plot