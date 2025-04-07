# client/main.py
import requests
import time
import logging
import numpy as np
import json # Added

from .config import SERVER_URL, CLIENT_ID, ENABLE_QKD_SIMULATION, FEATURE_COUNT # Import QKD flag
from . import crypto_manager
from . import data_simulator
from . import local_trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- State ---
current_round = -1
expected_feature_count = -1
derived_qkd_key = None # Store result of QKD sim

# --- Modified Registration Function ---
def register_with_server():
    """Registers the client with the orchestrator server. Optionally performs QKD simulation."""
    global expected_feature_count, derived_qkd_key
    url = f"{SERVER_URL}/register"
    payload = {'client_id': CLIENT_ID}
    alice_bits, alice_bases = None, None # Store QKD state

    # --- QKD Simulation (Client Step 1: Prepare) ---
    if ENABLE_QKD_SIMULATION:
        logger.info("QKD Simulation: Generating client's (Alice's) bits and bases...")
        alice_bits, alice_bases, alice_bases_str = crypto_manager.generate_qkd_client_initial_state()
        payload['qkd_alice_bases'] = alice_bases_str # Send bases to server
        logger.info("QKD Simulation: Sending bases to server.")

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status() # Raise HTTPError for bad responses
        data = response.json()

        # --- QKD Simulation (Client Step 2: Reconcile) ---
        bob_bases_str = data.get('qkd_bob_bases')
        if ENABLE_QKD_SIMULATION and bob_bases_str is not None:
            logger.info("QKD Simulation: Received Bob's bases. Reconciling key...")
            derived_qkd_key = crypto_manager.simulate_qkd_client_protocol(bob_bases_str, alice_bits, alice_bases)
            if derived_qkd_key:
                 logger.info(f"QKD Simulation: Client derived shared key. This key would decrypt the HE public key if it were encrypted.")
                 # In a real implementation:
                 # encrypted_he_key = data.get('encrypted_he_public_key')
                 # serialized_he_key = decrypt_with_aes_gcm(encrypted_he_key, derived_qkd_key)
                 # he_pub_key_data = json.loads(serialized_he_key)
                 # crypto_manager.set_public_key(he_pub_key_data)
            else:
                 logger.error("QKD Simulation: Failed to derive shared key on client side.")
                 return False # Treat QKD failure as registration failure? Or proceed without?
        elif ENABLE_QKD_SIMULATION:
             logger.warning("QKD Simulation: Server did not return Bob's bases. Cannot complete QKD.")
             # Decide how to handle - maybe registration fails if QKD is mandatory

        # Set HE Public Key (using the potentially unencrypted key from server in this simulation)
        he_pub_key_data = data.get('public_key')
        if not he_pub_key_data:
            logger.error("Registration failed: Server did not provide HE public key.")
            return False
        crypto_manager.set_public_key(he_pub_key_data) # Pass the dict directly


        # Get expected feature count
        expected_feature_count = data.get('feature_count', -1)
        if expected_feature_count == -1 :
            logger.warning("Server did not provide expected feature count.")
        else:
            if expected_feature_count != FEATURE_COUNT:
                 logger.error(f"FATAL: Feature count mismatch! Server expects {expected_feature_count}, client configured for {FEATURE_COUNT}.")
                 return False
        logger.info(f"Successfully registered Client {CLIENT_ID}. Server expects {expected_feature_count} features.")
        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to register with server at {url}: {e}")
        return False
    except Exception as e:
        logger.error(f"An error occurred during registration: {e}", exc_info=True)
        return False

# --- get_global_model (Unchanged) ---
def get_global_model(round_num):
    # ... (no changes needed) ...
    url = f"{SERVER_URL}/get_model"
    params = {'client_id': CLIENT_ID, 'round': round_num}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if data['round'] != round_num:
             logger.warning(f"Received model for round {data['round']}, expected {round_num}. Skipping update.")
             return None
        logger.info(f"Received global model weights for round {round_num}.")
        return np.array(data['weights'])
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get global model from server: {e}")
        return None
    except Exception as e:
        logger.error(f"An error occurred while fetching model: {e}")
        return None

# --- submit_encrypted_update (Unchanged) ---
def submit_encrypted_update(round_num, encrypted_update_json):
    # ... (no changes needed) ...
    url = f"{SERVER_URL}/submit_update"
    payload = {
        'client_id': CLIENT_ID,
        'round': round_num,
        'update': encrypted_update_json
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        logger.info(f"Successfully submitted encrypted update for round {round_num}.")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to submit update to server: {e}")
        return False
    except Exception as e:
        logger.error(f"An error occurred during update submission: {e}")
        return False

# --- run_client_round (Unchanged internally, relies on successful registration) ---
def run_client_round():
    # ... (no changes needed in the core logic) ...
    global current_round

    logger.info(f"Attempting to fetch model for round {current_round}...")
    global_weights = get_global_model(current_round)
    if global_weights is None:
        logger.warning(f"Could not get global model for round {current_round}. Skipping this round.")
        return

    logger.info("Generating local security data...")
    X_local, y_local = data_simulator.generate_data(CLIENT_ID)
    if X_local.shape[1] != expected_feature_count:
         logger.error(f"FATAL: Generated data feature count {X_local.shape[1]} doesn't match expected {expected_feature_count}.")
         return

    logger.info("Training local model...")
    start_train_time = time.time()
    weight_difference = local_trainer.train_local_model(global_weights, X_local, y_local, CLIENT_ID)
    train_time = time.time() - start_train_time
    logger.info(f"Local training finished in {train_time:.4f} seconds.")

    logger.info("Encrypting model update...")
    start_encrypt_time = time.time()
    try:
        encrypted_update_json = crypto_manager.encrypt_vector(weight_difference)
        encrypt_time = time.time() - start_encrypt_time
        logger.info(f"Encryption finished in {encrypt_time:.4f} seconds.")
    except ValueError as e:
        logger.error(f"Encryption failed: {e}. Cannot submit update.")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred during encryption: {e}")
        return

    logger.info("Submitting encrypted update to server...")
    submit_encrypted_update(current_round, encrypted_update_json)


# --- main_loop (Unchanged internally, relies on successful registration) ---
def main_loop():
    # ... (no changes needed) ...
    global current_round
    if not register_with_server(): # Registration now includes QKD simulation attempt
        logger.error("Client registration failed. Exiting.")
        return

    server_rounds = 5
    from .config import SERVER_URL
    r_check_url = f"{SERVER_URL}/get_model"
    processed_rounds = set()

    while True:
        try:
            params = {'client_id': CLIENT_ID, 'round': -1}
            response = requests.get(r_check_url, params=params, timeout=10)
            server_round = -1
            if response.status_code == 400 and "current is" in response.text:
                 try:
                     msg = response.json().get("error", "")
                     server_round = int(msg.split("current is ")[1].split(")")[0])
                 except Exception:
                      logger.warning("Could not parse current round from server response.")
                      time.sleep(10)
                      continue
            elif response.status_code == 200:
                 server_round = response.json().get("round", -1)
            else:
                 logger.warning(f"Unexpected response ({response.status_code}) when checking server round.")
                 time.sleep(10)
                 continue

        except requests.exceptions.Timeout:
             logger.warning("Timeout checking server round.")
             time.sleep(15)
             continue
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not connect to server to check round: {e}")
            time.sleep(15)
            continue

        if server_round > current_round and server_round not in processed_rounds:
            logger.info(f"Detected new server round: {server_round}")
            current_round = server_round
            run_client_round()
            processed_rounds.add(current_round)
        elif server_round == -1 :
             logger.info("Server not yet started or finished rounds. Waiting.")
        else:
            pass

        if len(processed_rounds) >= server_rounds:
             logger.info("Finished participating in all rounds.")
             break

        time.sleep(5)


if __name__ == '__main__':
    logger.info(f"Starting Client: {CLIENT_ID}")
    main_loop()
    logger.info(f"Client {CLIENT_ID} shutting down.")