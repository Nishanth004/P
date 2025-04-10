# client/main.py
import requests
import time
import logging
import numpy as np
import sys

from .config import SERVER_URL, CLIENT_ID
from . import crypto_manager
from . import data_simulator
from . import local_trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- State ---
current_client_round = -1 # What round the client thinks it is participating in
expected_feature_count = -1
server_status = {"state": "UNKNOWN", "current_round": -1}

def register_with_server():
    """Registers the client with the orchestrator server and gets the public key."""
    global expected_feature_count
    url = f"{SERVER_URL}/register"
    try:
        response = requests.post(url, json={'client_id': CLIENT_ID}, timeout=15)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        crypto_manager.set_public_key(data['public_key'])
        expected_feature_count = data.get('feature_count', -1)
        if expected_feature_count == -1 :
            logger.warning("Server did not provide expected feature count.")
            return False # Treat missing feature count as critical
        else:
            # Validate against local config
            from .config import FEATURE_COUNT
            if expected_feature_count != FEATURE_COUNT:
                 logger.error(f"FATAL: Feature count mismatch! Server expects {expected_feature_count}, client configured for {FEATURE_COUNT}.")
                 return False # Indicate registration failure due to mismatch
            logger.info(f"Successfully registered Client {CLIENT_ID}. Server expects {expected_feature_count} features.")
        return True
    except requests.exceptions.Timeout:
        logger.error(f"Timeout during registration with server at {url}")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to register with server at {url}: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during registration: {e}", exc_info=True)
        return False

def check_server_status():
    """Checks the server's current status (state and round)."""
    global server_status
    url = f"{SERVER_URL}/status"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        server_status = response.json()
        # logger.debug(f"Server status check: {server_status}") # Optional: log status every check
        return True
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout checking server status at {url}.")
        # Keep old status, maybe server is temporarily busy
        return False
    except requests.exceptions.RequestException as e:
        logger.warning(f"Could not connect to server for status check: {e}")
        server_status = {"state": "UNREACHABLE", "current_round": -1} # Update status
        return False
    except Exception as e:
        logger.error(f"An error occurred during status check: {e}", exc_info=True)
        return False


def get_global_model(round_num):
    """Fetches the current global model weights from the server for a specific round."""
    url = f"{SERVER_URL}/get_model"
    params = {'client_id': CLIENT_ID, 'round': round_num}
    try:
        response = requests.get(url, params=params, timeout=20) # Slightly longer timeout for model download
        response.raise_for_status() # Raises HTTPError for 4xx/5xx
        data = response.json()

        # Cross-check round number (should match due to params, but good practice)
        if data.get('round') != round_num:
             logger.warning(f"Received model for round {data.get('round')}, but expected {round_num}. Skipping update.")
             return None
        # Check server state consistency
        if data.get('state') != "RUNNING":
             logger.warning(f"Server state is '{data.get('state')}' while trying to get model for round {round_num}. Aborting get_model.")
             return None

        logger.info(f"Received global model weights for round {round_num}.")
        return np.array(data['weights'])

    except requests.exceptions.HTTPError as e:
        # Handle specific errors like 400 (wrong round), 403 (not registered), 404 (not found - less likely)
        if e.response.status_code == 400:
            logger.warning(f"Server rejected get_model request for round {round_num} (Server in different round or state?). Response: {e.response.text}")
        elif e.response.status_code == 403:
             logger.error(f"Server rejected get_model request: Client {CLIENT_ID} not registered or invalid ID. Response: {e.response.text}")
        else:
            logger.error(f"HTTP error getting global model: {e.response.status_code} {e.response.text}")
        return None
    except requests.exceptions.Timeout:
        logger.error(f"Timeout getting global model from server for round {round_num}.")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get global model from server: {e}")
        return None
    except Exception as e:
        logger.error(f"An error occurred while fetching/processing model: {e}", exc_info=True)
        return None


def submit_encrypted_update(round_num, encrypted_update_json):
    """Submits the encrypted local model update to the server."""
    url = f"{SERVER_URL}/submit_update"
    payload = {
        'client_id': CLIENT_ID,
        'round': round_num,
        'update': encrypted_update_json # This is the JSON string from crypto_manager
    }
    try:
        response = requests.post(url, json=payload, timeout=30) # Longer timeout for upload
        response.raise_for_status()
        logger.info(f"Successfully submitted encrypted update for round {round_num}.")
        return True
    except requests.exceptions.HTTPError as e:
         if e.response.status_code == 400:
             logger.error(f"Server rejected update submission for round {round_num} (likely wrong round). Response: {e.response.text}")
         elif e.response.status_code == 403:
              logger.error(f"Server rejected update submission: Client {CLIENT_ID} not registered? Response: {e.response.text}")
         elif e.response.status_code == 503:
              logger.warning(f"Server rejected update submission: Not accepting updates (state might be FINISHED?). Response: {e.response.text}")
         else:
            logger.error(f"HTTP error submitting update: {e.response.status_code} {e.response.text}")
         return False
    except requests.exceptions.Timeout:
        logger.error(f"Timeout submitting update to server for round {round_num}.")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to submit update to server: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during update submission: {e}", exc_info=True)
        return False


def run_client_round(target_round):
    """Executes one round of federated learning from the client's perspective for a specific target round."""
    global current_client_round

    # 1. Get Global Model for the target round
    logger.info(f"Attempting to fetch model for round {target_round}...")
    global_weights = get_global_model(target_round)
    if global_weights is None:
        logger.warning(f"Could not get global model for round {target_round}. Skipping participation in this round.")
        return False # Indicate failure for this round

    # Validate weight dimensions against expected feature count (basic check)
    # Expected size = features + intercept (1 for binary Logistic Regression)
    expected_len = expected_feature_count + 1
    if len(global_weights) != expected_len:
         logger.error(f"Received model weights have unexpected length {len(global_weights)}, expected {expected_len}. Check model configuration consistency between server/client.")
         return False # Critical mismatch

    # 2. Generate Local Data
    logger.info("Generating local security data...")
    X_local, y_local = data_simulator.generate_data(CLIENT_ID)
    if X_local.shape[1] != expected_feature_count:
         logger.error(f"FATAL: Generated data feature count {X_local.shape[1]} doesn't match expected {expected_feature_count}.")
         return False # Critical mismatch

    # 3. Train Local Model
    logger.info("Training local model...")
    start_train_time = time.time()
    weight_difference = local_trainer.train_local_model(global_weights, X_local, y_local, CLIENT_ID)
    train_time = time.time() - start_train_time
    logger.info(f"Local training finished in {train_time:.4f} seconds.")

    # 4. Encrypt the Update (Weight Difference)
    logger.info("Encrypting model update...")
    start_encrypt_time = time.time()
    try:
        encrypted_update_json = crypto_manager.encrypt_vector(weight_difference)
        encrypt_time = time.time() - start_encrypt_time
        logger.info(f"Encryption finished in {encrypt_time:.4f} seconds.")
    except ValueError as e: # Catch errors like public key not set
        logger.error(f"Encryption failed: {e}. Cannot submit update.")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during encryption: {e}", exc_info=True)
        return False

    # 5. Submit Encrypted Update
    logger.info("Submitting encrypted update to server...")
    success = submit_encrypted_update(target_round, encrypted_update_json)
    if success:
        current_client_round = target_round # Update client's notion of its last successful round
    return success


def main_loop():
    """Main loop for the client."""
    global current_client_round

    # Initial registration
    retry_delay = 5
    max_retries = 5
    retries = 0
    while not register_with_server():
        retries += 1
        if retries > max_retries:
            logger.error("Max registration retries reached. Exiting.")
            sys.exit(1)
        logger.info(f"Registration failed, retrying in {retry_delay} seconds... ({retries}/{max_retries})")
        time.sleep(retry_delay)
        retry_delay = min(retry_delay * 2, 30) # Exponential backoff

    processed_rounds = set() # Keep track of rounds this client has successfully processed

    while True:
        # Check server status
        if not check_server_status():
            # Failed to get status, could be network issue or server down
            logger.warning("Failed to get server status. Waiting before retry...")
            time.sleep(15) # Wait longer if status check fails
            continue

        server_state = server_status.get("state", "UNKNOWN")
        server_round = server_status.get("current_round", -1)

        if server_state == "FINISHED":
            logger.info(f"Server state is FINISHED (last round was {server_round}). Client shutting down.")
            break
        elif server_state == "RUNNING":
            # Server is running, check if it's a new round we haven't processed
            if server_round > current_client_round and server_round not in processed_rounds:
                logger.info(f"Detected new server round: {server_round}. Current client round: {current_client_round}")
                success = run_client_round(server_round)
                if success:
                    processed_rounds.add(server_round)
                    logger.info(f"Successfully completed participation in round {server_round}.")
                else:
                    logger.warning(f"Failed to complete participation in round {server_round}. Will retry check later.")
                    # Optional: Add logic here if client should stop on round failure
            #else: # Server is running but same round or an older round (e.g. client restarted)
                 # logger.debug(f"Server is in round {server_round}. Client last completed {current_client_round}. Waiting for next round.")
                # pass # Already processed or waiting for server to advance
        elif server_state == "INITIALIZING":
            logger.info("Server is INITIALIZING. Waiting...")
        elif server_state == "UNREACHABLE":
             logger.warning("Server is UNREACHABLE. Waiting...")
        else: # UNKNOWN or other states
            logger.warning(f"Server in unexpected state '{server_state}'. Waiting...")

        # Check total rounds configured vs processed rounds (optional exit condition)
        num_rounds_configured = server_status.get("num_rounds_configured", -1)
        if num_rounds_configured > 0 and len(processed_rounds) >= num_rounds_configured:
             logger.info(f"Client has processed {len(processed_rounds)} rounds, matching server configuration ({num_rounds_configured}). Shutting down.")
             break

        # Wait before checking status again
        time.sleep(10) # Check every 10 seconds


if __name__ == '__main__':
    logger.info(f"Starting Client: {CLIENT_ID}")
    try:
        main_loop()
    except KeyboardInterrupt:
        logger.info("Client process interrupted by user (Ctrl+C).")
    except Exception as e:
         logger.error(f"Unhandled exception in main loop: {e}", exc_info=True)
    finally:
         logger.info(f"Client {CLIENT_ID} shutting down.")
         sys.exit(0)