# server/main.py
from flask import Flask, request, jsonify
import logging
import time
import threading
import json # Added for parsing
from concurrent.futures import ThreadPoolExecutor

from . import crypto_manager
from . import model_manager
from .config import (
    SERVER_HOST, SERVER_PORT, NUM_ROUNDS, CLIENTS_PER_ROUND,
    MIN_CLIENTS_FOR_AGGREGATION, ENABLE_QKD_SIMULATION, QKD_KEY_LENGTH
) # Import QKD config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ... (executor, global state variables remain the same) ...
client_registry = {} # client_id -> last_seen
round_updates = {} # round_number -> {client_id: encrypted_update_json}
current_round = 0
global_model = None
server_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=4)

def initialize_server():
    """Initialize server components."""
    global global_model
    logger.info("Initializing orchestrator server...")
    crypto_manager.generate_keys() # Generates HE keys
    global_model = model_manager.load_model() # Loads model & initializes QIFA velocity
    logger.info("Server initialized.")

# --- Modified Registration Endpoint for QKD Simulation ---
@app.route('/register', methods=['POST'])
def register_client():
    """Allows clients to register. Optionally performs simulated QKD."""
    data = request.json
    client_id = data.get('client_id')
    if not client_id:
        return jsonify({"error": "Client ID required"}), 400

    serialized_he_pub_key = crypto_manager.get_serialized_public_key()
    qkd_bases_bob = None
    qkd_shared_key = None # Store the derived key conceptually

    # --- Simulated QKD Exchange ---
    if ENABLE_QKD_SIMULATION:
        alice_bases_str = data.get('qkd_alice_bases')
        if alice_bases_str:
            logger.info(f"QKD Simulation: Received Alice's bases from {client_id}. Simulating Bob's side...")
            qkd_bases_bob, qkd_shared_key = crypto_manager.simulate_qkd_server_protocol(client_id, alice_bases_str)
            if qkd_shared_key:
                logger.info(f"QKD Simulation: Derived shared key for {client_id}. This key would ideally encrypt the HE public key transmission.")
                # In a real implementation:
                # encrypted_he_pub_key = encrypt_with_aes_gcm(serialized_he_pub_key, qkd_shared_key)
                # response_data['encrypted_he_public_key'] = encrypted_he_pub_key
            else:
                logger.error(f"QKD Simulation Failed for {client_id}. Proceeding without QKD protection.")
                # Fallback: Send unencrypted (as done currently) or deny registration
        else:
            logger.warning(f"Client {client_id} registered but did not provide QKD bases (QKD enabled).")
            # Decide policy: allow registration without QKD or deny? Allowing for now.

    # Store client info (simplified)
    with server_lock:
        client_registry[client_id] = {'last_seen': time.time(), 'qkd_key': qkd_shared_key}

    # Prepare response
    response_data = {
        "message": "Registered successfully",
        "public_key": json.loads(serialized_he_pub_key), # Send standard public key (simulation only)
        "feature_count": model_manager.MODEL_FEATURE_COUNT
    }
    if qkd_bases_bob is not None:
         response_data["qkd_bob_bases"] = qkd_bases_bob # Send Bob's bases for client reconciliation simulation

    logger.info(f"Client {client_id} registered.")
    return jsonify(response_data)

# --- /get_model endpoint (Unchanged) ---
@app.route('/get_model', methods=['GET'])
def get_model():
    # ... (no changes needed) ...
    client_id = request.args.get('client_id')
    request_round = request.args.get('round', type=int)
    if not client_id or client_id not in client_registry:
         return jsonify({"error": "Client not registered or invalid ID"}), 403
    if request_round != current_round:
         return jsonify({"error": f"Requesting model for wrong round ({request_round}), current is {current_round}"}), 400
    model_weights = model_manager.get_model_weights(global_model)
    logger.info(f"Sending model (round {current_round}) to client {client_id}")
    return jsonify({
        "round": current_round,
        "weights": model_weights.tolist()
        })


# --- /submit_update endpoint (Unchanged) ---
@app.route('/submit_update', methods=['POST'])
def submit_update():
    # ... (no changes needed) ...
    data = request.json
    client_id = data.get('client_id')
    round_num = data.get('round')
    encrypted_update_json = data.get('update') # This is already JSON string from client

    if not client_id or client_id not in client_registry:
        return jsonify({"error": "Client not registered or invalid ID"}), 403
    if round_num != current_round:
        return jsonify({"error": f"Update submitted for wrong round ({round_num}), current is {current_round}"}), 400
    if not encrypted_update_json:
         return jsonify({"error": "Encrypted update missing"}), 400

    with server_lock:
        if current_round not in round_updates:
            round_updates[current_round] = {}
        if client_id in round_updates[current_round]:
            logger.warning(f"Client {client_id} already submitted update for round {current_round}. Ignoring.")
            return jsonify({"message": "Update already received for this round"}), 200

        round_updates[current_round][client_id] = encrypted_update_json
        logger.info(f"Received encrypted update from {client_id} for round {current_round}. Total updates this round: {len(round_updates[current_round])}")

        if len(round_updates[current_round]) >= MIN_CLIENTS_FOR_AGGREGATION:
             logger.info(f"Minimum updates ({MIN_CLIENTS_FOR_AGGREGATION}) reached for round {current_round}. Aggregation can proceed.")

    return jsonify({"message": "Update received successfully"})


# --- Modified Federated Round Logic ---
def run_federated_round(round_num):
    """Manages a single round of federated learning."""
    global global_model, current_round
    logger.info(f"--- Starting Federated Round {round_num} ---")

    with server_lock:
        current_round = round_num
        round_updates[current_round] = {} # Clear updates for the new round

    logger.info(f"Waiting for client updates for round {round_num}...")
    round_start_time = time.time()
    wait_time_seconds = 60

    while time.time() - round_start_time < wait_time_seconds:
        with server_lock:
             num_received = len(round_updates.get(current_round, {}))
        if num_received >= MIN_CLIENTS_FOR_AGGREGATION:
             logger.info(f"Round {current_round}: Reached minimum {num_received} updates. Proceeding early.")
             break
        time.sleep(5)

    # --- Aggregation and Update ---
    with server_lock:
        updates_to_process = round_updates.get(current_round, {})
        num_updates = len(updates_to_process)
        logger.info(f"Round {current_round} ended. Received {num_updates} updates.")

        if num_updates < MIN_CLIENTS_FOR_AGGREGATION:
            logger.warning(f"Round {current_round}: Not enough updates ({num_updates}) received. Skipping model update for this round.")
            return

        # Submit aggregation and decryption to the thread pool
        logger.info("Submitting aggregation and decryption tasks to executor...")
        # Pass the list of encrypted update JSON strings
        future = executor.submit(aggregate_and_decrypt, list(updates_to_process.values()))

        try:
            # Result is the decrypted *sum* of scaled updates
            aggregated_decrypted_updates = future.result(timeout=120)

            if aggregated_decrypted_updates:
                logger.info("Aggregation and decryption complete. Updating global model.")
                # The update function now handles averaging and QIFA momentum internally
                global_model = model_manager.update_global_model(aggregated_decrypted_updates, num_updates)

                logger.info("Evaluating model and checking for autonomous actions...")
                model_manager.evaluate_model_and_trigger_action(global_model)

            else:
                logger.error("Aggregation/decryption failed or returned no result.")

        except Exception as e:
            logger.error(f"Error during aggregation/decryption task execution: {e}", exc_info=True)

    logger.info(f"--- Federated Round {round_num} Complete ---")


def aggregate_and_decrypt(encrypted_updates_list):
    """
    Aggregates encrypted updates and decrypts the sum.
    Args:
        encrypted_updates_list (list): List of JSON strings, each an encrypted vector.
    Returns:
        list: The decrypted aggregated vector (list of integers) or None on failure.
    """
    num_clients = len(encrypted_updates_list)
    if num_clients == 0: return None
    try:
        logger.info(f"Aggregating {num_clients} encrypted vectors using HE...")
        aggregated_encrypted_json = crypto_manager.aggregate_encrypted_vectors(encrypted_updates_list)
        if aggregated_encrypted_json is None:
            logger.error("HE Aggregation resulted in None.")
            return None

        logger.info("Decrypting aggregated vector...")
        start_decrypt_time = time.time()
        # Decrypts the single aggregated vector
        aggregated_decrypted_updates = crypto_manager.decrypt_vector(aggregated_encrypted_json)
        decrypt_time = time.time() - start_decrypt_time
        logger.info(f"Decryption took {decrypt_time:.4f} seconds.")

        return aggregated_decrypted_updates # Return the list of decrypted summed integers
    except Exception as e:
         logger.error(f"Error in aggregate_and_decrypt: {e}", exc_info=True)
         return None


def run_server():
    initialize_server()
    flask_thread = threading.Thread(target=lambda: app.run(host=SERVER_HOST, port=SERVER_PORT, threaded=True), daemon=True)
    flask_thread.start()
    logger.info(f"Flask server running on http://{SERVER_HOST}:{SERVER_PORT}")

    for r in range(NUM_ROUNDS):
        run_federated_round(r)
        time.sleep(5)

    logger.info("Federated learning process finished.")
    # Keep Flask running... add shutdown if needed

if __name__ == '__main__':
    run_server()