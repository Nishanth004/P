# server/main.py
from flask import Flask, request, jsonify, Response
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import sys # For exiting

from . import crypto_manager
from . import model_manager
# Import MIN_CLIENTS_FOR_AGGREGATION which is now set to 1
from .config import SERVER_HOST, SERVER_PORT, NUM_ROUNDS, CLIENTS_PER_ROUND, MIN_CLIENTS_FOR_AGGREGATION

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- Server State ---
client_registry = {} # client_id -> last_seen
round_updates = {} # round_number -> {client_id: encrypted_update_json}
current_round = -1 # Start at -1 before initialization
global_model = None
server_lock = threading.Lock() # To protect shared state
server_state = "INITIALIZING" # States: INITIALIZING, RUNNING, FINISHED
fl_process_finished_event = threading.Event() # To signal Flask thread to potentially exit

# Thread pool for handling concurrent tasks like decryption/aggregation
executor = ThreadPoolExecutor(max_workers=4) # Adjust worker count based on server resources

def initialize_server():
    """Initialize server components."""
    global global_model, server_state, current_round
    logger.info("Initializing orchestrator server...")
    crypto_manager.generate_keys()
    global_model = model_manager.load_model()
    server_state = "RUNNING" # Ready to start rounds
    current_round = 0 # Set initial round
    logger.info("Server initialized and RUNNING.")

@app.route('/register', methods=['POST'])
def register_client():
    """Allows clients to register and get the public key."""
    # Allow registration even if server is finishing, clients might join late
    # if server_state != "RUNNING":
    #     return jsonify({"error": f"Server not in RUNNING state (current: {server_state})"}), 503

    client_id = request.json.get('client_id')
    if not client_id:
        return jsonify({"error": "Client ID required"}), 400

    with server_lock:
        client_registry[client_id] = time.time()

    pub_key = crypto_manager.get_public_key()
    pub_key_json = {'n': str(pub_key.n)}

    logger.info(f"Client {client_id} registered.")
    return jsonify({
        "message": "Registered successfully",
        "public_key": pub_key_json,
        "feature_count": model_manager.MODEL_FEATURE_COUNT
        })

@app.route('/status', methods=['GET'])
def get_status():
    """Provides the current server state and round number."""
    with server_lock:
         state_info = {
             "state": server_state,
             "current_round": current_round,
             "num_rounds_configured": NUM_ROUNDS
         }
    return jsonify(state_info)

@app.route('/get_model', methods=['GET'])
def get_model():
    """Provides the current global model parameters to clients for a specific round."""
    if server_state != "RUNNING":
         # Still return state info even if not running, useful for client loop
         return jsonify({"error": f"Server not in RUNNING state for model distribution (current: {server_state})", "state": server_state, "current_round": current_round}), 400 # Use 400 to indicate bad request *for a model*

    client_id = request.args.get('client_id')
    request_round = request.args.get('round', type=int)

    with server_lock: # Lock needed to access client_registry and current_round safely
        if not client_id or client_id not in client_registry:
            return jsonify({"error": "Client not registered or invalid ID", "state": server_state, "current_round": current_round}), 403
        # Check if the requested round matches the server's current round
        if request_round is None or request_round != current_round:
            return jsonify({"error": f"Requesting model for wrong round ({request_round}), server is in round {current_round}", "state": server_state, "current_round": current_round}), 400

    # Load model weights (no lock needed if model_manager is thread-safe, which joblib load is)
    model_weights = model_manager.get_model_weights(global_model)
    logger.info(f"Sending model (round {current_round}) to client {client_id}")

    return jsonify({
        "round": current_round,
        "weights": model_weights.tolist(), # Send weights as a list
        "state": server_state # Include state for consistency
        })

@app.route('/submit_update', methods=['POST'])
def submit_update():
    """Receives encrypted model updates from clients."""
    data = request.json
    client_id = data.get('client_id')
    round_num = data.get('round')
    encrypted_update_json = data.get('update') # This is already JSON string from client

    with server_lock: # Lock needed to access registry, round_updates, current_round
        if server_state != "RUNNING":
            return jsonify({"error": f"Server not accepting updates (state: {server_state})", "state": server_state, "current_round": current_round}), 503 # Service Unavailable

        if not client_id or client_id not in client_registry:
            return jsonify({"error": "Client not registered or invalid ID", "state": server_state, "current_round": current_round}), 403
        if round_num is None or round_num != current_round:
            # Log the mismatch but return a more informative error
            logger.warning(f"Update from {client_id} rejected: Submitted for round {round_num}, server is in round {current_round}.")
            return jsonify({"error": f"Update submitted for wrong round ({round_num}), server is in round {current_round}", "state": server_state, "current_round": current_round}), 400
        if not encrypted_update_json:
            return jsonify({"error": "Encrypted update missing", "state": server_state, "current_round": current_round}), 400

        # Initialize round updates if it's the first update for this round
        if current_round not in round_updates:
            round_updates[current_round] = {}

        # Only accept one update per client per round
        if client_id in round_updates[current_round]:
            logger.warning(f"Client {client_id} already submitted update for round {current_round}. Ignoring duplicate.")
            # Return success even for duplicate to avoid client confusion, but log it.
            return jsonify({"message": "Update already received for this round", "state": server_state}), 200

        # Store the update
        round_updates[current_round][client_id] = encrypted_update_json
        num_received = len(round_updates[current_round])
        logger.info(f"Received encrypted update from {client_id} for round {current_round}. Total updates this round: {num_received}")

        # Check if enough updates received (>=1) - mainly for logging now
        if num_received >= MIN_CLIENTS_FOR_AGGREGATION: # MIN_CLIENTS_FOR_AGGREGATION is now 1
             logger.info(f"Minimum updates ({MIN_CLIENTS_FOR_AGGREGATION}) reached for round {current_round}. Aggregation/update will proceed at round end.")

    return jsonify({"message": "Update received successfully", "state": server_state})

# Add a handler for favicon.ico to avoid 404 errors in logs
@app.route('/favicon.ico')
def favicon():
    return Response(status=204) # No Content response

def run_federated_round(round_num):
    """Manages a single round of federated learning."""
    global global_model, current_round
    logger.info(f"--- Starting Federated Round {round_num} ---")

    with server_lock:
        # Ensure current_round is updated correctly for this round
        if current_round != round_num:
             logger.warning(f"Round mismatch detected in run_federated_round. Expected {round_num}, was {current_round}. Forcing update.")
             current_round = round_num
        # Clear previous round updates if necessary (should be handled by dict keying, but safety)
        round_updates[current_round] = {}

    # Wait for clients to fetch the model and submit updates
    # Update log message to reflect MIN_CLIENTS_FOR_AGGREGATION = 1
    logger.info(f"Waiting for client updates for round {round_num} (min {MIN_CLIENTS_FOR_AGGREGATION} needed)...")

    round_start_time = time.time()
    wait_time_seconds = 60 # Wait up to 60 seconds for updates

    proceed_to_aggregation = False
    while time.time() - round_start_time < wait_time_seconds:
        with server_lock:
             num_received = len(round_updates.get(current_round, {}))
        # --- MODIFIED CONDITION ---
        # Check if at least one client submitted
        if num_received >= MIN_CLIENTS_FOR_AGGREGATION: # MIN_CLIENTS_FOR_AGGREGATION is now 1
             logger.info(f"Round {current_round}: Minimum {num_received} updates received. Proceeding with aggregation/update.")
             proceed_to_aggregation = True
             break
        # Check if the overall server process is trying to shut down (e.g., via external signal)
        if fl_process_finished_event.is_set():
             logger.warning(f"Round {current_round}: Shutdown signal received. Terminating round early.")
             return False # Indicate premature termination

        time.sleep(5) # Check every 5 seconds
    else: # Loop finished without breaking (timeout)
        with server_lock:
            num_received = len(round_updates.get(current_round, {}))
        # --- MODIFIED CONDITION ---
        # Check if timeout occurred AND we didn't get even one update
        if num_received < MIN_CLIENTS_FOR_AGGREGATION: # MIN_CLIENTS_FOR_AGGREGATION is now 1
             logger.warning(f"Round {current_round}: Timeout reached. Received only {num_received}/{MIN_CLIENTS_FOR_AGGREGATION} updates. Skipping aggregation/update.")
             proceed_to_aggregation = False
        else:
             # Timeout reached, but we got at least one update, so proceed
             logger.info(f"Round {current_round}: Timeout reached. Proceeding with {num_received} updates.")
             proceed_to_aggregation = True


    # --- Aggregation and Update ---
    if not proceed_to_aggregation:
        logger.info(f"--- Federated Round {round_num} Complete (No updates processed) ---")
        return True # Indicate round finished normally, but skipped update


    # Proceed only if we received at least one update
    with server_lock:
        # Re-acquire lock to get the final list of updates safely
        updates_to_process = round_updates.get(current_round, {})
        num_updates = len(updates_to_process)
        logger.info(f"Round {current_round} update period ended. Processing {num_updates} updates.")

        # --- MODIFIED CONDITION (redundant due to proceed_to_aggregation check, but safe) ---
        if num_updates < MIN_CLIENTS_FOR_AGGREGATION: # Should not happen if proceed_to_aggregation is True
            logger.warning(f"Round {current_round}: Aggregation condition unmet ({num_updates} < {MIN_CLIENTS_FOR_AGGREGATION}). This shouldn't happen here. Skipping.")
            return True # Continue to next round

        # --- System Programming Aspect: Concurrent Aggregation/Decryption ---
        log_prefix = "Aggregating" if num_updates > 1 else "Processing"
        logger.info(f"{log_prefix} and decrypting tasks submitted to executor...")
        future = executor.submit(aggregate_and_decrypt, list(updates_to_process.values()), num_updates)

        # Wait for the result
        try:
            aggregated_decrypted_updates = future.result(timeout=120) # Timeout for decryption/processing

            if aggregated_decrypted_updates is not None: # Check for None explicitly
                logger.info("Aggregation/Decryption complete. Updating global model.")
                global_model = model_manager.update_global_model(aggregated_decrypted_updates, num_updates)

                # --- Autonomous Action Trigger ---
                logger.info("Evaluating model and checking for autonomous actions...")
                model_manager.evaluate_model_and_trigger_action(global_model)
            else:
                logger.error("Aggregation/decryption failed or returned no result. Skipping model update.")

        except Exception as e:
            logger.error(f"Error during aggregation/decryption task execution: {e}", exc_info=True)
            # Decide how to handle - skip round? retry? For now, skip update.
            logger.warning(f"Skipping model update for round {current_round} due to aggregation/decryption error.")

    logger.info(f"--- Federated Round {round_num} Complete (Processed {num_updates} update(s)) ---")
    return True # Indicate round finished normally

def aggregate_and_decrypt(encrypted_updates_list, num_clients):
    """
    Aggregates encrypted updates (if num_clients > 1) and decrypts the result.
    If num_clients is 1, it simply decrypts the single update.
    """
    try:
        aggregated_encrypted_json = None
        if num_clients == 1:
            # No aggregation needed, just take the first (only) update
            logger.info("Processing 1 encrypted vector (no aggregation needed)...")
            aggregated_encrypted_json = encrypted_updates_list[0]
        else:
            # 1. Aggregate Encrypted Updates
            logger.info(f"Aggregating {num_clients} encrypted vectors...")
            start_agg_time = time.time()
            aggregated_encrypted_json = crypto_manager.aggregate_encrypted_vectors(encrypted_updates_list)
            agg_time = time.time() - start_agg_time
            if aggregated_encrypted_json is None:
                logger.error("Aggregation resulted in None.")
                return None
            logger.info(f"Aggregation complete ({agg_time:.4f} seconds). Size: ~{len(aggregated_encrypted_json)/1024:.2f} KB")

        # 2. Decrypt the Result (either single or aggregated)
        logger.info("Decrypting vector...")
        start_decrypt_time = time.time()
        aggregated_decrypted_updates = crypto_manager.decrypt_vector(aggregated_encrypted_json)
        decrypt_time = time.time() - start_decrypt_time

        # Check if decryption yielded expected results (e.g., correct length)
        expected_length = model_manager.MODEL_FEATURE_COUNT + 1 # features + intercept
        if not isinstance(aggregated_decrypted_updates, list) or len(aggregated_decrypted_updates) != expected_length:
             logger.error(f"Decryption error: Result vector length or type mismatch. Expected list of length {expected_length}, got {type(aggregated_decrypted_updates)} of length {len(aggregated_decrypted_updates) if isinstance(aggregated_decrypted_updates, list) else 'N/A'}")
             return None

        logger.info(f"Decryption complete ({decrypt_time:.4f} seconds). Vector length: {len(aggregated_decrypted_updates)}")
        return aggregated_decrypted_updates
    except Exception as e:
         logger.error(f"Error in aggregate_and_decrypt: {e}", exc_info=True)
         return None


def run_server():
    global server_state, current_round
    initialize_server() # This sets state to RUNNING and current_round to 0

    # Start Flask server in a background thread
    flask_thread = threading.Thread(target=lambda: app.run(host=SERVER_HOST, port=SERVER_PORT, threaded=True), daemon=True)
    flask_thread.start()
    logger.info(f"Flask server running on http://{SERVER_HOST}:{SERVER_PORT}")

    # Give Flask a moment to start up
    time.sleep(1)

    # Run federated learning rounds
    for r in range(NUM_ROUNDS):
        with server_lock:
            current_round = r # Set current round for this iteration
        success = run_federated_round(r)
        if not success: # Check if the round was terminated early
            logger.warning("Federated learning process terminated early.")
            break
        # Optional: Add delay between rounds
        time.sleep(5)

    logger.info("Federated learning process finished.")
    with server_lock:
        server_state = "FINISHED"
        # current_round remains at the last completed round number
        logger.info(f"Server state set to FINISHED. Last round attempted: {current_round}")


    # Keep the Flask server running to allow clients to fetch status, or implement clean shutdown
    logger.info("Server main logic complete. Flask thread remains active for status checks. Press Ctrl+C to exit.")
    # Keep alive loop
    try:
        while True:
             time.sleep(60)
    except KeyboardInterrupt:
         logger.info("Server process interrupted by user (Ctrl+C). Shutting down.")
         executor.shutdown(wait=False) # Don't wait for ongoing tasks
         sys.exit(0)



if __name__ == '__main__':
    run_server()