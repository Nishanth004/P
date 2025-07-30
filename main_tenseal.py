# FEDMED_TenSEAL/main_tenseal.py

import logging
import numpy as np
import random
import torch

# Important: Ensure config_tenseal and other _tenseal modules are imported
from config_tenseal import *
from data_loader import load_and_preprocess_data # Reusable
from models import get_model_template # Reusable
from client_tenseal import ClientTenSEAL
from server_tenseal import ServerTenSEAL
from utils_tenseal import set_seeds, plot_metrics, get_model_params_vector, set_model_params_vector

def main():
    logging.basicConfig(level=LOG_LEVEL,
                        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
                        handlers=[logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler()])
    
    set_seeds(SEED)
    logging.info(f"FEDMED-TenSEAL Simulation Started with SEED: {SEED}")
    logging.info(f"Device: {DEVICE if torch.cuda.is_available() else 'cpu'}")

    client_dataloaders, test_loader, input_dim, pos_weight_value, preprocessor = load_and_preprocess_data()
    
    if not CALCULATE_POS_WEIGHT:
        pos_weight_value = None 
        logging.info("Using unweighted loss (CALCULATE_POS_WEIGHT is False or pos_weight is None).")

    if input_dim == 0 or not any(loader is not None for loader in client_dataloaders):
        logging.error("Failed to load data or no data for clients. Exiting.")
        return

    server = ServerTenSEAL(model_template=get_model_template,
                           input_dim=input_dim,
                           num_clients=NUM_CLIENTS,
                           test_loader=test_loader,
                           pos_weight=pos_weight_value)
    
    # Get serialized public context components for clients
    public_tenseal_context_bytes = server.get_public_tenseal_context_bytes()

    clients = []
    all_client_indices = list(range(NUM_CLIENTS))
    random.shuffle(all_client_indices)
    num_noisy_clients = int(FRACTION_NOISY_CLIENTS * NUM_CLIENTS)
    noisy_client_indices = set(all_client_indices[:num_noisy_clients])
    remaining_indices = all_client_indices[num_noisy_clients:]
    num_adversarial_clients = int(FRACTION_ADVERSARIAL_CLIENTS * NUM_CLIENTS)
    if len(remaining_indices) < num_adversarial_clients:
        num_adversarial_clients = len(remaining_indices)
    adversarial_client_indices = set(remaining_indices[:num_adversarial_clients])

    logging.info(f"Total clients: {NUM_CLIENTS}")
    logging.info(f"Noisy clients ({len(noisy_client_indices)}): {sorted(list(noisy_client_indices))}")
    logging.info(f"Adversarial clients ({len(adversarial_client_indices)}): {sorted(list(adversarial_client_indices))}")

    for i in range(NUM_CLIENTS):
        is_noisy = i in noisy_client_indices
        is_adversary = i in adversarial_client_indices
        
        client = ClientTenSEAL(client_id=i,
                               train_loader=client_dataloaders[i],
                               model_template=lambda: get_model_template(input_dim=input_dim),
                               tenseal_public_context_bytes=public_tenseal_context_bytes,
                               is_noisy=is_noisy,
                               is_adversary=is_adversary,
                               pos_weight=pos_weight_value)
        clients.append(client)

    active_clients = [c for c in clients if c.train_loader is not None]
    if not active_clients:
        logging.error("No active clients with data. Federation cannot proceed. Exiting.")
        return

    metrics_history = []
    for round_num in range(1, NUM_ROUNDS + 1):
        logging.info(f"--- Round {round_num}/{NUM_ROUNDS} ---")
        
        global_model_params_vector = server.get_global_model_params_vector()
        client_updates = []

        num_selected_clients = max(1, int(FRACTION_CLIENTS_PER_ROUND * len(active_clients)))
        selected_client_indices = np.random.choice(len(active_clients), num_selected_clients, replace=False)
        
        logging.info(f"Selected {num_selected_clients} clients for round {round_num}: {[active_clients[i].client_id for i in selected_client_indices]}")

        for client_idx in selected_client_indices:
            client = active_clients[client_idx]
            logging.debug(f"Processing client {client.client_id}...")
            client.set_global_model_params(global_model_params_vector) # Sends float32 numpy array
            # Train returns: serialized_encrypted_delta_b64, quality_score, num_samples
            update_package = client.train() 
            
            if update_package is not None and update_package[0] is not None:
                 client_updates.append(update_package)
            else:
                logging.warning(f"Client {client.client_id} did not return a valid update.")
        
        if not client_updates:
            logging.warning(f"Round {round_num}: No client updates collected. Skipping aggregation. Evaluating current model.")
            round_metrics = server.evaluate_global_model()
            round_metrics['round'] = round_num
            metrics_history.append(round_metrics)
            continue

        server.aggregate_updates(client_updates)
        
        round_metrics = server.evaluate_global_model()
        round_metrics['round'] = round_num
        metrics_history.append(round_metrics)

    logging.info("--- Federated Training Complete (TenSEAL) ---")

    if PLOT_RESULTS and metrics_history:
        plot_metrics(metrics_history, PLOT_FILENAME)

    logging.info("--- Demonstrating New Client Prediction (TenSEAL Model) ---")
    if test_loader and len(test_loader.dataset) > 0:
        final_global_model_params = server.get_global_model_params_vector()
        prediction_model = get_model_template(input_dim=input_dim).to(server.device)
        set_model_params_vector(prediction_model, final_global_model_params)
        prediction_model.eval()
        sample_data, sample_label = next(iter(test_loader))
        single_sample_data = sample_data[0:1].to(server.device)
        single_sample_label = sample_label[0:1].item()
        logging.info(f"Predicting for a sample with true label: {single_sample_label}")
        with torch.no_grad():
            output = prediction_model(single_sample_data)
            prediction_prob = torch.sigmoid(output).item()
            predicted_class = (prediction_prob > 0.5)
        logging.info(f"Raw output: {output.item():.4f}, Prediction probability (smoker): {prediction_prob:.4f}, Predicted class: {int(predicted_class)}")
    else:
        logging.warning("Test loader empty. Skipping new client prediction demonstration.")

    logging.info("FEDMED-TenSEAL Simulation Finished.")

if __name__ == "__main__":
    main()