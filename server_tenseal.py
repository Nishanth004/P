# FEDMED_TenSEAL/server_tenseal.py (For Step 8b: FEDMED No Clip)

import torch
import torch.nn as nn
import numpy as np
import logging
import tenseal as ts
import base64
# from scipy import stats # Not needed if USE_ROBUST_QUALITY_AGGREGATION is False

from utils_tenseal import (create_tenseal_context,
                           deserialize_ckks_vector, 
                           get_model_params_vector, set_model_params_vector)
# Config import - USE_ROBUST_QUALITY_AGGREGATION will be False for this run (Step 8b)
from config_tenseal import (USE_ROBUST_QUALITY_AGGREGATION, 
                            QUALITY_SCORE_CLIP_PERCENTILE_LOWER, # Used only if MAD fallback occurs
                            QUALITY_SCORE_CLIP_PERCENTILE_UPPER, # Used only if MAD fallback occurs
                            DEVICE)

class ServerTenSEAL:
    def __init__(self, model_template, input_dim, num_clients, test_loader, pos_weight=None):
        self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        self.model = model_template(input_dim=input_dim).to(self.device)
        self.global_model_params_vector = get_model_params_vector(self.model)
        
        self.num_clients = num_clients
        self.test_loader = test_loader
        
        logging.info("Server: Creating TenSEAL CKKS context using create_tenseal_context from utils...")
        self.tenseal_context = create_tenseal_context()
        logging.info("Server: TenSEAL context created and keys generated.")

        temp_context_for_public_serialization = ts.context_from(self.tenseal_context.serialize())
        temp_context_for_public_serialization.make_context_public(generate_galois_keys=False, generate_relin_keys=False) 
        self.public_tenseal_context_bytes = temp_context_for_public_serialization.serialize(
            save_public_key=True,
            save_secret_key=False,
            save_galois_keys=False,
            save_relin_keys=False
        )
        logging.info(f"Server: Serialized public TenSEAL context for clients (size: {len(self.public_tenseal_context_bytes)} bytes).")

        self.pos_weight = None
        if pos_weight is not None:
            self.pos_weight = torch.tensor([pos_weight], device=self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def get_public_tenseal_context_bytes(self):
        return self.public_tenseal_context_bytes

    def get_global_model_params_vector(self):
        return self.global_model_params_vector.copy()

    def aggregate_updates(self, client_updates):
        logging.info("Server: Using FEDMED Quality-Aware Aggregation.")

        if not client_updates:
            logging.warning("Server: No client updates received for aggregation.")
            return

        valid_updates = [upd for upd in client_updates if upd is not None and upd[0] is not None]
        if not valid_updates:
            logging.warning("Server: No valid client updates after filtering. Skipping aggregation.")
            return
        
        serialized_encrypted_deltas_b64 = [upd[0] for upd in valid_updates]
        quality_scores = np.array([upd[1] for upd in valid_updates])
        num_samples_list = np.array([upd[2] for upd in valid_updates])

        # For Step 8b, USE_ROBUST_QUALITY_AGGREGATION from config_tenseal.py will be False.
        # Therefore, this 'if' block for any form of clipping (Percentile or MAD) will be SKIPPED.
        if USE_ROBUST_QUALITY_AGGREGATION and len(quality_scores) > 1:
            # This block contains the logic for robust aggregation (e.g., MAD or Percentile)
            # Which specific one depends on the version of server_tenseal.py being used.
            # For Step 8b, this whole block is bypassed due to config setting.
            # For completeness, if one were to put a specific clipping here for a different experiment:
            logging.error("Server: Robust aggregation is True, but this server version path is for No-Clip setup. Config/Server mismatch for intended experiment.")
            # As a safety, just use raw scores if this contradictory state is reached.
            clipped_scores = quality_scores
        else: 
            # This path is taken when USE_ROBUST_QUALITY_AGGREGATION is False (as in Step 8b)
            # or if there are not enough scores for robust aggregation.
            clipped_scores = quality_scores # Use raw scores directly
            logging.info(f"Server: Quality scores (Robust aggregation OFF or not enough scores): {clipped_scores}")

        sum_clipped_scores = np.sum(clipped_scores)
        if sum_clipped_scores > 1e-9:
            weights = clipped_scores / sum_clipped_scores
        else: 
            logging.warning("Server (FEDMED Path): Sum of quality scores is near zero. Falling back to sample-based/equal weights.")
            total_samples_in_round = np.sum(num_samples_list)
            if total_samples_in_round > 0:
                weights = num_samples_list / total_samples_in_round
            elif len(valid_updates) > 0:
                weights = np.ones(len(valid_updates)) / len(valid_updates)
            else:
                logging.warning("Server (FEDMED Path): No valid updates to determine weights. Aggregation cannot proceed.")
                return
        
        logging.info(f"Server: Aggregation weights (based on raw quality scores): {weights}")
        
        num_params = len(self.global_model_params_vector)
        aggregated_delta_encrypted = ts.ckks_vector(self.tenseal_context, np.zeros(num_params, dtype=np.float64))

        for i, ser_enc_delta_client_b64 in enumerate(serialized_encrypted_deltas_b64):
            try:
                enc_delta_client = deserialize_ckks_vector(self.tenseal_context, ser_enc_delta_client_b64)
                weight = weights[i] 
                weighted_delta_client = enc_delta_client * weight
                aggregated_delta_encrypted = aggregated_delta_encrypted + weighted_delta_client
            except Exception as e:
                logging.error(f"Server: Error processing client update {i} (delta preview: '{ser_enc_delta_client_b64[:30]}...'): {e}. Skipping this update.")
                continue
        
        logging.debug("Server: Decrypting aggregated model delta with TenSEAL...")
        if not isinstance(aggregated_delta_encrypted, ts.CKKSVector):
            logging.error("Server: aggregated_delta_encrypted is not a valid CKKSVector. Cannot decrypt.")
            return

        decrypted_aggregated_delta_list = aggregated_delta_encrypted.decrypt()
        if decrypted_aggregated_delta_list is None:
            logging.error("Server: Decryption of aggregated delta returned None. HE parameters likely need adjustment. Model not updated.")
            return
            
        decrypted_aggregated_delta = np.array(decrypted_aggregated_delta_list, dtype=np.float32)
        
        if np.isnan(decrypted_aggregated_delta).any() or np.isinf(decrypted_aggregated_delta).any():
            logging.error("Server: Decrypted aggregated delta contains NaN/Inf values. Model not updated.")
            return 

        logging.debug(f"Server: TenSEAL decryption complete. Norm of aggregated delta: {np.linalg.norm(decrypted_aggregated_delta):.4f}")

        self.global_model_params_vector += decrypted_aggregated_delta
        set_model_params_vector(self.model, self.global_model_params_vector)
        logging.info("Server: Global model updated.")

    def evaluate_global_model(self):
        self.model.eval()
        total_loss = 0.0; correct = 0; total_samples = 0
        all_targets = []; all_predictions = []
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data); loss = self.criterion(output, target)
                total_loss += loss.item() * data.size(0)
                predicted_probs = torch.sigmoid(output); predicted_labels = (predicted_probs > 0.5).float()
                correct += (predicted_labels == target).sum().item(); total_samples += target.size(0)
                all_targets.extend(target.cpu().numpy()); all_predictions.extend(predicted_labels.cpu().numpy())
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        accuracy = correct / total_samples if total_samples > 0 else 0.0
        from sklearn.metrics import f1_score, precision_score, recall_score
        f1 = f1_score(all_targets, all_predictions, zero_division=0)
        precision = precision_score(all_targets, all_predictions, zero_division=0)
        recall = recall_score(all_targets, all_predictions, zero_division=0)
        logging.info(f"Server: Global model evaluation - Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        return {"loss": avg_loss, "accuracy": accuracy, "f1_score": f1, "precision": precision, "recall": recall}