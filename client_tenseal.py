# FEDMED_TenSEAL/client_tenseal.py (For Step 8a: Honest Score Adversaries)

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import logging
import copy
import tenseal as ts

from utils_tenseal import get_model_params_vector, set_model_params_vector, encrypt_vector_tenseal
from config_tenseal import (LEARNING_RATE, LOCAL_EPOCHS, QUALITY_SCORE_EPSILON,
                            NOISE_LEVEL, ATTACK_TYPE, ATTACK_SCALE, DEVICE)

class ClientTenSEAL:
    def __init__(self, client_id, train_loader, model_template,
                 tenseal_public_context_bytes: bytes,
                 is_noisy=False, is_adversary=False, pos_weight=None):
        self.client_id = client_id
        self.train_loader = train_loader
        self.model_template = model_template
        
        try:
            self.tenseal_context = ts.context_from(tenseal_public_context_bytes)
            if self.tenseal_context.has_secret_key():
                logging.warning(f"Client {self.client_id}: Loaded TenSEAL context unexpectedly contains a secret key.")
        except Exception as e:
            logging.error(f"Client {self.client_id}: Failed to create TenSEAL context from public bytes: {e}")
            raise

        self.is_noisy = is_noisy
        self.is_adversary = is_adversary
        self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        
        self.model = self.model_template().to(self.device)
        
        self.pos_weight = None
        if pos_weight is not None:
            self.pos_weight = torch.tensor([pos_weight], device=self.device)
        
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
        self.initial_global_params_vector = None

    def set_global_model_params(self, global_model_params_vector):
        set_model_params_vector(self.model, global_model_params_vector)
        self.initial_global_params_vector = copy.deepcopy(global_model_params_vector)

    def _apply_label_noise(self, labels):
        if self.is_noisy and NOISE_LEVEL > 0:
            num_flips = int(NOISE_LEVEL * len(labels))
            if num_flips > 0 and len(labels) > 0:
                flip_indices = np.random.choice(len(labels), num_flips, replace=False)
                labels[flip_indices] = 1 - labels[flip_indices]
        return labels

    def train(self):
        if self.train_loader is None or len(self.train_loader.dataset) == 0:
            logging.warning(f"Client {self.client_id}: No training data. Skipping.")
            if self.initial_global_params_vector is not None:
                 num_params = len(self.initial_global_params_vector)
                 zero_delta_np = np.zeros(num_params, dtype=np.float64)
                 encrypted_zero_delta_b64 = encrypt_vector_tenseal(self.tenseal_context, zero_delta_np)
                 return encrypted_zero_delta_b64, 1.0, 0
            else:
                 logging.error(f"Client {self.client_id}: initial_global_params_vector is None. Cannot create zero delta.")
                 return None, 1.0, 0

        self.model.train()
        total_loss = 0.0
        num_batches_for_loss_avg = 0

        for epoch in range(LOCAL_EPOCHS):
            epoch_loss = 0.0
            epoch_batches = 0
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                if self.is_noisy:
                    target = self._apply_label_noise(target.clone())

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                epoch_batches += 1
            
            avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
            logging.debug(f"Client {self.client_id}, Epoch {epoch+1}/{LOCAL_EPOCHS}, Avg Epoch Loss: {avg_epoch_loss:.4f}")
            if epoch_batches > 0:
                total_loss += avg_epoch_loss 
                num_batches_for_loss_avg +=1

        average_loss_overall = total_loss / num_batches_for_loss_avg if num_batches_for_loss_avg > 0 else float('inf')
        
        # Calculate quality score based on actual loss
        quality_score = 1.0 / (average_loss_overall + QUALITY_SCORE_EPSILON)
        if np.isnan(quality_score) or np.isinf(quality_score):
            logging.warning(f"Client {self.client_id} produced invalid quality score ({quality_score}) from loss {average_loss_overall}. Setting to small value.")
            quality_score = QUALITY_SCORE_EPSILON

        local_params_vector = get_model_params_vector(self.model)
        if self.initial_global_params_vector is None:
            logging.error(f"Client {self.client_id}: initial_global_params_vector is None before delta calculation!")
            return None, quality_score, 0
        model_delta_vector = local_params_vector - self.initial_global_params_vector

        # Adversarial modifications
        if self.is_adversary:
            if ATTACK_TYPE == "model_poisoning_opposite": # ATTACK_TYPE from config
                logging.info(f"Client {self.client_id} (Adversary): Applying model poisoning (scale: {ATTACK_SCALE}).") # ATTACK_SCALE from config
                model_delta_vector = model_delta_vector * ATTACK_SCALE
            # Adversaries report their "natural" quality_score based on their (potentially good) local training loss.
            logging.info(f"Client {self.client_id} (Adversary): Reporting score based on its local training loss: {quality_score:.4f}")
        
        logging.debug(f"Client {self.client_id}: Encrypting model delta of size {len(model_delta_vector)} with TenSEAL...")
        model_delta_vector_float64 = model_delta_vector.astype(np.float64)
        encrypted_model_delta_b64 = encrypt_vector_tenseal(self.tenseal_context, model_delta_vector_float64)
        logging.debug(f"Client {self.client_id}: TenSEAL Encryption complete. Serialized base64 size: {len(encrypted_model_delta_b64)} chars.")

        num_samples_trained = len(self.train_loader.dataset)
        logging.info(f"Client {self.client_id}: Local training complete. Avg Overall Loss: {average_loss_overall:.4f}, Reported Quality Score: {quality_score:.4f}, Samples: {num_samples_trained}")
        
        return encrypted_model_delta_b64, quality_score, num_samples_trained