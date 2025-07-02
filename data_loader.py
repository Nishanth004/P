# FEDMED/data_loader.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import torch
from torch.utils.data import TensorDataset, DataLoader
import logging

from config_tenseal import (DATA_PATH, TEST_SPLIT_RATIO, LABEL_COLUMN, CATEGORICAL_COLS,
                            SEED, NUM_CLIENTS, BATCH_SIZE)

def load_and_preprocess_data():
    """
    Loads the smoking dataset, preprocesses it, and splits it for federated learning.
    """
    logging.info("Loading and preprocessing data...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        logging.error(f"Dataset not found at {DATA_PATH}. Please place smoking.csv in the data/ directory.")
        raise

    # Drop ID
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
    
    # Convert Y/N to 1/0 for specific columns before OHE, if applicable
    # For 'oral', 'tartar', get_dummies will handle 'Y'/'N' fine.
    # 'dental caries' is already 0/1.
    # 'gender' is 'F'/'M'.

    X = df.drop(columns=[LABEL_COLUMN])
    y = df[LABEL_COLUMN]

    # Identify numerical features (excluding binary/already encoded ones and target)
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    # Remove known binary/categorical-like int columns from numerical_cols if they exist and are not in CATEGORICAL_COLS
    # For this dataset, 'dental caries' is binary, 'hearing(left/right)', 'Urine protein' might be categorical codes
    # Let's assume 'dental caries', 'hearing(left/right)', 'Urine protein' are fine as numerical features for now
    # or will be handled if explicitly added to CATEGORICAL_COLS.
    # The current CATEGORICAL_COLS only includes 'gender', 'oral', 'tartar'.

    # Preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, [col for col in numerical_cols if col not in CATEGORICAL_COLS]),
            ('cat', categorical_transformer, CATEGORICAL_COLS)
        ],
        remainder='passthrough' # Keep other columns (e.g., already binary 'dental caries')
    )
    
    # Split data into training and global test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT_RATIO, random_state=SEED, stratify=y
    )

    # Fit preprocessor on training data and transform both train and test
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after transformation for model input_dim
    try:
        feature_names = preprocessor.get_feature_names_out()
        input_dim = len(feature_names)
    except AttributeError: # older scikit-learn
        # Manual way if get_feature_names_out is not available (less robust)
        input_dim = X_train_processed.shape[1]
        logging.warning("preprocessor.get_feature_names_out() not available. Input dimension inferred directly from shape.")
        

    logging.info(f"Data preprocessed. Input dimension: {input_dim}")
    logging.info(f"Training samples: {X_train_processed.shape[0]}, Test samples: {X_test_processed.shape[0]}")

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    # Create global test DataLoader
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Calculate pos_weight for class imbalance from the entire training set
    pos_weight_value = None
    if y_train_tensor.numel() > 0: # Check if y_train_tensor is not empty
        num_positives = torch.sum(y_train_tensor == 1).item()
        num_negatives = torch.sum(y_train_tensor == 0).item()
        if num_positives > 0 and num_negatives > 0 :
            pos_weight_value = num_negatives / num_positives
            logging.info(f"Calculated pos_weight for BCEWithLogitsLoss: {pos_weight_value:.4f}")
        elif num_positives == 0 and num_negatives > 0 :
            logging.warning("No positive samples in training data. pos_weight cannot be calculated.")
            pos_weight_value = 1.0 # Default to 1 if no positive samples.
        elif num_negatives == 0 and num_positives > 0 :
            logging.warning("No negative samples in training data. pos_weight cannot be calculated effectively.")
            # This case is unusual for pos_weight, but setting to a default or a high value might be options.
            # Let's use a very small value to effectively down-weight the non-existent negative class.
            # Or simply default to 1.0. For now, let's keep it as 1.0.
            pos_weight_value = 1.0 # Default to 1 if no negative samples.
        else: # Both are zero (empty dataset)
            logging.warning("Training data is empty. pos_weight cannot be calculated.")
            pos_weight_value = 1.0

    # Distribute training data among clients (IID)
    client_dataloaders = []
    if X_train_tensor.shape[0] > 0 :
        total_train_samples = X_train_tensor.shape[0]
        samples_per_client = total_train_samples // NUM_CLIENTS
        
        # Shuffle data before distributing
        perm = torch.randperm(total_train_samples)
        X_train_shuffled = X_train_tensor[perm]
        y_train_shuffled = y_train_tensor[perm]

        for i in range(NUM_CLIENTS):
            start_idx = i * samples_per_client
            # Ensure the last client gets all remaining samples
            if i == NUM_CLIENTS - 1:
                end_idx = total_train_samples
            else:
                end_idx = start_idx + samples_per_client
            
            X_client = X_train_shuffled[start_idx:end_idx]
            y_client = y_train_shuffled[start_idx:end_idx]

            if len(X_client) > 0:
                client_dataset = TensorDataset(X_client, y_client)
                # Drop last batch if it's smaller than BATCH_SIZE and causes issues, or handle it
                # For training, drop_last=True can stabilize if batch norm is sensitive. Here, it's probably fine.
                client_loader = DataLoader(client_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
                client_dataloaders.append(client_loader)
            else:
                logging.warning(f"Client {i} received no data. Check NUM_CLIENTS and dataset size.")
                # Add an empty loader or handle as per desired logic
                client_dataloaders.append(None) 
    else: # No training data
        logging.warning("No training data available to distribute to clients.")
        for i in range(NUM_CLIENTS):
            client_dataloaders.append(None)


    return client_dataloaders, test_loader, input_dim, pos_weight_value, preprocessor