# Autonomous Cloud Security Orchestrator using FL and HE

This project demonstrates a conceptual Autonomous Cloud Security Orchestrator. It uses Federated Learning (FL) to train a threat detection model across multiple simulated cloud environments without sharing raw data. Homomorphic Encryption (HE) (specifically, the Paillier cryptosystem) is employed to protect the confidentiality of model updates during aggregation. The system simulates autonomous actions based on the learned global model's insights.

## Architecture

*   **Orchestrator Server (`server/`):**
    *   Manages the overall FL process (rounds).
    *   Initializes and distributes the global threat detection model (Logistic Regression).
    *   Generates and manages Paillier HE keys (public/private).
    *   Receives encrypted model updates (weight differences) from clients.
    *   Uses Paillier's additive homomorphism to aggregate encrypted updates.
    *   Decrypts the aggregated update using the private key.
    *   Updates the global model using the averaged decrypted updates.
    *   Simulates autonomous actions (e.g., logging a warning) if threat metrics exceed a threshold.
    *   Uses Flask for basic API communication.
    *   Uses `concurrent.futures` to simulate efficient handling of cryptographic operations.
*   **Cloud Client Agent (`client/`):**
    *   Simulates an agent running in a specific cloud environment.
    *   Registers with the server to get the HE public key and model details.
    *   Generates synthetic local security data (simulating logs/events) with potential biases.
    *   Receives the global model from the server.
    *   Trains the model locally on its data (`local_trainer.py`).
    *   Calculates the difference between its updated weights and the received global weights.
    *   Encrypts this weight difference vector using the server's public HE key (`crypto_manager.py`).
    *   Sends the encrypted update back to the server.

## Core Technologies

*   **Federated Learning:** Enables collaborative model training without centralizing sensitive data. The server orchestrates rounds, and clients train locally.
*   **Homomorphic Encryption (Paillier):** Allows the server to perform computations (addition for aggregation) on encrypted data without decrypting individual contributions. This protects the confidentiality of each client's model update.
*   **System Programming Concepts (Simulated):** The architecture emphasizes component separation (server, client, crypto, model). Concurrency (`ThreadPoolExecutor`) is used on the server to handle potentially intensive cryptographic operations efficiently, mimicking optimizations needed in real systems. Efficient libraries like NumPy are used.
*   **Autonomous Orchestration:** The server includes a placeholder `evaluate_model_and_trigger_action` function that simulates taking automated security actions based on the global model's state (e.g., detecting a high probability of threats).

## Setup and Running

**1. Prerequisites:**
   *   Python 3.8+
   *   Install GMP library (see instructions in the code comments or original prompt).
        *   Linux (Debian/Ubuntu): `sudo apt-get update && sudo apt-get install libgmp-dev`
        *   Linux (Fedora/CentOS): `sudo yum install gmp-devel`
        *   macOS (Homebrew): `brew install gmp`
        *   Windows: Use WSL or find pre-compiled binaries/conda.

**2. Install Python Dependencies:**
   Navigate to the `server` directory and run:
   ```bash
   pip install -r requirements.txt