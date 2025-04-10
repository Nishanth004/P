# Autonomous Cloud Orchestrator

This project simulates an autonomous federated learning environment with HE support.

## Structure
- `server/`: Handles orchestration and model aggregation.
- `client/`: Simulates clients that train and encrypt model updates.

## Quickstart
1. Run the server:
    ```bash
    cd orch/server
    pip install -r requirements.txt
    python main.py
    ```

2. Run the client:
    ```bash
    cd orch/client
    pip install -r requirements.txt
    python main.py
    ```
