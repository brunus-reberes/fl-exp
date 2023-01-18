from pathlib import Path
import os
os.chdir(Path(__file__).parent)
import flwr as fl
import federated as fed
from settings import SIMULATE, CLIENTS, ROUNDS
import logging
logging.basicConfig()


if __name__ == "__main__":
    if SIMULATE:
        # Start simulation
        fl.simulation.start_simulation(
            client_fn=fed.client_fn,
            num_clients=CLIENTS,
            config=fl.server.ServerConfig(num_rounds=ROUNDS),
            strategy=fed.GeneticStrategy(),
        )


