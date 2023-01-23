from pathlib import Path
import os
os.chdir(Path(__file__).parent)
import flwr as fl
import server
import client
import dataset 
from settings import CLIENTS, ROUNDS, BATCH, TRAIN_SIZE, TEST_SIZE, DATASET


if __name__ == "__main__":
    # Start simulation

    def client_fn(cid: str) -> client.GeneticClient:
        if BATCH:
            ds = dataset.load_dataset_batches(DATASET, int(cid), train_size=TRAIN_SIZE, test_size=TEST_SIZE)
        else:
            ds = dataset.load_dataset(DATASET, train_size=TRAIN_SIZE, test_size=TEST_SIZE)
        return client.GeneticClient(cid, ds[:2], ds[2:])

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=CLIENTS,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=server.GeneticStrategy(),
    )


