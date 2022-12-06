import flwr as fl
import federated as fed

CLASSES = ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica')
NUM_CLIENTS = 2

BATCH_SIZE = 32

SIMULATE = True

if __name__ == "__main__":
    if SIMULATE:
        # Start simulation
        fl.simulation.start_simulation(
            client_fn=fed.client_fn,
            num_clients=NUM_CLIENTS,
            config=fl.server.ServerConfig(num_rounds=2),
            strategy=fed.GeneticStrategy(),
        )

