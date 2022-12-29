import flwr as fl
import federated as fed

CLIENTS = 1
ROUNDS = 1
SIMULATE = True

if __name__ == "__main__":
    if SIMULATE:
        # Start simulation
        fl.simulation.start_simulation(
            client_fn=fed.client_fn,
            num_clients=CLIENTS,
            config=fl.server.ServerConfig(num_rounds=ROUNDS),
            strategy=fed.GeneticStrategy(),
        )

