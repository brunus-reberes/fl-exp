import flwr as fl


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, test_data):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data

    def get_parameters(self, config):
        return []

    def set_parameters(self, parameters):
        pass

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.train_data, popsize=10000, gen=1000, verbose=False)
        return self.get_parameters(self.model), len(self.train_data), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        accuracy = test(self.model, self.test_data)
        return 0, 0, {"accuracy": float(accuracy)}

def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    # Load model
    model = ellyn(classification=True, 
                class_m4gp=True, 
                prto_arch_on=True,
                selection='lexicase',
                fit_type='F1', # can be 'F1' or 'F1W' (weighted F1)dition=False,
                #stop_condition=True
               )

    train_data, test_data = load_datasets()

    # Create a  single Flower client representing a single organization
    return FlowerClient(model, train_data, test_data)

class GeneticStrategy(fl.server.strategy.Strategy):
    def initialize_parameters(self, client_manager):
        # Your implementation here

    def configure_fit(self, server_round, parameters, client_manager):
        # Your implementation here

    def aggregate_fit(self, server_round, results, failures):
        # Your implementation here

    def configure_evaluate(self, server_round, parameters, client_manager):
        # Your implementation here

    def aggregate_evaluate(self, server_round, results, failures):
        # Your implementation here

    def evaluate(self, parameters):
        # Your implementation here

# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=2),
    strategy=fl.server.strategy.FedAvg(),
)