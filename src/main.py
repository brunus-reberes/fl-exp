from collections import OrderedDict
from typing import List, Tuple

import flwr as fl
from ellyn import ellyn
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Union, Optional, Dict

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

CLASSES = ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica')
NUM_CLIENTS = 2

BATCH_SIZE = 32

def load_datasets():
    df = pd.read_csv('src/datasets/iris_csv.csv')
    train, test = train_test_split(df)
    train = train.iloc[:, :-1].values, train.iloc[:, [-1]].values
    test = test.iloc[:, :-1].values, test.iloc[:, [-1]].values
    return train, test


def train(model, train_data, popsize, gen, verbose=False):
    """Train the M4GP on the training set."""
    print(type(model))
    model.verbosity = verbose
    model.popsize = popsize
    model.g = gen
    model.fit(train_data[0], train_data[1])


def test(model, test_data):
    """Evaluate the network on the entire test set."""
    print(type(model))
    model.predict(test_data[0])
    score = model.score(test_data[0], test_data[1])
    return score

#train(model, train_data, popsize=10000, gen=1000,verbose=False)
#print(test(model, test_data))




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


class FedCustom(fl.server.strategy.Strategy):
    def __repr__(self) -> str:
        return "FedCustom"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        return []

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        # TODO WIP - add implementation

        return []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        # TODO WIP - add implementation

        return None, {}

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""

        # TODO WIP - add implementation

        return []

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""

        # TODO WIP - add implementation

        return None, {}

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""

        # TODO WIP - add implementation

        return None


# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=2),
    strategy=fl.server.strategy.FedAvg(),
)
