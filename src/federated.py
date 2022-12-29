import flwr as fl
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Config, NDArrays, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

import model

class GeneticClient(fl.client.NumPyClient):
    def __init__(self, cid) -> None:
        self.hof = []
        self.cid = cid

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        """Returns a client's set of properties.

        Parameters
        ----------
        config : Config
            Configuration parameters requested by the server.
            This can be used to tell the client which properties
            are needed along with some Scalar attributes.

        Returns
        -------
        properties : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of type
            bool, bytes, float, int, or str. It can be used to communicate
            arbitrary property values back to the server.
        """
        return {}

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the current local model parameters.

        Parameters
        ----------
        config : Config
            Configuration parameters requested by the server.
            This can be used to tell the client which parameters
            are needed along with some Scalar attributes.

        Returns
        -------
        parameters : NDArrays
            The local model parameters as a list of NumPy ndarrays.
        """
        return self.hof

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train the provided parameters using the locally held dataset.

        Parameters
        ----------
        parameters : NDArrays
            The current (global) model parameters.
        config : Dict[str, Scalar]
            Configuration parameters which allow the
            server to influence training on the client. It can be used to
            communicate arbitrary values from the server to the client, for
            example, to set the number of (local) training epochs.

        Returns
        -------
        parameters : NDArrays
            The locally updated model parameters.
        num_examples : int
            The number of examples used for training.
        metrics : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of type
            bool, bytes, float, int, or str. It can be used to communicate
            arbitrary values back to the server.
        """
        self.hof = model.train(parameters.toList(), True)
        return self.hof, 1, {}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the provided parameters using the locally held dataset.

        Parameters
        ----------
        parameters : NDArrays
            The current (global) model parameters.
        config : Dict[str, Scalar]
            Configuration parameters which allow the server to influence
            evaluation on the client. It can be used to communicate
            arbitrary values from the server to the client, for example,
            to influence the number of examples used for evaluation.

        Returns
        -------
        loss : float
            The evaluation loss of the model on the local dataset.
        num_examples : int
            The number of examples used for evaluation.
        metrics : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of
            type bool, bytes, float, int, or str. It can be used to
            communicate arbitrary values back to the server.

        Warning
        -------
        The previous return type format (int, float, float) and the
        extended format (int, float, float, Dict[str, Scalar]) have been
        deprecated and removed since Flower 0.19.
        """
        return float(0), 1, {}

class GeneticStrategy(fl.server.strategy.Strategy):
    def initialize_parameters(
            self, client_manager: ClientManager
        ) -> Optional[Parameters]:
        """Initialize the (global) model parameters.
        Parameters
        ----------
        client_manager : ClientManager
            The client manager which holds all currently connected clients.
        Returns
        -------
        parameters : Optional[Parameters]
            If parameters are returned, then the server will treat these as the
            initial global model parameters.
        """
        return fl.common.ndarrays_to_parameters(model.toolbox.population(n=300))

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training.
        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        parameters : Parameters
            The current (global) model parameters.
        client_manager : ClientManager
            The client manager which holds all currently connected clients.
        Returns
        -------
        fit_configuration : List[Tuple[ClientProxy, FitIns]]
            A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
            `FitIns` for this particular `ClientProxy`. If a particular `ClientProxy`
            is not included in this list, it means that this `ClientProxy`
            will not participate in the next round of federated learning.
        """
        clients = client_manager.all().values()
        return [(client, FitIns(parameters, {})) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results.
        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        results : List[Tuple[ClientProxy, FitRes]]
            Successful updates from the previously selected and configured
            clients. Each pair of `(ClientProxy, FitRes)` constitutes a
            successful update from one of the previously selected clients. Not
            that not all previously selected clients are necessarily included in
            this list: a client might drop out and not submit a result. For each
            client that did not submit an update, there should be an `Exception`
            in `failures`.
        failures : List[Union[Tuple[ClientProxy, FitRes], BaseException]]
            Exceptions that occurred while the server was waiting for client
            updates.
        Returns
        -------
        parameters : Optional[Parameters]
            If parameters are returned, then the server will treat these as the
            new global model parameters (i.e., it will replace the previous
            parameters with the ones returned from this method). If `None` is
            returned (e.g., because there were only failures and no viable
            results) then the server will no update the previous model
            parameters, the updates received in this round are discarded, and
            the global model parameters remain the same.
        """
        new_hof = model.tools.HallOfFame(10)
        for _, hof in results:
            new_hof.update(hof)
        return fl.common.ndarrays_to_parameters(new_hof), {}

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation.
        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        parameters : Parameters
            The current (global) model parameters.
        client_manager : ClientManager
            The client manager which holds all currently connected clients.
        Returns
        -------
        evaluate_configuration : List[Tuple[ClientProxy, EvaluateIns]]
            A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
            `EvaluateIns` for this particular `ClientProxy`. If a particular
            `ClientProxy` is not included in this list, it means that this
            `ClientProxy` will not participate in the next round of federated
            evaluation.
        """
        clients = client_manager.all().values()
        return [(client, EvaluateIns(parameters, {})) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results.
        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        results : List[Tuple[ClientProxy, FitRes]]
            Successful updates from the
            previously selected and configured clients. Each pair of
            `(ClientProxy, FitRes` constitutes a successful update from one of the
            previously selected clients. Not that not all previously selected
            clients are necessarily included in this list: a client might drop out
            and not submit a result. For each client that did not submit an update,
            there should be an `Exception` in `failures`.
        failures : List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
            Exceptions that occurred while the server was waiting for client updates.
        Returns
        -------
        aggregation_result : Optional[float]
            The aggregated evaluation result. Aggregation typically uses some variant
            of a weighted average.
        """
        return None, {}

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate the current model parameters.
        This function can be used to perform centralized (i.e., server-side) evaluation
        of model parameters.
        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        parameters: Parameters
            The current (global) model parameters.
        Returns
        -------
        evaluation_result : Optional[Tuple[float, Dict[str, Scalar]]]
            The evaluation result, usually a Tuple containing loss and a
            dictionary containing task-specific metrics (e.g., accuracy).
        """
        return None

def client_fn(cid: str) -> GeneticClient:
    return GeneticClient(cid)
