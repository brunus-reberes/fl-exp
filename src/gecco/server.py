import flwr as fl
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import os
os.chdir(Path(__file__).parent)

from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from settings import *
import model
from deap.tools import HallOfFame
import numpy as np

class GeneticStrategy(fl.server.strategy.Strategy):
    def __init__(self) -> None:
        super().__init__()
        #vars
        self.hof = HallOfFame(HOF_SIZE)

    def initialize_parameters(
            self, client_manager: ClientManager
        ) -> Optional[Parameters]:
        client_manager.wait_for(CLIENTS)
        return None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        clients = client_manager.all().values()
        return [(client, FitIns(parameters, {})) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        pop = []
        for _, fitres in results:
            for string in fitres.metrics:
                ind = model.string_to_individual(string)
                ind.fitness.values = (fitres.metrics[string],)
                pop.append(ind)
        self.hof.update(pop)
        hof = model.individuals_to_strings(self.hof)
        return ndarrays_to_parameters(hof), {}

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        clients = client_manager.all().values()
        return [(client, EvaluateIns(parameters, {})) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        errors = {}
        for _, evares in results:
            for string in evares.metrics:
                if string not in errors:
                    errors[string] = []
                errors[string].append(evares.metrics[string])
        
        print(errors)

        self.hof.clear()
        pop = []
        for string in errors:
            errors[string] = np.mean(errors[string])
            ind = model.string_to_individual(string)
            ind.fitness.values = (errors[string],)
            pop.append(ind)
        self.hof.update(pop)
        print(errors)
        print(self.hof)

        return float(self.hof[0].fitness.values[0]), errors

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None

fl.server.start_server(config=fl.server.ServerConfig(num_rounds=ROUNDS), strategy=GeneticStrategy())
