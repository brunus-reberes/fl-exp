import flwr as fl
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Config, NDArrays, Scalar, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from deap import tools
import numpy

from dataset import load_dataset 
import logging
import pickle

from settings import TRAIN_SIZE, TEST_SIZE, TEST, POPULATION, CROSSOVER, MUTATION, GENERATION, INIT_MIN_DEPTH, INIT_MAX_DEPTH, MAX_DEPTH, ELITISM, HOF_SIZE, RUNS, DATASET, TOURNMENT_SIZE, SEED, VERBOSE

if TEST:
    import model_test as model
else:
    import model

class GeneticClient(fl.client.NumPyClient):
    def __init__(self, cid, train_set=None, test_set=None) -> None:
        self.hof = []
        self.cid = cid
        self.round = 0
        self.train_set = train_set
        self.test_set = test_set
        self.logger = logging.getLogger(f'GeneticClientID{cid}')
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(f'logs/GeneticClientID{cid}.log')
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        # TODO: receber hyperparametros do servidor
        return {}

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return []

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.round = self.round + 1
        self.logger.info(f'ROUND {self.round}')
        self.logger.info(f'initial population (from server): {parameters}')
        init_population = model.strings_to_individuals(parameters)
        _, log, self.hof = model.run(
            init_population=init_population, 
            population=POPULATION, 
            generation=GENERATION, 
            crossover_rate=CROSSOVER, 
            mutation_rate=MUTATION, 
            elitism_rate=ELITISM, 
            init_min_depth=INIT_MIN_DEPTH, 
            init_max_depth=INIT_MAX_DEPTH, 
            max_depth=MAX_DEPTH, 
            hof_size=HOF_SIZE, 
            runs=RUNS, 
            train_set=self.train_set,
            test_set=self.test_set,
            seed=SEED+int(self.cid),
            tournment_size=TOURNMENT_SIZE,
            )
        if VERBOSE:
            self.logger.info(str(log))
        pickle.dump(log, open(f'pickles/GeneticClientID{self.cid}ROUND{self.round}.pickle', "wb"))
        hof = model.individuals_to_strings(self.hof)
        return [hof], len(self.train_set[0]), {}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        parameters = model.strings_to_individuals(parameters[0])
        accuracy = model.test(parameters[0], self.train_set[0], self.train_set[1], self.test_set[0], self.test_set[1])
        self.logger.info(f'accuracy = {accuracy}')
        self.logger.info(f'classification_error_rate = {100-accuracy}')
        return 0., len(self.test_set[0]), {"classification_error_rate": 100-accuracy}

class GeneticStrategy(fl.server.strategy.Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(f'GeneticStrategy')
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(f'logs/GeneticStrategy.log')
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)
        self.round = 0
        self.logbook = tools.Logbook()
        stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats_size_tree = tools.Statistics(key=len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size_tree=stats_size_tree)
        mstats.register("avg", numpy.mean)
        mstats.register("std", numpy.std)
        mstats.register("min", numpy.min)
        mstats.register("max", numpy.max)
        self.mstats = mstats
        self.logbook.header = ["round", "nclients", "evals"] + self.mstats.fields

    def initialize_parameters(
            self, client_manager: ClientManager
        ) -> Optional[Parameters]:
        self.round = self.round + 1
        self.logger.info(f"ROUND {self.round}")
        return None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        clients = client_manager.all().values()
        self.nclients = len(clients)
        return [(client, FitIns(parameters, {})) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        parameters = []
        for _, fitres in results:
            hof = parameters_to_ndarrays(fitres.parameters)[0]
            parameters.extend(hof)
        parameters = model.strings_to_individuals(parameters)
        result = model.aggregate(parameters, HOF_SIZE)
        print(result)
        print(type(result))
        #logging
        record = self.mstats.compile(list(result))
        self.logbook.record(round=self.round, nclients=self.nclients, evals=len(result), **record)
        self.logger.info('Global Best Individuals')
        for ind in parameters:
            self.logger.info(ind) 
        self.logger.info(str(self.logbook))
        result = model.individuals_to_strings(result)
        return ndarrays_to_parameters([result]), {}

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
        error = []
        for _, fitres in results:
            error.append(fitres.metrics["classification_error_rate"])
        if len(error) != 0:
            error = np.mean(error)
        else:
            error = 100
        self.logger.info(f'Error mean: {error}')
        return 0., {"classification_error_rate_mean": error}

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        pickle.dump(self.logbook, open(f'pickles/GeneticStrategyROUND{self.round}.pickle', "wb"))
        return None

def client_fn(cid: str) -> GeneticClient:
    dataset = load_dataset(DATASET, train_size=TRAIN_SIZE, test_size=TEST_SIZE)
    return GeneticClient(cid, dataset[:2], dataset[2:])
