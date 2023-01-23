import flwr as fl
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Config, NDArrays, Scalar, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from deap import tools, gp
import numpy

from dataset import load_dataset, batch
import logging
import pickle

from settings import TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, TRAIN_SIZE, TEST_SIZE, TEST, POPULATION, CROSSOVER, MUTATION, GENERATION, INIT_MIN_DEPTH, INIT_MAX_DEPTH, MAX_DEPTH, ELITISM, HOF_SIZE, RUNS, DATASET, TOURNMENT_SIZE, SEED, VERBOSE

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
        #logs
        self.round = self.round + 1
        self.logger.info(f'ROUND {self.round}')
        self.logger.info(f'initial population (from server): {parameters}')
        
        #evolution
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
        self.logger.info(f'Seed: {SEED+int(self.cid)}')
        if VERBOSE:
            self.logger.info(str(log))
        
        #prints best individuals with updated hall of fame
        self.logger.info('Global Best Individuals')
        for i, ind in enumerate(self.hof):
            self.logger.info(f'{i} (fitness: {round(ind.fitness.values[0], 3)}): {str(ind)}')
        
        #pickle
        pickle.dump(log, open(f'pickles/GeneticClientID{self.cid}ROUND{self.round}.pickle', "ab"))

        #get errors to send to server
        errors = {}
        for ind in self.hof:
            errors[str(ind)] = ind.fitness.values[0]

        hof = model.individuals_to_strings(self.hof)
        return [hof], len(self.train_set[0]), errors

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        parameters = model.strings_to_individuals(parameters[0])
        errors = {}
        for param in parameters:
            error = model.test(param, self.train_set[0], self.train_set[1], self.test_set[0], self.test_set[1])
            errors[str(param)] = error
        self.logger.info(f'classification_error_rate = {errors}')
        return 0., len(self.test_set[0]), errors

class GeneticStrategy(fl.server.strategy.Strategy):
    def __init__(self) -> None:
        super().__init__()
        #vars
        self.round = 0
        self.hof = []
        #logger
        self.logger = logging.getLogger(f'GeneticStrategy')
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(f'logs/GeneticStrategy.log')
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)
        #logbook
        self.logbook = tools.Logbook()
        stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats_size_tree = tools.Statistics(key=len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size_tree)
        mstats.register("avg", numpy.mean)
        mstats.register("std", numpy.std)
        mstats.register("min", numpy.min)
        mstats.register("max", numpy.max)
        self.mstats = mstats
        self.logbook.header = ["round", "nclients", "evals"] + self.mstats.fields
        self.logbook.chapters["fitness"].header = "min", "avg", "max", "std"
        self.logbook.chapters["size"].header = "min", "avg", "max", "std"

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
        #extract individuals from parameters
        parameters = []
        for _, fitres in results:
            param = parameters_to_ndarrays(fitres.parameters)[0]
            param = model.strings_to_individuals(param)
            #fill individuals with fitness values
            for p in param:
                p.fitness.values = (fitres.metrics[str(p)],)  
            parameters.extend(param)

        #aggregate
        self.hof = model.aggregate(parameters, HOF_SIZE)
        strings = model.individuals_to_strings(self.hof)
        return ndarrays_to_parameters([strings]), {}

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        clients = client_manager.all().values()
        self.logger.info(f'Number of clients to evaluate: {len(clients)}')
        return [(client, EvaluateIns(parameters, {})) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        client_errors = {}
        ind_mean_error = {}

        #get the list of errors per client
        for client, fitres in results:
            client_errors[client.cid] = fitres.metrics.values()
            self.logger.info(f'Errors of Client{client.cid}: {str(client_errors[client.cid])}')

        #calculates the mean error of each hall of fame individual
        self.logger.info('Global Best Individuals (before mean)')
        for i, ind in enumerate(self.hof):
            errors = []
            id= str(ind)
            for client, fitres in results:
                if id in fitres.metrics:
                    errors.append(fitres.metrics[id])
            ind_mean_error[id] = np.mean(errors)
            self.logger.info(f'{i} (mean: {round(ind_mean_error[id],3)}): {id}')

        #updates the hall of fame based on the mean error
        hof = tools.HallOfFame(HOF_SIZE)
        for ind in self.hof:
            ind.fitness.values = (ind_mean_error[str(ind)],)
            hof.update([ind])
        self.hof = hof

        #prints best endividuals with updated hall of fame
        self.logger.info('Global Best Individuals (after mean)')
        for i, ind in enumerate(self.hof):
            self.logger.info(f'{i} (mean: {round(ind.fitness.values[0],3)}): {str(ind)}')

        #logging
        record = self.mstats.compile(self.hof)
        self.logbook.record(round=self.round, nclients=self.nclients, evals=len(self.hof), **record)
        self.logger.info(str(self.logbook))

        pickle.dump(self.logbook, open(f'pickles/GeneticStrategyROUND{self.round}.pickle', "ab"))

        return 0., ind_mean_error

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None

dataset = load_dataset(DATASET, train_size=TRAIN_SIZE, test_size=TEST_SIZE)
train_data_batches = list(batch(dataset[0], TRAIN_BATCH_SIZE))
train_labels_batches = list(batch(dataset[1], TRAIN_BATCH_SIZE))
test_data_batches = list(batch(dataset[2], TEST_BATCH_SIZE))
test_labels_batches = list(batch(dataset[3], TEST_BATCH_SIZE))

def client_fn(cid: str) -> GeneticClient:
    train_data = train_data_batches[int(cid)]
    train_labels = train_labels_batches[int(cid)]
    test_data = test_data_batches[int(cid)]
    test_labels = test_labels_batches[int(cid)]
    return GeneticClient(cid, (train_data, train_labels), (test_data, test_labels))
