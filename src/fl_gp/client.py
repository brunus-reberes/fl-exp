import flwr as fl
from typing import Dict, Tuple
from flwr.common import Config, NDArrays, Scalar
import logging
import pickle

from settings import PICKLES, LOGS, POPULATION, CROSSOVER, MUTATION, GENERATION, INIT_MIN_DEPTH, INIT_MAX_DEPTH, MAX_DEPTH, ELITISM, HOF_SIZE, RUNS, TOURNMENT_SIZE, SEED, VERBOSE

import model


class GeneticClient(fl.client.NumPyClient):
    def __init__(self, cid, train_set, test_set) -> None:
        self.hof = []
        self.cid = cid
        self.round = 0
        self.train_set = train_set
        self.test_set = test_set

        #logger
        self.logger = logging.getLogger(f'GeneticClientID{cid}')
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(f'{LOGS}/GeneticClientID{cid}.log')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(fmt=' %(name)s :: %(levelname)-8s :: %(message)s')
        fh.setFormatter(formatter)
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
        pickle.dump(log, open(f'{PICKLES}/GeneticClientID{self.cid}ROUND{self.round}.pickle', "ab"))

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

