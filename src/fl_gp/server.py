import flwr as fl
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from deap import tools
import numpy

import logging
import pickle

from settings import HOF_SIZE, LOGS, PICKLES

import model

class GeneticStrategy(fl.server.strategy.Strategy):
    def __init__(self) -> None:
        super().__init__()
        #vars
        self.round = 0
        self.hof = []

        #logger
        self.logger = logging.getLogger(f'GeneticStrategy')
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(f'{LOGS}/GeneticStrategy.log', mode="w")
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(fmt=' %(name)s :: %(levelname)-8s :: %(message)s')
        fh.setFormatter(formatter)
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
        return ndarrays_to_parameters(self.hof)

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
        population = []
        for _, fitres in results:
            param = parameters_to_ndarrays(fitres.parameters)[0]
            print(param)
            try:
                param = model.strings_to_individuals(param)
                #fill individuals with fitness values
                for p in param:
                    p.fitness.values = (fitres.metrics[str(p)],) 
                population.extend(param)
            except:
                for string in param:
                    try:
                        ind = model.string_to_individual(string)
                        ind.fitness.values = (fitres.metrics[string],) 
                        population.append(ind)
                    except:
                        pass

        #aggregate
        self.logger.info(f'population size: {len(population)}')
        self.hof = model.aggregate(population, HOF_SIZE)
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
        if len(self.hof) == 0:
            self.logger.info('empty...')
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
        if len(self.hof) == 0:
            self.logger.info('empty...')
        else:
            for i, ind in enumerate(self.hof):
                self.logger.info(f'{i} (mean: {round(ind.fitness.values[0],3)}): {str(ind)}')

            #logging
            record = self.mstats.compile(self.hof)
            self.logbook.record(round=self.round, nclients=self.nclients, evals=len(self.hof), **record)
            self.logger.info(f'\n{str(self.logbook)}')

            pickle.dump(self.logbook, open(f'{PICKLES}/GeneticStrategyROUND{self.round}.pickle', "ab"))

        return 0., ind_mean_error

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None

