from pathlib import Path
import os
os.chdir(Path(__file__).parent)
import sys
import flwr as fl
from typing import Dict, Tuple
from flwr.common import Config, NDArrays, Scalar
import dataset
from settings import *
import model

train_data_batch, train_labels_batch, test_data_batch, test_labels_batch = dataset.load_dataset_batches(DATASET, train_size=TRAIN_SIZE, test_size=TEST_SIZE)


class GeneticClient(fl.client.NumPyClient):
    def __init__(self, cid) -> None:
        self.cid = cid

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        return {}

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return []

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:

        #get data
        train_data = train_data_batch[int(self.cid)]
        train_labels = train_labels_batch[int(self.cid)]

        #evolution
        _, log, hof = model.run(
            population=POPULATION, 
            generation=GENERATION, 
            hof_size=HOF_SIZE, 
            train_set=(train_data, train_labels),
            seed=SEED+int(self.cid),
            )

        print(log)

        pop = {}
        for ind in hof:
            pop[str(ind)] = ind.fitness.values[0]
            
        return [], len(train_data), pop

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        train_data = train_data_batch[int(self.cid)]
        train_labels = train_labels_batch[int(self.cid)]
        test_data = test_data_batch[int(self.cid)]
        test_labels = test_labels_batch[int(self.cid)]
        errors = {}
        for param in parameters:
            string = str(param)
            ind = model.string_to_individual(string)
            errors[string] = model.test(ind, train_data, train_labels, test_data, test_labels)
        return 0., len(test_data), errors

fl.client.start_numpy_client(server_address="localhost:8080", client=GeneticClient(sys.argv[1]))