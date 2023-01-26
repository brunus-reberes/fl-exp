from dataset import load_dataset
import time
from model import run, test, results, pset, creator, gp_restrict, strongGPDataType, strings_to_individuals
from deap import gp
from pathlib import Path
import os
os.chdir(Path(__file__).parent)
from deap import creator
import jsonpickle

s = "Root3(SobelF(MeanF(ReLUF(MaxPF(GaborF(SobelXF(MaxP(Image0, 2, 2)), 4, 2), 2, 4)))), MaxPF(Roots2(Roots3(FGlobal_HOG(Image0), Global_HOG(LoG1F(MaxP(Image0, 2, 2))), FGlobal_uLBP(Image0)), FGlobal_HOG(Image0)), 4, 4), MedF(MaxP(GauD(Image0, 1, 1, 0), 2, 2)))"
ind = strings_to_individuals([s])
print(ind[0])
exit()

if __name__ == "__main__":
    train_size, test_size = 100, 100
    #get data: MNIST normal
    dataset = load_dataset("mnist", train_size=train_size, test_size=test_size)
    
    #start evolution
    beginTime = time.process_time()
    pop, log, hof = run(
        population=10,
        generation=1,
        crossover_rate=0.8,
        mutation_rate=0.2,
        train_set=(dataset[0], dataset[1]),
        seed=10,
        verbose=True
    )
    endTime = time.process_time()

    #test best individual
    train_time = endTime - beginTime
    error = test(hof[0], dataset[0], dataset[1], dataset[2], dataset[3])
    test_time = time.process_time() - endTime

    #print results
    print(f"train time: {train_time}")
    print(f"test time: {test_time}")
    print(f"best individual: {hof[0]}")
    print(f"error: {error}")
    print(f"accuracy: {100-error}")
    print("test preview")
    results(hof[0], dataset[0], dataset[1], dataset[2], dataset[3])
    print('End')