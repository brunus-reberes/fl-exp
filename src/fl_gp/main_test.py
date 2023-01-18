from dataset import load_dataset
import time
from model import run, test, results, pset, creator, gp_restrict, strongGPDataType
from deap import gp
from pathlib import Path
import os
os.chdir(Path(__file__).parent)
from deap import creator
import jsonpickle

if __name__ == "__main__":
    train_size, test_size = 1000, 100
    #get data: MNIST normal
    dataset = load_dataset("mnist", train_size=train_size, test_size=test_size)
    if False:
        expr = gp_restrict.genHalfAndHalfMD(pset=pset, min_=2, max_=6)
        ind = creator.Individual(expr)
        print(ind)
        print(code := jsonpickle.encode(ind, include_properties=True))
        ind = jsonpickle.decode(code, classes=[gp.Terminal, gp.PrimitiveTree, gp.PrimitiveSetTyped, gp.Primitive, gp.Ephemeral])
        ind = creator.Individual(ind)
        print(ind)
        test(ind, dataset[0], dataset[1], dataset[2], dataset[3])
        exit()
    else:
        class int(strongGPDataType.Int3):
            pass
        ind = "Roots3(Global_SIFT(SobelXF(MinF(MaxP(ReLU(Image0), 2, 3)))), FGlobal_uLBP(Image0), Global_HOG(ReLUF(ReLUF(SqrtF(HoGF(SobelXF(SobelYF(SqrtF(MaxF(GauF(MaxP(Image0, 4, 4), 4)))))))))))"
        ind = creator.Individual.from_string(ind, pset)
        print(ind)
    #start evolution
    beginTime = time.process_time()
    pop, log, hof = run(
        population=10,
        generation=2,
        crossover_rate=0.8,
        mutation_rate=0.2,
        train_set=(dataset[0], dataset[1]),
        seed=10,
        verbose=True
    )
    endTime = time.process_time()

    #test best individual
    train_time = endTime - beginTime
    accuracy = test(hof[0], dataset[0], dataset[1], dataset[2], dataset[3])
    test_time = time.process_time() - endTime

    #print results
    print(f"train time: {train_time}")
    print(f"test time: {test_time}")
    print(f"best individual: {hof[0]}")
    print(f"accuracy: {accuracy}")
    print(f"best individuals: {hof}")
    print("test preview")
    results(hof[0], dataset[0], dataset[1], dataset[2], dataset[3])
    print('End')