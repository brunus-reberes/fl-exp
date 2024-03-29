from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.absolute()))
import random
import time
import evalGP_fgp as evalGP
import gp_restrict as gp_restrict
import numpy
from deap import base, creator, tools, gp
from strongGPDataType import Int1, Int2, Int3, Float1, Float2, Float3, Img, Img1, Vector, Vector1
import fgp_functions as fe_fs
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
#import saveFile
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import operator

creator.create("Fitness", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp_restrict.PrimitiveTree, fitness=creator.Fitness)
#Primitives
pset = gp.PrimitiveSetTyped('MAIN', [Img], Vector1, prefix='Image')
#feature concatenation
pset.addPrimitive(fe_fs.root_con, [Vector1, Vector1], Vector1, name='Root')
pset.addPrimitive(fe_fs.root_conVector2, [Img1, Img1], Vector1, name='Root2')
pset.addPrimitive(fe_fs.root_conVector3, [Img1, Img1, Img1], Vector1, name='Root3')
pset.addPrimitive(fe_fs.root_con, [Vector, Vector], Vector1, name='Roots2')
pset.addPrimitive(fe_fs.root_con, [Vector, Vector, Vector], Vector1, name='Roots3')
pset.addPrimitive(fe_fs.root_con, [Vector, Vector, Vector, Vector], Vector1, name='Roots4')
##feature extraction
pset.addPrimitive(fe_fs.global_hog_small, [Img1], Vector, name='Global_HOG')
pset.addPrimitive(fe_fs.all_lbp, [Img1], Vector, name='Global_uLBP')
pset.addPrimitive(fe_fs.all_sift, [Img1], Vector, name='Global_SIFT')
pset.addPrimitive(fe_fs.global_hog_small, [Img], Vector, name='FGlobal_HOG')
pset.addPrimitive(fe_fs.all_lbp, [Img], Vector, name='FGlobal_uLBP')
pset.addPrimitive(fe_fs.all_sift, [Img], Vector, name='FGlobal_SIFT')
# pooling
pset.addPrimitive(fe_fs.maxP, [Img1, Int3, Int3], Img1, name='MaxPF')
#filtering
pset.addPrimitive(fe_fs.gau, [Img1, Int1], Img1, name='GauF')
pset.addPrimitive(fe_fs.gauD, [Img1, Int1, Int2, Int2], Img1, name='GauDF')
pset.addPrimitive(fe_fs.gab, [Img1, Float1, Float2], Img1, name='GaborF')
pset.addPrimitive(fe_fs.laplace, [Img1], Img1, name='LapF')
pset.addPrimitive(fe_fs.gaussian_Laplace1, [Img1], Img1, name='LoG1F')
pset.addPrimitive(fe_fs.gaussian_Laplace2, [Img1], Img1, name='LoG2F')
pset.addPrimitive(fe_fs.sobelxy, [Img1], Img1, name='SobelF')
pset.addPrimitive(fe_fs.sobelx, [Img1], Img1, name='SobelXF')
pset.addPrimitive(fe_fs.sobely, [Img1], Img1, name='SobelYF')
pset.addPrimitive(fe_fs.medianf, [Img1], Img1, name='MedF')
pset.addPrimitive(fe_fs.meanf, [Img1], Img1, name='MeanF')
pset.addPrimitive(fe_fs.minf, [Img1], Img1, name='MinF')
pset.addPrimitive(fe_fs.maxf, [Img1], Img1, name='MaxF')
pset.addPrimitive(fe_fs.lbp, [Img1], Img1, name='LBPF')
pset.addPrimitive(fe_fs.hog_feature, [Img1], Img1, name='HoGF')
pset.addPrimitive(fe_fs.mixconadd, [Img1, Float3, Img1, Float3], Img1, name='W_AddF')
pset.addPrimitive(fe_fs.mixconsub, [Img1, Float3, Img1, Float3], Img1, name='W_SubF')
pset.addPrimitive(fe_fs.sqrt, [Img1], Img1, name='SqrtF')
pset.addPrimitive(fe_fs.relu, [Img1], Img1, name='ReLUF')
# pooling
pset.addPrimitive(fe_fs.maxP, [Img, Int3, Int3], Img1, name='MaxP')
# filtering
pset.addPrimitive(fe_fs.gau, [Img, Int1], Img, name='Gau')
pset.addPrimitive(fe_fs.gauD, [Img, Int1, Int2, Int2], Img, name='GauD')
pset.addPrimitive(fe_fs.gab, [Img, Float1, Float2], Img, name='Gabor')
pset.addPrimitive(fe_fs.laplace, [Img], Img, name='Lap')
pset.addPrimitive(fe_fs.gaussian_Laplace1, [Img], Img, name='LoG1')
pset.addPrimitive(fe_fs.gaussian_Laplace2, [Img], Img, name='LoG2')
pset.addPrimitive(fe_fs.sobelxy, [Img], Img, name='Sobel')
pset.addPrimitive(fe_fs.sobelx, [Img], Img, name='SobelX')
pset.addPrimitive(fe_fs.sobely, [Img], Img, name='SobelY')
pset.addPrimitive(fe_fs.medianf, [Img], Img, name='Med')
pset.addPrimitive(fe_fs.meanf, [Img], Img, name='Mean')
pset.addPrimitive(fe_fs.minf, [Img], Img, name='Min')
pset.addPrimitive(fe_fs.maxf, [Img], Img, name='Max')
pset.addPrimitive(fe_fs.lbp, [Img], Img, name='LBP_F')
pset.addPrimitive(fe_fs.hog_feature, [Img], Img, name='HOG_F')
pset.addPrimitive(fe_fs.mixconadd, [Img, Float3, Img, Float3], Img, name='W_Add')
pset.addPrimitive(fe_fs.mixconsub, [Img, Float3, Img, Float3], Img, name='W_Sub')
pset.addPrimitive(fe_fs.sqrt, [Img], Img, name='Sqrt')
pset.addPrimitive(fe_fs.relu, [Img], Img, name='ReLU')
# Terminals
pset.renameArguments(ARG0='Image')
pset.addEphemeralConstant('Singma', lambda: random.randint(1, 4), Int1)
pset.addEphemeralConstant('Order', lambda: random.randint(0, 3), Int2)
pset.addEphemeralConstant('Theta', lambda: random.randint(0, 8), Float1)
pset.addEphemeralConstant('Frequency', lambda: random.randint(0, 5), Float2)
pset.addEphemeralConstant('n', lambda: round(random.random(), 3), Float3)
pset.addEphemeralConstant('KernelSize', lambda: random.randrange(2, 5, 2), Int3)




@ignore_warnings(category=ConvergenceWarning)
def _evaluate(individual, compile, train_data, train_label):
    try:
        func = compile(individual)
        train_tf = []
        for i in range(0, len(train_label)):
            train_tf.append(numpy.asarray(func(train_data[i, :, :])))
        train_tf = numpy.asarray(train_tf, dtype=float)
        min_max_scaler = preprocessing.MinMaxScaler()
        train_norm = min_max_scaler.fit_transform(train_tf)
        lsvm = LinearSVC()
        accuracy = round(100 * cross_val_score(lsvm, train_norm, train_label, cv=5).mean(), 2)
    except:
        accuracy = 0
    return 100-accuracy,

def aggregate(populations, hof_size:int = 10):
    hof = tools.HallOfFame(hof_size)
    hof.update(populations)
    return hof

def run(init_population: list = [], population: int = 500, generation: int = 50, crossover_rate: float = 0.8, mutation_rate: float = 0.19, elitism_rate: float = 0.01, init_min_depth: int = 2, init_max_depth: int = 6, max_depth: int = 8, hof_size: int = 10, runs: int = 1, train_set = [], test_set = [], seed: int = random.randint(1, 100), tournment_size: int = 7, verbose: bool = False):
    ##GP
    toolbox = base.Toolbox()
    toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=init_min_depth, max_=init_max_depth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("mapp", map)
    # genetic operator
    toolbox.register("evaluate", _evaluate, compile=toolbox.compile, train_data=train_set[0], train_label=train_set[1])
    toolbox.register("select", tools.selTournament, tournsize=tournment_size)
    toolbox.register("selectElitism", tools.selBest)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))

    random.seed(seed)

    pop = init_population
    pop.extend(toolbox.population(n=population-len(init_population)))
    
    hof = tools.HallOfFame(hof_size)
    log = tools.Logbook()
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size_tree = tools.Statistics(key=len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size_tree=stats_size_tree)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)
    log.header = ["gen", "evals"] + mstats.fields
    log.chapters["fitness"].header = "min", "avg", "max", "std"
    log.chapters["size"].header = "min", "avg", "max", "std"

    pop, log = evalGP.eaSimple(pop, toolbox, crossover_rate, mutation_rate, elitism_rate, generation,
                            stats=mstats, halloffame=hof, verbose=verbose)
    return pop, log, hof

@ignore_warnings(category=ConvergenceWarning)
def test(individual, train_data, train_label, test_data, test_label):
    func = gp.compile(individual, pset)
    train_tf = []
    test_tf = []
    for i in range(0, len(train_label)):
        train_tf.append(numpy.asarray(func(train_data[i, :, :])))
    for j in range(0, len(test_label)):
        test_tf.append(numpy.asarray(func(test_data[j, :, :])))
    train_tf = numpy.asarray(train_tf, dtype=float)
    test_tf = numpy.asarray(test_tf, dtype=float)
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(train_tf)
    test_norm = min_max_scaler.transform(test_tf)
    lsvm= LinearSVC()
    lsvm.fit(train_norm, train_label)
    accuracy = round(100*lsvm.score(test_norm, test_label),2)
    return 100-accuracy

@ignore_warnings(category=ConvergenceWarning)
def classifier(individual, train_data, train_label):
    func = gp.compile(individual, pset)
    train_tf = []
    for i in range(0, len(train_label)):
        train_tf.append(numpy.asarray(func(train_data[i, :, :])))
    train_tf = numpy.asarray(train_tf, dtype=float)
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(train_tf)
    lsvm= LinearSVC()
    lsvm.fit(train_norm, train_label)
    return lambda img: lsvm.predict(min_max_scaler.transform([numpy.asarray(func(img), dtype=float)]))[0]

def results(ind, train_data, train_labels, test_data, test_labels):
    #ind = "Roots2(FGlobal_HOG(Image0), FGlobal_SIFT(Mean(Image0)))"
    func = classifier(ind, train_data, train_labels)
    for i, img in enumerate(test_data[:10]):
        print(f"expected:{test_labels[i]}; outcome:{func(img)}")


def individuals_to_strings(indviduals):
    return list(map(lambda x: str(x), indviduals))

def strings_to_individuals(strings):
    return list(map(lambda x: creator.Individual.from_string(x, pset), strings))

def string_to_individual(string):
    return creator.Individual.from_string(string, pset)

