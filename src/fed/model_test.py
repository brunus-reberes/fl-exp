from deap import gp, creator, base, tools, algorithms
import operator
import random
import math
import numpy
from typing import Callable
import numpy as np

if not hasattr(creator, "Individual") and not hasattr(creator, "Fitness"):
    creator.create("Fitness", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)

def _protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def aggregate(populations, hof_size:int = 10):
    hof = tools.HallOfFame(hof_size)
    for pop in populations:
        individuals = np.array([creator.Individual.from_string(ind, _primitives()) for ind in pop])
        hof.update(individuals)
    hof_str = []
    for ind in hof:
        hof_str.append(str(ind))
    return hof_str

def _evaluate(individual, compile):
        func = compile(individual)
        points = [x/10. for x in range(-10,10)]
        sqerrors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in points)
        return math.fsum(sqerrors) / len(points),

def _primitives():
    pset = gp.PrimitiveSet("MAIN", 1, "x")
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(_protectedDiv, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(math.sin, 1)
    pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
    return pset

def run(init_population: list = [], population: int = 500, generation: int = 50, crossover_rate: float = 0.8, mutation_rate: float = 0.19, elitism_rate: float = 0.01, init_min_depth: int = 2, init_max_depth: int = 6, max_depth: int = 8, hof_size: int = 10, runs: int = 1, train_data = [], test_data = [], seed: int = random.randint(), tournment_size: int = 7, verbose: bool = False):
    pset = _primitives()

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=init_min_depth, max_=init_max_depth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", _evaluate, compile=toolbox.compile)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=init_min_depth, max_=init_max_depth)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))


    #converting strings into individuals
    pop = []
    for ind in init_population:
        pop.append(creator.Individual.from_string(ind, pset))

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop.extend(toolbox.population(n=population-len(init_population)))
    hof = tools.HallOfFame(hof_size)
    pop, log = algorithms.eaSimple(pop, toolbox, crossover_rate, mutation_rate, generation, stats=mstats,
                                    halloffame=hof, verbose=verbose)
    #converting individuals to strings
    new_hof = []
    for ind in hof:
        new_hof.append(str(ind))
    return pop, log, new_hof
