from deap import gp, creator, base, tools, algorithms
import operator
import random
import math
import numpy as np

creator.create("Fitness", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)
def _protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1
pset = gp.PrimitiveSet("MAIN", 1, "x")
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(_protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))


def individuals_to_strings(indviduals):
    return list(map(lambda x: str(x), indviduals))

def strings_to_individuals(strings):
    return list(map(lambda x: creator.Individual.from_string(x, pset), strings))

def aggregate(populations, hof_size:int = 10):
    hof = tools.HallOfFame(hof_size)
    hof.update(populations)
    return hof

def _evaluate(individual, compile):
        func = compile(individual)
        points = [x/10. for x in range(-10,10)]
        sqerrors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in points)
        return math.fsum(sqerrors) / len(points),

def run(init_population: list = [], population: int = 500, generation: int = 50, crossover_rate: float = 0.8, mutation_rate: float = 0.19, elitism_rate: float = 0.01, init_min_depth: int = 2, init_max_depth: int = 6, max_depth: int = 8, hof_size: int = 10, runs: int = 1, train_set = [], test_set = [], seed: int = random.randint(1,100), tournment_size: int = 7, verbose: bool = False):
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

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    #converting strings into individuals
    pop = init_population
    pop.extend(toolbox.population(n=population-len(init_population)))

    hof = tools.HallOfFame(hof_size)
    pop, log = algorithms.eaSimple(pop, toolbox, crossover_rate, mutation_rate, generation, stats=mstats,
                                    halloffame=hof, verbose=verbose)
    return pop, log, hof

def test(individual, train_data, train_label, test_data, test_label):
    func = gp.compile(individual, pset)
    points = [x/10. for x in range(-10,10)]
    sqerrors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in points)
    return math.fsum(sqerrors) / len(points)