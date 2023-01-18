#System
TEST = True
SIMULATE = True
VERBOSE = True
SEED = 23

#Datasets
DATASET = "mnist-rot" #mnist|mnist-rot|mnist-back-image|mnist-back-rand|mnist-rot-back-image
TRAIN_SIZE = 1000
TEST_SIZE = 100

#FL
CLIENTS = 2
ROUNDS = 1

#GP
RUNS = 1
POPULATION = 10      #500
GENERATION = 5       #50
CROSSOVER = 0.8
MUTATION = 0.2       #0.19
ELITISM = 0.01
TOURNMENT_SIZE = 7
INIT_MIN_DEPTH = 2
INIT_MAX_DEPTH = 6
MAX_DEPTH = 8
HOF_SIZE = 1
