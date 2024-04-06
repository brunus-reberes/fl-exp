#System
TEST = False
SIMULATE = True
VERBOSE = True
SEED = 23

#FL
CLIENTS = 3
ROUNDS = 30

#Datasets
DATASET = "mnist-rot" #mnist|mnist-rot|mnist-back-image|mnist-back-rand|mnist-rot-back-image
TRAIN_SIZE = None #ignored if batch_size not none
TEST_SIZE = None
TRAIN_BATCH_SIZE = int(12000/CLIENTS)
TEST_BATCH_SIZE = int(60000/CLIENTS)

#GP
RUNS = 1
POPULATION = 500      #500
GENERATION = 50       #50
CROSSOVER = 0.8
MUTATION = 0.19       #0.19
ELITISM = 0.01
TOURNMENT_SIZE = 7
INIT_MIN_DEPTH = 2
INIT_MAX_DEPTH = 6
MAX_DEPTH = 8
HOF_SIZE = 10
