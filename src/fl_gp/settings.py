#System
VERBOSE = True
SEED = 23
LOGS = "logs"
PICKLES = "pickles"

#FL
CLIENTS = 1
ROUNDS = 2

#Datasets
DATASET = "mnist" #mnist|mnist-rot|mnist-back-image|mnist-back-rand|mnist-rot-back-image
TRAIN_SIZE = 100  #12000      #ignored if batch_size not none
TEST_SIZE = 100   #60000

#GP
RUNS = 1
POPULATION = 10      #500
GENERATION = 5       #50
HOF_SIZE = 2

CROSSOVER = 0.8
MUTATION = 0.19       #0.19
ELITISM = 0.01
TOURNMENT_SIZE = 7
INIT_MIN_DEPTH = 2
INIT_MAX_DEPTH = 6
MAX_DEPTH = 8
