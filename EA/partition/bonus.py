import random
import numpy as np
import functools

import utils

#add argument --file to read the file name from command line
import sys
if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = None

K = 5  #number of piles
POP_SIZE =  400 # population size
MAX_GEN = 1000 # maximum number of generations
CX_PROB = 0.04 # crossover probability
MUT_PROB = 0.4 # mutation probability
MUT_FLIP_PROB = 0.1 # probability of changing value during mutation
REPEATS = 1 # number of runs of algorithm (should be at least 10)
OUT_DIR = 'partition' # output directory for logs
EXP_ID = 'default' # the ID of this experiment (used to create log names)

# reads the input set of values of objects
def read_weights(filename):
    with open(filename) as f:
        return list(map(int, f.readlines()))

# computes the bin weights
# - bins are the indices of bins into which the object belongs
def bin_weights(weights, bins):
    bw = [0]*K
    for w, b in zip(weights, bins):
        bw[b] += w
    return bw

# the fitness function
def fitness(ind, weights):
    bw = bin_weights(weights, ind)
    return utils.FitObjPair(fitness=1/(max(bw) - min(bw) + 1), 
                            objective=max(bw) - min(bw))

# creates the individual
def create_ind(ind_len):
    return [random.randrange(0, K) for _ in range(ind_len)]

# creates the population using the create individual function
def create_pop(pop_size, create_individual):
    return [create_individual() for _ in range(pop_size)]

# the roulette wheel selection
def roulette_wheel_selection(pop, fits, k):
    return random.choices(pop, fits, k=k)

def tournament(pop, fits):
    tournament_size = 2
    tournament = random.sample(list(zip(pop, fits)), tournament_size)
    return max(tournament, key=lambda pair: pair[1])[0]

def tournament_selection(pop, fits, k):
    return [tournament(pop, fits) for _ in range(k)]

#best individual selection
def best_individual_selection(pop, fits, k):
    return [p for _, p in sorted(zip(fits, pop), key=lambda pair: pair[0], reverse=False)][:k]

# implements the one-point crossover of two individuals
def one_pt_cross(p1, p2):
    point = random.randrange(1, len(p1))
    o1 = p1[:point] + p2[point:]
    o2 = p2[:point] + p1[point:]
    return o1, o2

# implements the "bit-flip" mutation of one individual
def flip_mutate(p, prob, upper):
    return [random.randrange(0, upper) if random.random() < prob else i for i in p]

# applies a list of genetic operators (functions with 1 argument - population) 
# to the population
def mate(pop, operators):
    for o in operators:
        pop = o(pop)
    return pop

# applies the cross function (implementing the crossover of two individuals)
# to the whole population (with probability cx_prob)
def crossover(pop, cross, cx_prob):
    off = []
    for p1, p2 in zip(pop[0::2], pop[1::2]):
        if random.random() < cx_prob:
            o1, o2 = cross(p1, p2)
        else:
            o1, o2 = p1[:], p2[:]
        off.append(o1)
        off.append(o2)
    return off

# applies the mutate function (implementing the mutation of a single individual)
# to the whole population with probability mut_prob)

def sort_population(pop, fitness):
    population_fitness = [fitness(p).fitness for p in pop]
    sorted_pop = [p for _, p in sorted(zip(population_fitness, pop), key=lambda pair: pair[0])]
    return sorted_pop

def mutation(pop, mutate, mut_prob, fitness):
    sorted_pop = sort_population(pop, fitness)
    N = len(sorted_pop)
    for i in range(2):
        sorted_pop = [mutate(p) if random.random() < mut_prob * (N - j) / N else p for j, p in enumerate(sorted_pop)]
    return sorted_pop

# implements the evolutionary algorithm
# arguments:
#   pop_size  - the initial population
#   max_gen   - maximum number of generation
#   fitness   - fitness function (takes individual as argument and returns 
#               FitObjPair)
#   operators - list of genetic operators (functions with one arguments - 
#               population; returning a population)
#   mate_sel  - mating selection (funtion with three arguments - population, 
#               fitness values, number of individuals to select; returning the 
#               selected population)
#   map_fn    - function to use to map fitness evaluation over the whole 
#               population (default `map`)
#   log       - a utils.Log structure to log the evolution run
def evolutionary_algorithm(pop, max_gen, fitness, operators, mate_sel, *, map_fn=map, log=None):
    evals = 0
    for G in range(max_gen):
        fits_objs = list(map_fn(fitness, pop))
        evals += len(pop)
        if log:
            log.add_gen(fits_objs, evals)
        fits = [f.fitness for f in fits_objs]
        objs = [f.objective for f in fits_objs]
        #print(min(objs), max(objs), sum(objs)/len(objs))
        mating_pool = mate_sel(pop, fits, POP_SIZE)
        offspring = mate(mating_pool, operators)
        pop = offspring[:]

    return pop

if __name__ == '__main__':
    # read the weights from input
    weights = read_weights('../evaTeaching-python/inputs/partition-easy.txt')
    import multiprocessing
    pool = multiprocessing.Pool()
    import matplotlib.pyplot as plt
    if filename is None:
        MUT_FLIP_PROB = 1/(1*len(weights))
        # use `functool.partial` to create fix some arguments of the functions 
        # and create functions with required signatures
        cr_ind = functools.partial(create_ind, ind_len=len(weights))
        fit = functools.partial(fitness, weights=weights)
        xover = functools.partial(crossover, cross=one_pt_cross, cx_prob=CX_PROB)
        mut = functools.partial(mutation, mut_prob=MUT_PROB, 
                                mutate=functools.partial(flip_mutate, prob=MUT_FLIP_PROB, upper=K), fitness = fit)

        # we can use multiprocessing to evaluate fitness in parallel

        #first, run the algorithm for 5 bins, then for each of the weight lists for the 5 bins, re run it for 2 bins to have equally splitted in 10 bins

        #first 5 bins : 

        # run the algorithm REPEATS times
        best_inds = []
        for run in range(REPEATS):
            # create the initial population
            pop = create_pop(POP_SIZE, cr_ind)
            # run evolution - notice we use pool.map for fitness evaluation
            pop = evolutionary_algorithm(pop, MAX_GEN, fit, [xover, mut], roulette_wheel_selection, map_fn=pool.map)
            # remember the best individual from the last generation, save it to best_inds
            best_inds.append(max(pop, key=fit))


        best_ind = max(best_inds, key=fit)
        print("Objective for the best 5 bin individual : ", fitness(best_ind, weights).objective)
        #now we have the best solution for 5 bins, we can use it to create the 10 bins
        #create the new weights list
        #save the best individual to file.txt
        with open(f'bonus5.best', 'w') as f:
            for w, b in zip(weights, best_ind):
                f.write(f'{w} {b}\n')
    else:
        with open(filename) as f:
            best_ind = [int(line.split()[1]) for line in f.readlines()]
    print("Objective for the best 5-bin individual : ", fitness(best_ind, weights).objective)

    weights_5 = []
    for i in range(K):
        weights_i = [weights[ind] for ind, bin in enumerate(best_ind) if bin == i]
        weights_5.append(weights_i)
    print("Weights for the 5 bins : ", weights_5)
    #now we have our new weights lists and we can run the algorithm again for 2 bins for each of the weights lists
    best_inds = []
    K = 2
    for i in range (len(weights_5)):
        MUT_FLIP_PROB = 1/(1*len(weights_5[i]))
        cr_ind = functools.partial(create_ind, ind_len=len(weights_5[i]))
        fit = functools.partial(fitness, weights=weights_5[i])
        xover = functools.partial(crossover, cross=one_pt_cross, cx_prob=CX_PROB)
        mut = functools.partial(mutation, mut_prob=MUT_PROB, 
                                mutate=functools.partial(flip_mutate, prob=MUT_FLIP_PROB, upper=K), fitness = fit)
        for run in range(REPEATS):
            # create the initial population
            pop = create_pop(POP_SIZE, cr_ind)
            # run evolution - notice we use pool.map for fitness evaluation
            pop = evolutionary_algorithm(pop, MAX_GEN, fit, [xover, mut], roulette_wheel_selection      , map_fn=pool.map)
            # remember the best individual from the last generation, save it to best_inds
            best_ind = max(pop, key=fit)
        print(best_ind)
        print("Objective for {}-th best individual for 2 bins : ".format(i+1), fitness(best_ind, weights_5[i]).objective)
        best_inds.append(best_ind)

    #now, compute back the best individual for 10 bins
    i=0
    best_overall_ind = []
    for best_ind, weights in zip(best_inds, weights_5):
        for j in range(len(best_ind)):
            best_overall_ind.append(i if best_ind[j] == 0 else i+1)
        i+=2
    def flatten_list(L):
        return [item for sublist in L for item in sublist]
    flattened_weights = flatten_list(weights_5)
    print("Best individual for 10 bins : ", best_overall_ind)
    K = 10
    print("Final Objective for 10 bins : ", fitness(best_overall_ind, flattened_weights ).objective)
    
    #save best ind to file.txt
    with open(f'bonus.best', 'w') as f:
        for w, b in zip(flattened_weights, best_overall_ind):
            f.write(f'{w} {b}\n')

    #save new weight list to file.txt
    with open(f'bonus.weights', 'w') as f:
        for w in flattened_weights:
            f.write(f'{w}\n')
    
        

