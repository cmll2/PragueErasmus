import random
import numpy as np
import functools

import utils

K = 10 #number of piles
POP_SIZE = 400 # population size
MAX_GEN = 10000 # maximum number of generations
CX_PROB = 0.4 # crossover probability
MUT_PROB = 0.6 # mutation probability
MUT_FLIP_PROB = 0.05 # probability of changing value during mutation
MUT_SWAP_PROB = 0.8 # probability of swapping value during mutation
REPEATS = 10 # number of runs of algorithm (should be at least 10)
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

def sus_selection(pop, fits, k):
    sum_fits = sum(fits)
    dist = sum_fits / k
    current_point = random.uniform(0, dist)
    selected_individuals = []
    for _ in range(k):
        current_fitness_sum = 0
        selected_individual = None
        for ind in pop:
            current_fitness_sum += fitness(ind, weights).fitness
            if current_fitness_sum > current_point:
                selected_individual = ind
                break
        selected_individuals.append(selected_individual)
        current_point += dist
    return selected_individuals
        

# creates the individual
def create_ind(ind_len):
    return [random.randrange(0, K) for _ in range(ind_len)]

# creates the population using the create individual function
def create_pop(pop_size, create_individual):
    return [create_individual() for _ in range(pop_size)]

# the roulette wheel selection
def roulette_wheel_selection(pop, fits, k):
    return random.choices(pop, fits, k=k)

# implements the one-point crossover of two individuals
def one_pt_cross(p1, p2):
    point = random.randrange(1, len(p1))
    o1 = p1[:point] + p2[point:]
    o2 = p2[:point] + p1[point:]
    return o1, o2

#implements the two-point crossover of two individuals
def two_pt_cross(p1, p2):
    point1 = random.randrange(1, len(p1))
    point2 = random.randrange(1, len(p1))
    if point1 > point2:
        point1, point2 = point2, point1
    o1 = p1[:point1] + p2[point1:point2] + p1[point2:]
    o2 = p2[:point1] + p1[point1:point2] + p2[point2:]
    return o1, o2

def three_pt_cross(p1, p2):
    point1 = random.randrange(1, len(p1))
    point2 = random.randrange(1, len(p1))
    point3 = random.randrange(1, len(p1))
    if point1 > point2:
        point1, point2 = point2, point1
    if point2 > point3:
        point2, point3 = point3, point2
    if point1 > point2:
        point1, point2 = point2, point1
    o1 = p1[:point1] + p2[point1:point2] + p1[point2:point3] + p2[point3:]
    o2 = p2[:point1] + p1[point1:point2] + p2[point2:point3] + p1[point3:]
    return o1, o2

# implements the "bit-flip" mutation of one individual
def flip_mutate(p, prob, upper):
    return [random.randrange(0, upper) if random.random() < prob else i for i in p]

# implements the "swap" mutation of one individual
def swap_mutate(p):
    point1 = random.randrange(0, len(p))
    point2 = random.randrange(0, len(p))
    p[point1], p[point2] = p[point2], p[point1]
    return p

#combine every mutation functions
def whole_mutation(p, flip_prob, upper):
    first_mutation = flip_mutate(p, flip_prob, upper)
    second_mutation = swap_mutate(first_mutation)
    return second_mutation


# applies a list of genetic operators (functions with 1 argument - population) 
# to the population
def mate(pop, operators):
    for o in operators:
        pop = o(pop)
    return pop

# applies the cross function (implementing the crossover of two individuals)
# to the whole population (with probability cx_prob)
def crossover(pop, cross, cx_prob, fitness=None):
    if fitness is None:
        off = []
        for p1, p2 in zip(pop[0::2], pop[1::2]):
            if random.random() < cx_prob:
                o1, o2 = cross(p1, p2)
            else:
                o1, o2 = p1[:], p2[:]
            off.append(o1)
            off.append(o2)
    else:
        off = []
        max_fit = max(fitness(p).fitness for p in pop)
        min_fit = min(fitness(p).fitness for p in pop)
        if min_fit == max_fit:
            diff = 1
        else:
            diff = max_fit - min_fit
        for p1, p2 in zip(pop[0::2], pop[1::2]):
            fitness1 = fitness(p1).fitness
            fitness2 = fitness(p2).fitness
            if fitness1 > fitness2:
                fitness1, fitness2 = fitness2, fitness1
            if random.random() < cx_prob * (fitness1 / fitness2) * (1 - (fitness1 - min_fit) / diff):
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

def mutation(pop, mutate, mut_prob, fitness=None):
    if fitness is None:
        return [mutate(p) if random.random() < mut_prob else p[:] for p in pop]
    else:
        sorted_pop = sort_population(pop, fitness)
        return [mutate(p) if random.random() < mut_prob*(1-(i/(len(sorted_pop)-1))) else p[:] for i, p in enumerate(sorted_pop)]

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

        mating_pool = mate_sel(pop, fits, POP_SIZE)
        offspring = mate(mating_pool, operators)
        pop = offspring[:]

    return pop

if __name__ == '__main__':
    # read the weights from input
    weights = read_weights('../evaTeaching-python/inputs/partition-easy.txt')
    MUT_FLIP_PROB = 1 / len(weights)
    # use `functool.partial` to create fix some arguments of the functions 
    # and create functions with required signatures
    cr_ind = functools.partial(create_ind, ind_len=len(weights))
    fit = functools.partial(fitness, weights=weights)
    xover = functools.partial(crossover, cross=two_pt_cross, cx_prob=CX_PROB, fitness = fit)
    mut = functools.partial(mutation, mut_prob=MUT_PROB, 
                            mutate=functools.partial(whole_mutation, flip_prob=MUT_FLIP_PROB, upper=K), fitness=fit)

    # we can use multiprocessing to evaluate fitness in parallel
    import multiprocessing
    pool = multiprocessing.Pool()

    import matplotlib.pyplot as plt

    # run the algorithm `REPEATS` times and remember the best solutions from 
    # last generations
    best_inds = []
    for run in range(REPEATS):
        # initialize the log structure
        log = utils.Log(OUT_DIR, EXP_ID, run, 
                        write_immediately=True, print_frequency=5)
        # create population
        pop = create_pop(POP_SIZE, cr_ind)
        # run evolution - notice we use the pool.map as the map_fn
        pop = evolutionary_algorithm(pop, MAX_GEN, fit, [xover, mut], roulette_wheel_selection, map_fn=pool.map, log=log)
        # remember the best individual from last generation, save it to file
        bi = max(pop, key=fit)
        best_inds.append(bi)

        with open(f'{OUT_DIR}/{EXP_ID}_{run}.best', 'w') as f:
            for w, b in zip(weights, bi):
                f.write(f'{w} {b}\n')
        
        # if we used write_immediately = False, we would need to save the 
        # files now
        # log.write_files()

    # print an overview of the best individuals from each run
    for i, bi in enumerate(best_inds):
        print(f'Run {i}: difference = {fit(bi).objective}, bin weights = {bin_weights(weights, bi)}')

    # write summary logs for the whole experiment
    utils.summarize_experiment(OUT_DIR, EXP_ID)

    # read the summary log and plot the experiment
    evals, lower, mean, upper = utils.get_plot_data(OUT_DIR, EXP_ID)
    plt.figure(figsize=(12, 8))
    utils.plot_experiment(evals, lower, mean, upper, legend_name = 'Default settings')
    plt.legend()
    plt.show()

    # you can also plot mutiple experiments at the same time using 
    # utils.plot_experiments, e.g. if you have two experiments 'default' and 
    # 'tuned' both in the 'partition' directory, you can call
    # utils.plot_experiments('partition', ['default', 'tuned'], 
    #                        rename_dict={'default': 'Default setting'})
    # the rename_dict can be used to make reasonable entries in the legend - 
    # experiments that are not in the dict use their id (in this case, the 
    # legend entries would be 'Default settings' and 'tuned') 