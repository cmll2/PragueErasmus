import random
import numpy as np
import functools

import co_functions as cf
import utils

DIMENSION = 10 # dimension of the problems
POP_SIZE = 100 # population size
MAX_GEN = 2500 # maximum number of generations
CX_PROB = 0.8 # crossover probability
MUT_PROB = 0.4 # mutation probability
MUT_STEP = 0.1 # size of the mutation steps
REPEATS = 10 # number of runs of algorithm (should be at least 10)
OUT_DIR = 'continuous_LamarckGrad2500' # output directory for logs
EXP_ID = 'default' # the ID of this experiment (used to create log names)


# creates the individual
def create_ind(ind_len):
    return np.random.uniform(-5, 5, size=(ind_len,))

# creates the population using the create individual function
def create_pop(pop_size, create_individual):
    return [create_individual() for _ in range(pop_size)]

# the tournament selection (roulette wheell would not work, because we can have 
# negative fitness)
def tournament_selection(pop, fits, k):
    selected = []
    for i in range(k):
        p1 = random.randrange(0, len(pop))
        p2 = random.randrange(0, len(pop))
        if fits[p1] > fits[p2]:
            selected.append(np.copy(pop[p1]))
        else:
            selected.append(np.copy(pop[p2]))

    return selected

# implements the one-point crossover of two individuals
def one_pt_cross(p1, p2):
    point = random.randrange(1, len(p1))
    o1 = np.append(p1[:point], p2[point:])
    o2 = np.append(p2[:point], p1[point:])
    return o1, o2

# gaussian mutation - we need a class because we want to change the step
# size of the mutation adaptively
class Mutation:

    def __init__(self, step_size):
        self.step_size = step_size

    def __call__(self, ind):
        return ind + self.step_size*np.random.normal(size=ind.shape)
    
    def cool_down(self):
        return

class DerivativeMutation(Mutation):

    def __init__(self, step_size, fitness_function, prob=0.2):
        super().__init__(step_size)
        self.fitness_function = fitness_function
        self.prob = prob

    def __call__(self, ind): #use gradient to mutate individual accordingly to gradient's right direction
        if random.random() < self.prob:
            return ind + self.step_size*np.random.normal(size=ind.shape)
        else:
            return ind - self.step_size*cf.numerical_derivative(self.fitness_function, ind)
        
class AdaptiveRankMutation(Mutation): #This mutation changes step size adaptively according to the rank of the individual in the population
    def __init__(self, step_size, pop_size, prob = 0.2):
        self.step_size = step_size
        self.pop_size = pop_size
        self.prob = prob

    def __call__(self, ind, rank):
        if random.random() < self.prob * (rank/self.pop_size):
            return ind + self.step_size*np.random.normal(size=ind.shape)
        else:   
            return ind
class AdaptiveBoolMutation(Mutation): #This mutation changes mut prob adaptively according to the rank of the individual in the population
    def __init__(self, step_size, high_prob, low_prob):
        self.step_size = step_size
        self.high_prob = high_prob
        self.low_prob = low_prob

    def __call__(self, ind, bool): #bool is a boolean value that indicates if the individual is in the worst half of the population
        if bool:
            if random.random() < self.high_prob:
                return ind + self.step_size*np.random.normal(size=ind.shape)
        else:
            if random.random() < self.low_prob:
                return ind + self.step_size*np.random.normal(size=ind.shape)
        return ind
    
class DifferentialMutation(Mutation): #This mutation is an implementation of the differential evolution
    def __init__(self, F, CR, fitness_function):
        self.F = F
        self.CR = CR
        self.fitness_function = fitness_function
    def __call__(self, pop):
        pop_size = len(pop)
        new_pop = []
        for i in range(pop_size):
            r1, r2, r3, r4 = random.sample(pop, 4) #3 parents for the mutation and 1 for the crossover
            #change F uniformly in range [0.5, 1.0]
            self.F = random.uniform(0.5, 1.0)
            new_ind = r1 + self.F*(r2 - r3)
            for j in range(len(new_ind)):
                if random.random() > self.CR:
                    new_ind[j] = r4[j]
            if self.fitness_function(new_ind).fitness > self.fitness_function(pop[i]).fitness:
                new_pop.append(new_ind)
            else:
                new_pop.append(r1)
        return new_pop

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

def sort_population(pop, fitness):
    population_fitness = [fitness(p).fitness for p in pop]
    sorted_pop = [p for _, p in sorted(zip(population_fitness, pop), key=lambda pair: pair[0])]
    return sorted_pop

# applies the mutate function (implementing the mutation of a single individual)
# to the whole population with probability mut_prob)


def mutation(pop, mutate, mut_prob): #base mutation function
    mutate.cool_down()
    return [mutate(p) if random.random() < mut_prob else p[:] for p in pop]

def forced_mutation(pop, mutate, mut_prob): #mutation function that forces mutation on all individuals
    mutate.cool_down()
    return [mutate(p) for p in pop]

def mutation_bool(pop, mutate, mut_prob, fit): #mutation function that changes mutation probability adaptively
    fit = list(map(fit, pop))
    average_fitness = np.mean(fit)
    worst_half = [f.fitness < average_fitness for f in fit]
    return [mutate(p, bool) for p, bool in zip(pop, worst_half)]


def mutation_rank(pop, mutate, mut_prob, fit): #mutation function that changes mutation probability adaptively to the rank of the individual in the population
    sorted_pop = sort_population(pop, fit)
    N = len(sorted_pop)
    sorted_pop = [mutate(p, i) for i, p in enumerate(sorted_pop)]
    return sorted_pop

def mutation_diff(pop, mutate):
    mutate.cool_down()
    return mutate(pop)

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
#   mutate_ind - reference to the class to mutate an individual - can be used to 
#               change the mutation step adaptively
#   map_fn    - function to use to map fitness evaluation over the whole 
#               population (default `map`)
#   log       - a utils.Log structure to log the evolution run
def evolutionary_algorithm(pop, max_gen, fitness, operators, mate_sel, mutate_ind, *, map_fn=map, log=None):
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
import math
class SimulatedAnnealingMutation(Mutation):
    def __init__(self, initial_temperature, cooling_rate, fitness_function, step_size=0.4):
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.fitness_function = fitness_function
        self.step_size = step_size

    def mutate(self, ind):
        mutated_ind = ind + self.step_size * np.random.normal(size=ind.shape)
        return mutated_ind

    def accept(self, ind, mutated_ind):
        current_fitness = self.fitness_function(ind).fitness
        mutated_fitness = self.fitness_function(mutated_ind).fitness

        if mutated_fitness > current_fitness:
            return mutated_ind
        else:
            probability = np.exp((mutated_fitness - current_fitness) / self.temperature)
            if np.random.rand() < probability:
                return mutated_ind
            else:
                return ind

    def anneal(self, ind):
        mutated_ind = self.mutate(ind)
        return self.accept(ind, mutated_ind)

    def cool_down(self):
        self.temperature *= self.cooling_rate
        self.step_size *= self.cooling_rate

    def __call__(self, ind):
        mutated_ind = self.anneal(ind)
        return mutated_ind
    
class LamarckMutation(Mutation):
    def __init__(self, step_size, fitness_function, improve_iterations):
        super().__init__(step_size)
        self.fitness_function = fitness_function
        self.improve_iterations = improve_iterations

    def improve_ind(self, ind):
        best_fitness = self.fitness_function(ind).fitness
        best_ind = ind.copy()

        for _ in range(self.improve_iterations):
            new_ind = ind = ind - self.step_size*cf.numerical_derivative(self.fitness_function, ind)
            new_fitness = self.fitness_function(new_ind).fitness
            if new_fitness > best_fitness:
                best_fitness = new_fitness
                best_ind = new_ind
        return best_ind
    
    def cool_down(self):
        self.step_size *= 0.99
    
    def __call__(self, ind):
        improved_ind = self.improve_ind(ind)
        
        # Additional mutation after improvement
        mutated_improved_ind = improved_ind + self.step_size * np.random.normal(size=improved_ind.shape)
        return mutated_improved_ind
    
    def __call__(self, ind):
        if random.random() < self.prob:
            return self.improve_ind(ind)
        else:
            return ind + self.step_size*np.random.normal(size=ind.shape)

if __name__ == '__main__':

    # use `functool.partial` to create fix some arguments of the functions 
    # and create functions with required signatures
    cr_ind = functools.partial(create_ind, ind_len=DIMENSION)
    # we will run the experiment on a number of different functions
    fit_generators = [cf.make_f01_sphere,
                      cf.make_f02_ellipsoidal,
                      cf.make_f06_attractive_sector,
                      cf.make_f08_rosenbrock,
                      cf.make_f10_rotated_ellipsoidal]
    fit_names = ['f01', 'f02', 'f06', 'f08', 'f10']

    for fit_gen, fit_name in zip(fit_generators, fit_names):
        fit = fit_gen(DIMENSION)
        if fit_name == 'f01' or fit_name == 'f02' or fit_name == 'f10':
            #separable functions so CR = 0.2
            CR = 0.2
        else:
            #non-separable functions so CR = 0.9
            CR = 0.9
        # adaptive rank mutation
        # mutate_ind = AdaptiveRankMutation(MUT_STEP, POP_SIZE, MUT_PROB)
        # mut = functools.partial(mutation_rank, mut_prob=MUT_PROB, mutate=mutate_ind, fit=fit)
        # adaptive bool mutation
        # mutate_ind = AdaptiveBoolMutation(MUT_STEP, 0.4, 0.05)
        # mut = functools.partial(mutation_bool, mut_prob=MUT_PROB, mutate=mutate_ind, fit=fit)
        #default mutation
        # mutate_ind = SimulatedAnnealingMutation(50, 0.99, fit)
        # mut = functools.partial(mutation, mut_prob=MUT_PROB, mutate=mutate_ind)
        # differential evolution / Lamarck mutation / Baldwin mutation / Simulated Annealing
        temp = 100
        #mutate_ind = LamarckMutation(MUT_STEP, fit, 5)
        xover = functools.partial(crossover, cross=one_pt_cross, cx_prob=CX_PROB)

        # run the algorithm `REPEATS` times and remember the best solutions from 
        # last generations
    
        best_inds = []
        for run in range(REPEATS):
            mutate_ind = LamarckMutation(MUT_STEP, fit, 5)
            mut = functools.partial(forced_mutation, mut_prob=MUT_PROB, mutate=mutate_ind)
            # initialize the log structure
            log = utils.Log(OUT_DIR, EXP_ID + '.' + fit_name , run, 
                            write_immediately=True, print_frequency=5)
            # create population
            pop = create_pop(POP_SIZE, cr_ind)
            # run evolution - notice we use the pool.map as the map_fn
            # pop = evolutionary_algorithm(pop, MAX_GEN, fit, [xover, mut], tournament_selection, mutate_ind, map_fn=map, log=log) #default
            pop = evolutionary_algorithm(pop, MAX_GEN, fit, [xover, mut], tournament_selection, mutate_ind, map_fn=map, log=log) #differential evolution
            # remember the best individual from last generation, save it to file
            bi = max(pop, key=fit)
            best_inds.append(bi)
            
            # if we used write_immediately = False, we would need to save the 
            # files now
            # log.write_files()

        # print an overview of the best individuals from each run
        for i, bi in enumerate(best_inds):
            print(f'Run {i}: objective = {fit(bi).objective}')

        # write summary logs for the whole experiment
        utils.summarize_experiment(OUT_DIR, EXP_ID + '.' + fit_name)