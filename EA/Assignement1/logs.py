import random
import pprint
import numpy as np
import matplotlib.pyplot as plt

MAX_GEN = 200
POP_SIZE = 50
IND_LEN = 25
MUTATION_PROB_PER_GEN = 1/IND_LEN
MUTATION_PROB = 0.5
CROSSOVER_PROB = 0.8

def random_individual(length=IND_LEN):
    return [random.randint(0, 1) for _ in range(length)]

def random_initial_population(size=POP_SIZE):
    return [random_individual() for _ in range(size)]

def fitness_OneMax(individual):
    return sum(individual)

def fitness_alternative(individual):
    fitness = 0
    if individual[0] != individual[1]:
        fitness += 1
    for i in range(1, len(individual)-1):
        if individual[i] != individual[i-1] and individual[i] != individual[i+1]:
            fitness += 1
    if individual[-1] != individual[-2]:
        fitness += 1
    return fitness

def population_fitness(pop, fitness):
    return [fitness(ind) for ind in pop]

def select(pop, fitness):
    fits = population_fitness(pop, fitness)
    return random.choices(pop, fits, k=len(pop))

def crossover(ind1, ind2, prob=CROSSOVER_PROB):
    if random.random() > prob:
        return ind1, ind2
    crossover_point = random.randrange(0, IND_LEN)
    o1 = ind1[:crossover_point] + ind2[crossover_point:]
    o2 = ind2[:crossover_point] + ind1[crossover_point:]
    return o1, o2

def mutation(ind, prob=MUTATION_PROB, prob_per_gen=MUTATION_PROB_PER_GEN):
    if random.random() > prob:
        return ind
    return [1-i if random.random() < prob_per_gen else i for i in ind]
    
def evolutionary_algorithm(pop, fitness):
    log = []
    population = pop
    for _ in range(MAX_GEN):
        offspring = []
        log.append(sum(population_fitness(population, fitness))/len(population))
        mating_pool = select(population, fitness)
        for p1, p2 in zip(mating_pool[::2], mating_pool[1::2]):
            o1, o2 = crossover(p1, p2)
            o1, o2 = mutation(o1), mutation(o2)
            offspring.extend([o1, o2])
        population = offspring
    return population, log

def run_ea(fitness, n_runs=10):
    logs = []
    best_fit = 0
    for _ in range(n_runs):
        pop = random_initial_population()
        result, log = evolutionary_algorithm(pop, fitness)
        #pprint.pprint(result)
        #print("First population fitness : ", sum(population_fitness(pop, chosen_fitness)), "Last population fitness : ", sum(population_fitness(result, chosen_fitness)))
        best_ind = max(result, key=fitness)
        if fitness(best_ind) > best_fit:
            best_overall_ind = best_ind
            best_fit = fitness(best_ind)
        logs.append(log)
    print("Best overall individual : ", best_overall_ind)
    logs = np.array(logs)
    return logs

def plot_logs(logs, title = ''):
    p25 = np.percentile(logs, 25, axis=0)
    p75 = np.percentile(logs, 75, axis=0)
    mean = logs.mean(axis=0)
    plt.plot(mean, label="mean")
    plt.fill_between(list(range(MAX_GEN)), p25, p75, alpha = 0.5,label="interquartile")
    plt.legend()
    plt.title(title)
    plt.show()

def plot_two_logs(logs1, logs2, legend1, legend2, title = ''):
    p25_1 = np.percentile(logs1, 25, axis=0)
    p75_1 = np.percentile(logs1, 75, axis=0)
    mean_1 = logs1.mean(axis=0)
    plt.plot(mean_1, label="mean " + legend1)
    plt.fill_between(list(range(MAX_GEN)), p25_1, p75_1, alpha = 0.5,label="interquartile " + legend1)
    p25_2 = np.percentile(logs2, 25, axis=0)
    p75_2 = np.percentile(logs2, 75, axis=0)
    mean_2 = logs2.mean(axis=0)
    plt.plot(mean_2, label="mean " + legend2)
    plt.fill_between(list(range(MAX_GEN)), p25_2, p75_2, alpha = 0.5,label="interquartile " + legend2)
    plt.title(title)
    plt.legend()
    plt.show()

fitness = fitness_OneMax
logs1 = run_ea(fitness, 20)
CROSSOVER_PROB = 0.4
MUTATION_PROB = 0.1
logs2 = run_ea(fitness, 20)

plot_two_logs(logs1, logs2, 'Crossover Prob = 0.8', 'Crossover Prob = 0.4', 'Comparaison between two Crossover Prob for OneMAX')