import random
import pprint

MAX_GEN = 100
POP_SIZE = 50
IND_LEN = 25
MUTATION_PROB_PER_GEN = 1/IND_LEN
MUTATION_PROB = 0.1
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

chosen_fitness = fitness_OneMax

def population_fitness(pop, fitness=chosen_fitness):
    return [fitness(ind) for ind in pop]

def select(pop, fitness=chosen_fitness):
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
    

def evolutionary_algorithm(pop):
    log = []
    population = pop
    for _ in range(MAX_GEN):
        offspring = []
        log.append(sum(population_fitness(population, chosen_fitness))/len(population))
        mating_pool = select(population, chosen_fitness)
        for p1, p2 in zip(mating_pool[::2], mating_pool[1::2]):
            o1, o2 = crossover(p1, p2)
            o1, o2 = mutation(o1), mutation(o2)
            offspring.extend([o1, o2])
        population = offspring
    return population, log

pop = random_initial_population()
result, log = evolutionary_algorithm(pop)
#pprint.pprint(result)
print("First population fitness : ", sum(population_fitness(pop, chosen_fitness)), "Last population fitness : ", sum(population_fitness(result, chosen_fitness)))
print("Best individual : ", max(result, key=chosen_fitness))

def plot_log():
    import matplotlib.pyplot as plt
    plt.plot(log)
    plt.title("Mean fitness over generations")
    plt.show()
plot_log()

pop = random_initial_population()
result1, log1 = evolutionary_algorithm(pop)
print("First setting (Mutation prob = {}) : ".format(MUTATION_PROB))
print("First population fitness : ", sum(population_fitness(pop, chosen_fitness)), "Last population fitness : ", sum(population_fitness(result1, chosen_fitness)))
print("Best individual : ", max(result1, key=chosen_fitness))
MUTATION_PROB = 0.5
result2, log2 = evolutionary_algorithm(pop)
print("Second setting (Mutation prob = {}) : ".format(MUTATION_PROB))
print("First population fitness : ", sum(population_fitness(pop, chosen_fitness)), "Last population fitness : ", sum(population_fitness(result2, chosen_fitness)))
print("Best individual : ", max(result2, key=chosen_fitness))
#plot log and log1 together to compare
def plot_compare_log():
    import matplotlib.pyplot as plt
    plt.plot(log1, label="1st setting")
    plt.plot(log2, label="2nd setting")
    plt.legend()
    plt.title("Comparaison of the convergence between two settings")
    plt.show()

plot_compare_log()