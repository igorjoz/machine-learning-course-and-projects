from itertools import compress
import random
import time
import matplotlib.pyplot as plt

from data import *


def initial_population(individual_size, population_size):
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]


def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))


def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness


items, knapsack_max_capacity = get_big()
print(items)

population_size = 200
generations = 400
n_selection = 180
n_elite = 2
mutation_rate = 0.1

n_selection = 120
mutation_rate = 0.2
n_elite = 1

# population_size = 1000
# generations = 1000
# n_selection = 600

start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []
population = initial_population(len(items), population_size)
for _ in range(generations):
    population_history.append(population)

    # TODO: implement genetic algorithm
    # evaluate fitness for the entire population
    fitness_scores = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    fitness_scores_sum = sum(fitness_scores)

    # important: selection - select top n individuals based on their fitness
    probabilities = []
    for score in fitness_scores:
        probability = score / fitness_scores_sum
        probabilities.append(probability)

    selected_individuals = random.choices(population, weights=probabilities, k=n_selection)

    # indexed_fitness_scores = list(enumerate(fitness_scores))
    # print(indexed_fitness_scores)
    # sorted_indexed_fitness_scores = sorted(indexed_fitness_scores, key=lambda pair: pair[1], reverse=True)
    # selected_indices = [index for index, _ in sorted_indexed_fitness_scores[:n_selection]]
    # selected_individuals = [population[i] for i in selected_indices]

    # important: crossover - create new individuals by combining genes of selected individuals
    new_population = selected_individuals[:n_elite]  # Preserve elite individuals unchanged
    while len(new_population) < population_size:
        parent1, parent2 = random.sample(selected_individuals, 2)
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        new_population.extend([child1, child2])

    # important: mutation - introduce random changes
    for individual in new_population[n_elite:]:  # Skip elite
        if random.random() < mutation_rate:
            mutation_point = random.randint(0, len(individual) - 1)
            individual[mutation_point] = not individual[mutation_point]

    population = new_population

    # append historic data
    best_individual, best_individual_fitness = population_best(items, knapsack_max_capacity, population)
    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness
    best_history.append(best_fitness)

# calculate algorithm running time
end_time = time.time()
total_time = end_time - start_time

# print summary
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

# prepare data for the plots
x = []
y = []
top_best = 15
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])

# 1. plot
plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()

# 2. Plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, marker='.', color='skyblue', alpha=0.6, label='Population Fitness')
plt.plot(best_history, 'r-', linewidth=2, label='Best Fitness')

plt.title('Genetic Algorithm Performance Over Generations')
plt.xlabel('Generation', fontsize=12)
plt.ylabel('Fitness', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()
