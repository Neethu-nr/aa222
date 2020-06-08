# Reference: Genetic Algorithms with Python
# Author: Clinton Sheppard 

import random
import numpy as np
from biped_walker_modified import Fitness
from crossover import CrossoverMethod
from mutation import MutationMethod
from selection import SelectionMethod

class Chromosome:
    def __init__(self, genes, fitness):
        self.Genes = genes
        self.Fitness = fitness

def cauchy (size, range):
    sample = np.random.standard_cauchy(size)
    sample_in_range =  (sample/25+range[0])*(range[1]-range[0])/2
    return np.clip(sample_in_range, range[0], range[1])

def starting_population(population_size, get_fitness, gene_range, gene_length):
    # creates random population list(Chromosome) with fitness    

    ## From normal distribution 
    # genes = [[np.random.randint(*gene_range[i]) for i in range(gene_length)] for _ in range(population_size)]

    ## From cauchy's distribution
    genes = [0]*gene_length
    for i in range(gene_length):
        np.random.seed(i+5)
        genes[i] = cauchy(population_size, gene_range[i])
    # import pdb; pdb.set_trace()

    genes = list(map(list, zip(*genes)))

    fitness_vals = get_fitness(genes)
    population = [Chromosome(genes[i], fitness_vals[i]) for i in range(population_size)]
    return population, fitness_vals

def get_best(population):
    best = population[0]
    for element in population[1:]:
        if element.Fitness > best.Fitness:
            best = element
    return best



def genetic_algorithm(population, functions, constants):
    population,fitness_vals = population 
    fitness, select, crossover, mutate = functions
    MAX_ITER, POPULATION_SIZE, SAMPLE_SIZE = constants

    def stats(verbose = True, plot =False):
        # stores statistics about the run
        # returns True if improvement in this iteration
        return_val = False
        if plot:
            # import pdb;pdb.set_trace()
            import matplotlib.pyplot as plt
            plt.plot(stats.best_current_hist,label='best per population')
            plt.plot(stats.best_global_hist,label='best so far')
            plt.plot(stats.avg,label='avg')
            plt.plot(stats.worst,label='min')
            plt.legend()
            plt.show()
            return return_val

        max_fitness = max(fitness_vals)
        if max_fitness > stats.best_fitness:
            return_val = True
            stats.best_fitness = max_fitness
            stats.best_choromosome = get_best(population)
        stats.best_global_hist.append(stats.best_fitness)
        stats.best_current_hist.append(max_fitness)
        stats.avg.append(np.average(fitness_vals))
        stats.worst.append(min(fitness_vals))

        if verbose:
            print("best score this turn:", max_fitness) 
            print("best score so far:", stats.best_fitness) 

        return return_val

    stats.best_fitness = -np.float("inf")
    stats.best_global_hist = []
    stats.best_current_hist = []
    stats.avg = []
    stats.worst = []
    best = population[0]
    stats.best_choromosome = None

    for k in range(MAX_ITER):

        print("#",k)
        if k%10 == 0 and k>0:
            stats(True,True)

        # Record statistics and insert the global best walker
        # if the last population didn't improve
        if not stats():
            # import pdb; pdb.set_trace()
            population.append(stats.best_choromosome)
            fitness_vals.append(stats.best_choromosome.Fitness)

        parents = select(population)            # list(Chromosomes)

        # sample parents according to probability proportional to fitness
        # import pdb;pdb.set_trace()
        min_val =min(fitness_vals)
        prob = [parents[i].Fitness - min_val for i in range(SAMPLE_SIZE) ]
        # prob = np.clip(prob,0,None)
        prob /= sum(prob)
        parent_comb = np.random.choice(a=np.arange(0,SAMPLE_SIZE), size=(POPULATION_SIZE,2), p=prob)
        children = [crossover(parents[parent_comb[i,0]].Genes, parents[parent_comb[i,1]].Genes) for i in range(POPULATION_SIZE)]

        # prev_children = list(children)
        for i in range(POPULATION_SIZE): 
            children[i] = mutate(children[i])        # list(Genes)
        # print("mutations ",sum([1 if prev_children[i][j] == children[i][j] else 0 for i in range(len(children)) for j in range(len(children[0]))]))

        # list(Chromosomes)
        fitness_vals = fitness(children)
        population = [Chromosome(children[i], fitness_vals[i]) for i in range(POPULATION_SIZE)]

    # plot statistics
    stats(True, True)
    return stats.best_choromosome 

if __name__ == "__main__":
    gene_range = [(10,100), (1,10), (1,10), (1,50), (1,50)]
    POPULATION_SIZE = 20
    GENE_LENGTH = len(gene_range)
    SAMPLE_SIZE = 15 # >= POPULATION_SIZE/2
    MAX_ITER = 50
    fitness = Fitness().fitness
    select = SelectionMethod(SAMPLE_SIZE).truncationSelection
    crossover = CrossoverMethod(GENE_LENGTH).interpolation
    mutate =  MutationMethod(GENE_LENGTH).GaussianMutation
    population, fitness_vals = starting_population(POPULATION_SIZE, fitness, gene_range, GENE_LENGTH)
    best = genetic_algorithm((population,fitness_vals), (fitness, select, crossover, mutate), (MAX_ITER, POPULATION_SIZE, SAMPLE_SIZE))
    print("best gene:",best.Genes)
    import pdb; pdb.set_trace()
    fitness([best.Genes], display=True)

