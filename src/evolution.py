import numpy as np
import pandas as pd
import multiprocessing as mp
import sys
import os
import math

from evaluation import fitness




np.set_printoptions(precision=5)

_logging = False
_model = None
_fitness_granularity = 0.5
_population_size = 100
_genome_length = 10
_parent_survivorship = 0
_mutation_probability = 0.5
_mutation_rate = 1.0
_crossover_probability = 0.5
_current_cycle = 0
_population_split = 50
_parent_indices = None
_parent_probabilities = None
_modified_parent_probabilities = None
_population = None
_population_fitnesses = None
_mutation_function = None
_crossover_function = None
_choice_resampler_function = None




def evolve (
    model,
    gene_reader,
    gene_count,
    fitness_granularity,
    cycles,
    population_size,
    parent_survivorship,
    mutation_probability,
    crossover_probability,
    logging = False
):
    
    best_genes = initialise_genes(gene_count)
    
    if (fitness_granularity > 1.0 or fitness_granularity < 0.0):
        print("\nERROR: fitness_granularity in evolve() for fitness() has to be between 0.0 and 1.0")
        sys.exit(1)
    
    if (math.modf(1.0 / fitness_granularity)[0] != 0):
        print("\nERROR: fitness_granularity in evolve() for fitness() has to divide 1 without remainder")
        sys.exit(1)
    
    if (population_size % 2 != 0):
        print("\nERROR: population_size in evolve() has to be a multiple of 2")
        sys.exit(1)
    
    if (mutation_probability > 1.0 or mutation_probability < 0.0):
        print("\nERROR: mutation_rate in evolve() has to be between 0.0 and 1.0.")
        sys.exit(1)
    
    if (crossover_probability > 1.0 or crossover_probability < 0.0):
        print("\nERROR: crossover_rate in evolve() has to be between 0.0 and 1.0.")
        sys.exit(1)
    
    global _logging
    global _model
    global _gene_reader
    global _fitness_granularity
    global _population_size
    global _genome_length
    global _parent_survivorship
    global _mutation_probability
    global _crossover_probability
    global _current_cycle
    global _population_split
    global _parent_indices
    global _population
    global _population_fitnesses
    global _mutation_function
    global _crossover_function
    global _choice_resampler_function
    
    _logging = logging
    _model = model
    _gene_reader = gene_reader
    _fitness_granularity = fitness_granularity
    _population_size = population_size
    _genome_length = gene_count
    _parent_survivorship = parent_survivorship
    _mutation_probability = mutation_probability
    _crossover_probability = crossover_probability
    _population_split = int(population_size / 2)
    _parent_indices = np.arange(_population_split)
    
    _mutation_function = np.vectorize(mutate, signature = "(n) -> (n)")
    _crossover_function = np.vectorize(crossover, signature = "(n), (n) -> (n)")
    _choice_resampler_function = np.vectorize(resample_choice, [np.float32])
    
    determine_parent_probabilities()
    populate(best_genes)
    
    for cycle in range(cycles):
        
        _current_cycle = cycle
        determine_mutation_rate()
        
        if (_logging):
            print("===== " + str(_current_cycle + 1) + " =====")
            print("\tmutation rate: " + str(_mutation_rate))
        
        select(_population_split)
        reproduce()

        if (_logging):
            print("\n")
            
    best_genes = _population[0]
    best_fitness = _population_fitnesses[0]
    
    if (_logging):
        print("Final Result:")
        print(best_fitness)
        print("-------------")
        print("genes:", best_genes)
        print("parameters:", np.array(_gene_reader(best_genes)))
    
    return best_genes

def initialise_genes (gene_count):
    return np.random.uniform(0.0, 1.0, gene_count)

def populate (gene_seed):
    
    global _population
    
    gene_seeds = np.full((_population_size, _genome_length), gene_seed)
    _population = _mutation_function(gene_seeds)

def reproduce ():
    
    if (_logging):
        print("\tpopulation reproducing...")
    
    global _population
    
    mother_indices = sample_parent()
    father_indices = sample_parent(mother_indices)
    
    child_genomes = _crossover_function(_population[mother_indices], _population[father_indices])
    child_genomes = _mutation_function(child_genomes)
    
    survivors = _population[:_parent_survivorship]
    
    _population = child_genomes
    _population[:_parent_survivorship] = survivors
    
    if (_logging):
        print("\tpopulation has finished reproducing.")

def sample_parent (used_indices = None):
    
    if isinstance(used_indices, type(None)):
        used_indices = np.full(_population_size, -1)
        
    random_indices = np.random.choice(
        _parent_indices,
        _population_size,
        True,
        _parent_probabilities
    )
    
    is_used_flags = np.equal(
        random_indices,
        used_indices
    )
    
    if np.sum(is_used_flags) > 0.0:
        random_indices[is_used_flags] = _choice_resampler_function(used_indices[is_used_flags])
        
    return random_indices

def resample_choice (used_choice):
    
    return np.random.choice(
        _parent_indices,
        p = _modified_parent_probabilities[used_choice]
    )

def determine_parent_probabilities ():
    
    global _parent_probabilities
    global _modified_parent_probabilities
    
    _parent_probabilities = sample_normal_distribution(np.arange(_population_split))
    _parent_probabilities = fill_normal_range(_parent_probabilities)
    
    _modified_parent_probabilities = np.full((_population_split, _population_split), _parent_probabilities)
    
    _modified_parent_probabilities = _modified_parent_probabilities.flatten()
    _modified_parent_probabilities[np.identity(_population_split, np.bool8).flatten()] = 0.0
    _modified_parent_probabilities = np.reshape(_modified_parent_probabilities, (_population_split, _population_split))
    
    _modified_parent_probabilities = np.apply_along_axis(fill_normal_range, 1, _modified_parent_probabilities)

def sample_normal_distribution (x):
        
    mean = 0.0
    standard_deviation = 1.0
    
    exponent = -0.5 * np.square((x - mean) / standard_deviation)
    factor = 1 / (standard_deviation * np.sqrt(2 * np.pi))
    
    return factor * np.exp(exponent)

def fill_normal_range (items):
    
    divider = np.sum(items)
    items /= divider
    
    return items

def mutate (genome):
    
    change_flags = np.greater(
        np.random.rand(_genome_length),
        np.full(_genome_length, _mutation_probability)
    )
    changes = np.random.uniform(-1.0 * _mutation_rate, _mutation_rate, _genome_length)
    
    individual = np.copy(genome)
    
    individual += changes * change_flags
    individual = np.clip(individual, 0.0, 1.0)
    
    return individual

def determine_mutation_rate ():
    
    global _mutation_rate    
    _mutation_rate = np.exp(-1 * _current_cycle * 0.1) * (1.0 / 2.0)

def crossover (mother_genome, father_genome):
        
    child_genome = np.copy(mother_genome)
    
    crossover_flags = np.greater(
        np.random.rand(_genome_length),
        np.full(_genome_length, _crossover_probability)
    )
    
    child_genome[crossover_flags] = father_genome[crossover_flags]
            
    return child_genome
        
def select (selection_size):
    
    if (_logging):
        print("\tselecting...")
    
    global _population
    global _population_fitnesses
    
    individual_fitnesses = pd.DataFrame({"individual": np.arange(_population_size)})
    
    if __name__ == "evolution":
        
        if (_logging):
            print("\t\tspawning subprocesses...")
        
        mp.freeze_support()
        mp.set_start_method("fork", True)
    
        processes = os.cpu_count()
        process_pool = mp.Pool(processes)
        
        batches = math.ceil(len(_population) / processes)
        fitnesses = process_pool.starmap(evaluate_individual, enumerate(_population), batches)
            
        process_pool.close()
        
    individual_fitnesses.insert(1, "fitness", fitnesses)
    individual_fitnesses = individual_fitnesses.sort_values("fitness").head(selection_size)
    
    _population = _population[individual_fitnesses["individual"].to_numpy()]
    _population_fitnesses = individual_fitnesses["fitness"].to_numpy()
    
    if (_logging):
        print("\tfittest individuals:")
        for index in range(_parent_survivorship):
            print("\t\t%.4f" % (_population_fitnesses[index]))
        print("\tselection done.")

def evaluate_individual (individual_index, individual):
    
    _model.set_kernel_parameters( _gene_reader(individual))
    individual_fitness = fitness(_model, _fitness_granularity)
    
    if (_logging):
        print("\t\t\tindividual %d evaluated: %.4f" % (individual_index, individual_fitness))
    
    return individual_fitness
