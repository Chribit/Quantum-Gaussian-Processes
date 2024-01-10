import pandas as pd
import numpy as np
import time
import sys
import string
import random as rnd

from data import generate_data, format_data
from src.gaussian_process import gaussian_process
from plot import plot_model_performance, plot_model_samples_dual, plot_window_comparison
from src.quantum_circuits import swap_test_circuit, inversion_test_circuit, data_reupload_circuit
from evolution import evolve
from evaluation import fitness




def window_size_exploration_pipeline (
        model_kernel,
        model_parameters,
        dataset_count = 100,
        dataset_value = 100,
        window_minimum = 5,
        window_maximum = 50,
        logging = False
    ):
    
    results = pd.DataFrame({
        "window_size": np.arange(window_minimum, window_maximum + 1, dtype = np.uint8),
        "average_difference": np.full(window_maximum - window_minimum + 1, 0.0, dtype = np.float32)
    })
    
    seed_letters = string.ascii_lowercase

    for dataset_index in range(dataset_count):
        
        dataset_seed = "".join(rnd.choice(seed_letters) for o in range(10))
            
        dataset = generate_data(window_maximum + 1, dataset_value, dataset_seed)
        
        target_value = dataset.at[window_maximum, "value"]
        
        if logging:
            print("Processing dataset %d with seed '%s'..." % (dataset_index + 1, dataset_seed))
            print("--> Target value at %.2f..." % (target_value))
        
        index = 0
        
        for window_size in range(window_minimum, window_maximum + 1):
            
            if logging:
                print("\tCurrent window size: %d" % (window_size))
            
            dataset_slice = dataset.tail(window_size + 1)
            dataset_slice = dataset_slice.drop(window_maximum)
            
            model = gaussian_process(dataset_slice, model_kernel, kernel_parameters = model_parameters)
            prediction = model.predict([window_maximum])[0][0]
            
            difference = abs(target_value - prediction)
            
            results.loc[index, "average_difference"] += difference
            
            index += 1
            
    results["average_difference"] /=  dataset_count
    
    plot_window_comparison(results, "Correlation of window size to prediction accuracy over %d distinct datasets" % (dataset_count), True)
    
def sampling_pipeline (
        data,
        model,
        samples = 5,
        plot_limits = False,
        plot_save = False
    ):
    
    training_x, training_y = model.get_training_data()
    prior_mean = np.average(training_y)
    
    prior_samples = model.sample_prior(samples, prior_mean)
    posterior_samples = model.sample_posterior(samples)
    
    plot_model_samples_dual(data, prior_samples, posterior_samples, "", plot_limits, plot_save)
    
def predicton_pipeline (
        kernel_function,
        is_quantum = False,
        initial_parameters = None,
        parameter_minimum = 0.000001,
        parameter_maximum = 1000.0,
        kernel_parameter_count = 1,
        quantum_kernel_layer_count = 1,
        quantum_circuit_type = "reupload",
        quantum_circuit_subtype = "11",
        run_evolution = True,
        evolution_generations = 10,
        evolution_population_size = 100,
        evolution_parent_survivorship = 10,
        evolution_mutation_rate = 0.5,
        evolution_crossover_rate = 0.5,
        fitness_granularity = 0.5,
        file_path = "", 
        day_count = 30, 
        initial_value = 100.0, 
        training_window = 10, 
        dataset_seed = "qcp",
        prediction_granularity = 0.25,
        log_runtime = False,
        log_evolution = False,
        log_fitness = False,
        plot_performance = False,
        plot_limits = False
    ):
    
    if quantum_circuit_type != "swap" and quantum_circuit_type != "inversion" and quantum_circuit_type != "reupload":
        print("\nERROR: No valid quantum circuit type provided. Valid types are 'swap', 'inversion' and 'reupload'.")
        sys.exit(1)
    
    before = time.time()
    
    if not is_quantum and quantum_kernel_layer_count != 1:
        quantum_kernel_layer_count = 1
    
    data = None
    if file_path != "":
        data = pd.DataFrame()
        # TODO: open file
    else:
        data = generate_data(day_count, initial_value, str(dataset_seed))
    
    training_data, testing_data = format_data(data, training_window)
    
    prediction_time_points = np.arange(0.0, float(day_count - 1) + prediction_granularity, prediction_granularity)
        
    if isinstance(initial_parameters, type(None)):
        initial_parameters = np.random.uniform(parameter_minimum, parameter_maximum, kernel_parameter_count * quantum_kernel_layer_count)
    else:
        if kernel_parameter_count * quantum_kernel_layer_count != len(initial_parameters):
            print("\nERROR: The provided initial parameter array does not have the correct amount of entries. %d expected, but %d recieved." % (kernel_parameter_count * quantum_kernel_layer_count, len(initial_parameters)))
            sys.exit(1)
        
    quantum_circuit_function = None
    if is_quantum:
        if quantum_circuit_type == "swap":
            quantum_circuit_function = lambda parameters, angle_scaling_minimum, angle_scaling_maximum: swap_test_circuit(parameters, quantum_circuit_subtype, angle_scaling_minimum, angle_scaling_maximum)
        elif quantum_circuit_type == "inversion":
            quantum_circuit_function = lambda parameters, angle_scaling_minimum, angle_scaling_maximum: inversion_test_circuit(parameters, quantum_circuit_subtype, angle_scaling_minimum, angle_scaling_maximum)
        elif quantum_circuit_type == "reupload":
            quantum_circuit_function = lambda parameters, angle_scaling_minimum, angle_scaling_maximum: data_reupload_circuit(parameters, quantum_circuit_subtype, angle_scaling_minimum, angle_scaling_maximum)
            
    model = gaussian_process(training_data, kernel_function, quantum_circuit_function, 0.0, float(day_count), initial_parameters)
    
    parameters = initial_parameters
    if run_evolution:
        parameters = evolve(
                parameters,
                model,
                fitness_granularity,
                evolution_generations,
                evolution_population_size,
                evolution_parent_survivorship,
                evolution_mutation_rate,
                evolution_crossover_rate,
                parameter_minimum,
                parameter_maximum,
                True,
                log_evolution
            )
        
        model.set_kernel_parameters(parameters)
    
    predictions, sigmas = model.predict(prediction_time_points)
    
    model_fitness = fitness(model, fitness_granularity)
    if log_fitness:
        print("\nFinal Model Fitness: %.5f\n" % model_fitness)
    
    after = time.time()
    if log_runtime:
        print(str(after - before) + " s")
        
    if plot_performance:
        
        type_string = "Classical"
        if is_quantum:
            type_string = "Quantum"
        
        specification_string = ""
        if is_quantum:
            
            quantum_circuit_type_string = ""
            if quantum_circuit_type == "swap":
                quantum_circuit_type_string = "Swap Test"
            elif quantum_circuit_type == "inversion":
                quantum_circuit_type_string = "Inversion Test"
            elif quantum_circuit_type == "reupload":
                quantum_circuit_type_string = "Data Reupload"
            
            specification_string = "(%s, %d Layer, Circuit %s)" % (quantum_circuit_type_string, quantum_kernel_layer_count, quantum_circuit_subtype)
        
        plot_model_performance(
            training_data,
            testing_data,
            prediction_time_points,
            predictions, 
            sigmas,
            "Gaussian Process Performance\n%s Kernel %s\nDataset Seed: '%s'" % (type_string, specification_string, dataset_seed),
            plot_limits
        )
        
    return (model, parameters, model_fitness, data, training_data, testing_data, predictions, sigmas)
