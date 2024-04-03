from data import generate_data, format_data, generate_seed
from model import gaussian_process
from evolution import evolve
from kernel import quantum_kernel_1
import numpy as np
import pandas as pd




def quantum_gene_reader (genes):
    
    model_parameters = genes * np.pi
    return model_parameters




minimum_window = 5
maximum_window = 15
samples = 1000
max_cycles = 15
prediction_granularity = 2
evaluation_data = []




qubit_count = 2
layer_count = 2
quantum_gene_count = layer_count * ((4 ** qubit_count) - 1)




for days in range(minimum_window, maximum_window + 1):
    
    iteration = 0
    
    while iteration < samples:
        
        seed = generate_seed(10)
        data = generate_data(days, 200, seed)
        training_data, testing_data = format_data(data, days)
        model = gaussian_process(
            training_data,
            quantum_kernel_1,
            np.random.uniform(0.0, np.pi, quantum_gene_count),
            True,
            days,
            qubit_count
        )
        best_parameters, cycles = evolve(
            model,
            quantum_gene_reader,
            quantum_gene_count,
            prediction_granularity,
            0.9,
            max_cycles,
            8,
            0.5,
            0.5,
            False
        )
        cycles += 1
        
        print(days, "days - iteration", iteration + 1, ":", cycles, "cycles")
        
        if (cycles < max_cycles):
            evaluation_data.append([seed, days, cycles])
            iteration += 1
            
            


evaluation_data = pd.DataFrame(evaluation_data, columns = ["seed", "days", "cycles"])
evaluation_data.to_csv("data/conversions_quantum.csv", index = False)
