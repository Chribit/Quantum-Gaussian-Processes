from data import generate_data, format_data, generate_seed
from model import gaussian_process
from evolution import evolve
from kernel import classical_kernel_1
import numpy as np
import pandas as pd




def classical_gene_reader (genes):
    
    model_parameters = genes * np.array([1.0, 10.0, 10.0, 100.0, 10.0, 10.0, 100.0])
    model_parameters = np.clip(model_parameters, 0.00001, 100.0)
    
    return model_parameters




minimum_window = 5
maximum_window = 15
samples = 1000
max_cycles = 15
prediction_granularity = 2
evaluation_data = []




for days in range(minimum_window, maximum_window):
    
    print("====", days, " days ====")
    
    iteration = 0
    
    while iteration < samples:
        
        seed = generate_seed(10)
        data = generate_data(days, 200, seed)
        training_data, testing_data = format_data(data, days)
        model = gaussian_process(
            training_data,
            classical_kernel_1
        )
        best_parameters, cycles = evolve(
            model,
            classical_gene_reader,
            7,
            prediction_granularity,
            0.9,
            max_cycles,
            10,
            0.5,
            0.5,
            False
        )
        cycles += 1
        
        print("\titeration", iteration + 1, ":", cycles, "cycles")
        
        if (cycles < max_cycles):
            evaluation_data.append([seed, days, cycles])
            iteration += 1
            
            


evaluation_data = pd.DataFrame(evaluation_data, columns = ["seed", "days", "cycles"])
evaluation_data.to_csv("data/conversions_classic.csv", index = False)
