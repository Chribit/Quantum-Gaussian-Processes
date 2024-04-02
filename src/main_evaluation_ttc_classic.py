from data import generate_data, format_data, generate_seed
from plot import plot_dataset
from model import gaussian_process
from evolution import evolve
from kernel import classical_kernel_1
import numpy as np




def classical_gene_reader (genes):
    
    model_parameters = genes * np.array([1.0, 10.0, 10.0, 100.0, 10.0, 10.0, 100.0])
    model_parameters = np.clip(model_parameters, 0.00001, 100.0)
    
    return model_parameters




prediction_granularity = 2




for days in range(5, 15):
    for iteration in range(1000):
        
        print("====", days, "days - iteration", iteration, "====")
        
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
            0.99,
            15,
            10,
            0.5,
            0.5,
            False
        )
        
        print("-->", cycles)