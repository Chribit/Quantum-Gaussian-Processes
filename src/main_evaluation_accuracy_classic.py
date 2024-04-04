from data import generate_data, format_data, generate_seed, build_prediction_timepoints
from model import gaussian_process
from evolution import evolve
from kernel import classical_kernel_1
import numpy as np
import pandas as pd




np.set_printoptions(linewidth = np.inf, precision = 2, suppress = True)




def classical_gene_reader (genes):
    
    model_parameters = genes * np.array([1.0, 10.0, 10.0, 100.0, 10.0, 10.0, 100.0])
    model_parameters = np.clip(model_parameters, 0.00001, 100.0)
    
    return model_parameters




days = 25
training_window = 15
samples = 1000
max_cycles = 15
prediction_granularity = 2
evaluated_distances = [0 for i in range(days - training_window)]
evaluated_trends = [0 for i in range(days - training_window)]




sample = 0
while sample < samples:
    
    print("---", sample, "---")
    
    seed = generate_seed(10)
    data = generate_data(days, 100, seed)
    training_data, testing_data = format_data(data, training_window)
    
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
        16,
        0.5,
        0.5,
        False
    )
    
    if (cycles + 1) == max_cycles:
        continue
    
    model.set_kernel_parameters( classical_gene_reader(best_parameters))
    prediction_x = build_prediction_timepoints(training_window, float(days), 1)
    predictions = np.array(model.predict(prediction_x)[0])
    target_values = testing_data["value"].to_numpy()
    
    distances = np.abs(np.subtract(target_values, predictions))
    print(distances)
    
    previous_values = data["value"].to_numpy()[training_window - 1 : -1]
    target_trends = np.sign(np.subtract(target_values, previous_values))
    predicted_trends = np.sign(np.subtract(predictions, previous_values))
    
    trends = np.divide(np.abs(np.add(target_trends, predicted_trends)), 2)
    print(trends)
    
    evaluated_distances = np.add(evaluated_distances, distances)
    evaluated_trends = np.add(evaluated_trends, trends)
    
    sample += 1
    
    


evaluated_distances = np.divide(evaluated_distances, samples)
evaluated_trends = np.divide(evaluated_trends, samples)
print("\n===\n")
print(evaluated_distances)
print(evaluated_trends)
evaluation_data = np.column_stack((evaluated_distances, evaluated_trends))
evaluation_data = pd.DataFrame(evaluation_data, columns = ["distance", "trend"])
evaluation_data.to_csv("data/accuracy_classic.csv", index = False)
