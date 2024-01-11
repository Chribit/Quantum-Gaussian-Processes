from data import generate_data, format_data, build_prediction_timepoints
from plot import plot_dataset, plot_prediction
from model import gaussian_process
from evolution import evolve
from classical_kernel import combined_kernel_1
from evaluation import fitness, build_fitness_target
import numpy as np
import sys



days = 20
seed = "lmu"
data = generate_data(days, 69, "lmu")
plot_dataset(data, "Generated Dataset (Seed: '" + seed + "')", True, "dataset_" + seed)

training_window = 15
training_data, testing_data = format_data(data, training_window)
prediction_x = build_prediction_timepoints(0.0, float(days), 0.1)




def classical_gene_reader (genes):
    
    model_parameters = genes * np.array([1.0, 10.0, 10.0, 100.0, 10.0, 10.0, 100.0])
    model_parameters = np.clip(model_parameters, 0.00001, 100.0)
    
    return model_parameters




model = gaussian_process(
    training_data,
    combined_kernel_1
)

best_parameters = evolve(
    model,
    classical_gene_reader,
    7,
    0.25,
    5,
    90,
    0.5,
    0.25,
    True
)

model.set_kernel_parameters( classical_gene_reader(best_parameters))

x_train = training_data["time"].to_numpy()
y_train = training_data["value"].to_numpy()
x_test  = testing_data["time"].to_numpy()
y_test  = testing_data["value"].to_numpy()
x_pred = prediction_x
y_pred, sigmas = model.predict(prediction_x)

window_prediction_x = build_prediction_timepoints(0.0, float(training_window), 0.1)
target_y, target_aucs = build_fitness_target(x_train, y_train, window_prediction_x, 0.1)
fitness(model, 0.1, x_train, window_prediction_x, target_aucs, True, y_train, target_y)

plot_prediction("Model Prediction Performance", x_train, y_train, x_test, y_test, x_pred, y_pred, sigmas, False, [np.min(data["value"].to_numpy()), np.max(data["value"].to_numpy())], True, "classical_prediction_performance")
