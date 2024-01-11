from data import generate_data, format_data, build_prediction_timepoints
from plot import plot_dataset, plot_prediction
from model import gaussian_process
from evolution import evolve
from classical_kernel import combined_kernel_1
from evaluation import fitness, build_fitness_target
import numpy as np




data = generate_data(20, 100, "stein")
plot_dataset(data, "Generated Dataset (Seed: 'stein')", True, "dataset_stein")

training_data, testing_data = format_data(data, 15)
prediction_x = build_prediction_timepoints(0.0, 20.0, 0.1)




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
    0.1,
    20,
    100,
    10,
    0.5,
    0.25,
    True
)

# best_parameters = [0.70334, 0.20593, 0.12266, 0.15243, 0.82494, 0.06506, 0.53513]

model.set_kernel_parameters( classical_gene_reader(best_parameters))

x_train = training_data["time"].to_numpy()
y_train = training_data["value"].to_numpy()
x_test  = testing_data["time"].to_numpy()
y_test  = testing_data["value"].to_numpy()
x_pred = prediction_x
y_pred, sigmas = model.predict(prediction_x)

target_y, target_aucs = build_fitness_target(x_train, y_train, x_pred, 0.1)
fitness(model, 0.1, x_train, x_pred, target_aucs, True, y_train, target_y)

plot_prediction("Model Prediction Performance", x_train, y_train, x_test, y_test, x_pred, y_pred, sigmas, False, [np.min(y_pred), np.max(y_pred)], True, "classical_prediction_performance")
