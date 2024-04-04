from data import generate_data, format_data, build_prediction_timepoints
from plot import plot_dataset, plot_prediction, plot_covariance_matrix
from model import gaussian_process
from evolution import evolve
from kernel import classical_kernel_1
from evaluation import fitness, build_fitness_target_AUC
import numpy as np




days = 20
seed = "vn96e5t40wr3aijis"

data = generate_data(days, 69, seed)
plot_dataset(data, "Generated Dataset (Seed: '" + seed + "')", True, "evolution/classic/dataset")

training_window = 15
prediction_granularity = 2

training_data, testing_data = format_data(data, training_window)
prediction_x = build_prediction_timepoints(0.0, float(days), prediction_granularity)





def classical_gene_reader (genes):
    
    model_parameters = genes * np.array([1.0, 10.0, 10.0, 100.0, 10.0, 10.0, 100.0])
    model_parameters = np.clip(model_parameters, 0.00001, 100.0)
    
    return model_parameters




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
    1,
    4,
    0.75,
    0.5,
    False,
    True,
    "evolution/classic/timeline"
)
# best_parameters = np.random.uniform(0.0, 1.0, 7)




model.set_kernel_parameters( classical_gene_reader(best_parameters))

x_train = training_data["time"].to_numpy()
y_train = training_data["value"].to_numpy()
x_test  = testing_data["time"].to_numpy()
y_test  = testing_data["value"].to_numpy()
x_pred = prediction_x
y_pred, sigmas = model.predict(prediction_x)

window_prediction_x = build_prediction_timepoints(0.0, float(training_window), prediction_granularity)
target_y, target_aucs = build_fitness_target_AUC(x_train, y_train, window_prediction_x, prediction_granularity)
auc_fitness = fitness(model, prediction_granularity, x_train, window_prediction_x, target_aucs, None, True, "evolution/classic/fitness", y_train, target_y)
print("\nFitness (AUC):", auc_fitness)

plot_prediction("Model Prediction Performance", x_train, y_train, x_test, y_test, x_pred, y_pred, sigmas, False, [np.min(data["value"].to_numpy()) - 2.0, np.max(data["value"].to_numpy()) + 2.0], True, "evolution/classic/performance")
plot_covariance_matrix(model.covariance_matrix, "Classical Model Covariance Matrix", True, "evolution/classic/matrix")
