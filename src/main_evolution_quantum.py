from data import generate_data, format_data, build_prediction_timepoints
from plot import plot_dataset, plot_prediction, plot_circuit, plot_fitness, plot_covariance_matrix
from model import gaussian_process
from evolution import evolve
from kernel import quantum_kernel_1
from evaluation import fitness, build_fitness_target_AUC, build_fitness_target_SMD
import numpy as np




days = 20
seed = "58z9gujsnfiyw"

data = generate_data(days, 69, seed)
plot_dataset(data, "Generated Dataset (Seed: '" + seed + "')", True, "evolution/dataset")

training_window = 15
prediction_granularity = 0.1

training_data, testing_data = format_data(data, training_window)
prediction_x = build_prediction_timepoints(0.0, float(days), prediction_granularity)




def quantum_gene_reader (genes):
    
    model_parameters = genes * np.pi
    
    return model_parameters




quantum_gene_count = 2 * ((4 ** 4) - 1)
quantum_parameters = np.zeros(quantum_gene_count)
quantum_parameters = np.random.uniform(0.0, np.pi, quantum_gene_count)

model = gaussian_process(
    training_data,
    quantum_kernel_1,
    quantum_parameters,
    True
)

plot_circuit("Reupload Circuit (Inversion Test)", model, True, "evolution/reupload_circuit_inversion_test_2_layer_zeros")

# best_parameters = evolve(
#     model,
#     quantum_gene_reader,
#     quantum_gene_count,
#     0.5,
#     0.98,
#     10,
#     24,
#     0.5,
#     0.25,
#     True
# )
best_parameters = quantum_parameters



model.set_kernel_parameters( quantum_gene_reader(best_parameters))

x_train = training_data["time"].to_numpy()
y_train = training_data["value"].to_numpy()
x_test  = testing_data["time"].to_numpy()
y_test  = testing_data["value"].to_numpy()
x_pred = prediction_x
y_pred, sigmas = model.predict(prediction_x)

window_prediction_x = build_prediction_timepoints(0.0, float(training_window), 0.1)
target_y, target_aucs = build_fitness_target_AUC(x_train, y_train, window_prediction_x, 0.1)
target_slopes = build_fitness_target_SMD(y_train)

auc_fitness = fitness(model, 0.1, x_train, window_prediction_x, target_aucs, None, True, "evolution/fitness", y_train, target_y)
print("\nFitness (AUC):", auc_fitness)

plot_prediction("Model Prediction Performance", x_train, y_train, x_test, y_test, x_pred, y_pred, sigmas, False, [np.min(data["value"].to_numpy()) - 2.0, np.max(data["value"].to_numpy()) + 2.0], True, "evolution/performance")
plot_covariance_matrix(model.covariance_matrix, "Classical Model Covariance Matrix", True, "evolution/matrix")
