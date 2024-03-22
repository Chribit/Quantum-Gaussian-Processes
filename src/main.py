from data import generate_data, format_data, build_prediction_timepoints
from plot import plot_dataset, plot_prediction, plot_circuit, plot_fitness, plot_covariance_matrix, plot_aapl
from model import gaussian_process
from evolution import evolve
from kernel import classical_kernel_1, quantum_kernel_1
from evaluation import fitness, build_fitness_target
import numpy as np
import sys




days = 365
# seed = "372rzdnso2"
# data = generate_data(days, 69, seed)
# plot_dataset(data, "Generated Dataset (Seed: '" + seed + "')", True, "dataset_" + seed)

seeds = [
    "adu38d1ed0k21kand",
    "7878uhe8450jffw3t5g5ggsvns3",
    "vi5u30ß021eijdaw3",
    "ß0ojn5btg7rfw8du3jhfnrwa",
    "90ßpüö.,l3mkrnjfugeeff",
    "7589rgjivmp3re2d3f49t+raük2jhrqf",
    "567gt8fhivnwvs0rikpslrt4g3ßa22",
    "47tg89t0bzoidjvsnjabzegu2d",
    "58z9gujsnfiyw",
    "vn96e5t40wr3aijis"
]

for index, seed in enumerate(seeds):
    data = generate_data(days, 69, seed)
    plot_dataset(data, "Generated Dataset (Seed: '" + seed + "')", True, "dataset_" + str(index))

# data = generate_data(days, 69, seeds[9])
# plot_dataset(data, "Generated Dataset (Seed: '" + seeds[9] + "')", True, "dataset_" + str(9))
plot_aapl()

# training_window = 15
# training_data, testing_data = format_data(data, training_window)
# prediction_x = build_prediction_timepoints(0.0, float(days), 0.1)





def classical_gene_reader (genes):
    
    model_parameters = genes * np.array([1.0, 10.0, 10.0, 100.0, 10.0, 10.0, 100.0])
    model_parameters = np.clip(model_parameters, 0.00001, 100.0)
    
    return model_parameters

def quantum_gene_reader (genes):
    
    model_parameters = genes * np.pi
    
    return model_parameters




# quantum_gene_count = 1 * ((4 ** 4) - 1)
# quantum_parameters = np.zeros(quantum_gene_count)
# # quantum_parameters = np.random.uniform(0.0, np.pi, quantum_gene_count)

# model = gaussian_process(
#     training_data,
#     quantum_kernel_1,
#     quantum_parameters,
#     True
# )

# plot_circuit("Reupload Circuit (Inversion Test)", model, True, "reupload_circuit_inversion_test_2_layer_zeros")

# best_parameters = evolve(
#     model,
#     quantum_gene_reader,
#     quantum_gene_count,
#     0.5,
#     10,
#     24,
#     0.5,
#     0.25,
#     True
# )




# model = gaussian_process(
#     training_data,
#     classical_kernel_1
# )

# best_parameters = evolve(
#     model,
#     classical_gene_reader,
#     7,
#     0.1,
#     10,
#     100,
#     0.5,
#     0.25,
#     True
# )

# for row in range(15):
#     for column in range(15):
        
#         print(row, column, (1.0 / 15.0) * (15.0 - abs((row + 1.0) - (column + 1.0))))




# model.set_kernel_parameters( quantum_gene_reader(best_parameters))
# model.set_kernel_parameters( classical_gene_reader(best_parameters))
# model.set_kernel_parameters(quantum_parameters)
# model.set_kernel_parameters(np.zeros(7))

# x_train = training_data["time"].to_numpy()
# y_train = training_data["value"].to_numpy()
# x_test  = testing_data["time"].to_numpy()
# y_test  = testing_data["value"].to_numpy()
# x_pred = prediction_x
# y_pred, sigmas = model.predict(prediction_x)

# window_prediction_x = build_prediction_timepoints(0.0, float(training_window), 0.1)
# target_y, target_aucs = build_fitness_target(x_train, y_train, window_prediction_x, 0.1)
# fitness = fitness(model, 0.1, x_train, window_prediction_x, target_aucs, True, y_train, target_y)

# print(fitness)

# plot_prediction("Model Prediction Performance", x_train, y_train, x_test, y_test, x_pred, y_pred, sigmas, False, [np.min(data["value"].to_numpy()), np.max(data["value"].to_numpy())], True, "performance")
# plot_covariance_matrix(model.covariance_matrix, "Classical Model Covariance Matrix", True, "matrix")
