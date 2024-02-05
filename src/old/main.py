import numpy as np
import matplotlib.pyplot as plt

from pipeline import predicton_pipeline, sampling_pipeline, window_size_exploration_pipeline
from src.kernel import rbf, white_noise, constant, sine_squared
from plot import plot_model_performance, __plot_init, plot_covariance_matrix
from data import generate_data, format_data
from src.gaussian_process import gaussian_process
from src.quantum_circuits import data_reupload_circuit




def classical_kernel (x1, x2, parameters):
    
    # output = white_noise(x1, x2, parameters[:1])
    # output += constant(x1, x2, parameters[1:2]) * sine_squared(x1, x2, parameters[2:5])
    # output += constant(x1, x2, parameters[5:6]) * rbf(x1, x2, parameters[6:8]) 
    # output += constant(x1, x2, parameters[8:9]) * sine_squared(x1, x2, parameters[9:])
    
    output = white_noise(x1, x2, parameters[:1])
    output += rbf(x1, x2, parameters[1:3])
    output += constant(x1, x2, parameters[3:4])
    output += sine_squared(x1, x2, parameters[4:])
    
    return output

def quantum_kernel (x1, x2, circuit):
    
    output = circuit(x1, x2)
    return output




# data = generate_data(12, 50.0, "basf-qcp")
# training_data, testing_data = format_data(data, 10)
# predictions_x = np.arange(0.0, 11.0 + 0.1, 0.1)




# classical_parameters = np.array([5.79825, 4.25336])
classical_parameters = np.array([0.00001, 1.64014, 2.10621, 50.0, 1.33718, 0.143265, 12.0])

# quantum_parameters = np.array([2.95249, 0.16572, 3.11833, 0.17274, 3.14159, 2.99228, 1.26662, 3.14159, 1.56207, 3.14159, 3.14159, 1.59143, 0.0, 0.3111, 2.6618, 3.14159, 3.14159, 0.50479, 0.0, 2.28252, 2.22157, 2.87927, 2.61325, 2.27364, 2.04065, 2.19619, 0.0, 2.8116, 3.14159, 0.52513, 1.53748, 1.27702, 2.6721, 3.14159, 0.73814, 2.66474, 1.95588, 0.39848, 2.25216, 0.43072, 1.93396, 0.49217, 1.42906, 0.71534, 0.0, 0.67342, 0.4056, 2.89051])
quantum_parameters = np.array([3.14159, 0.62926, 3.08582, 0.0, 1.86067, 2.99228, 1.26662, 3.1272, 2.27653, 2.85918, 3.14159, 0.61827, 0.0, 1.23778, 3.14159, 3.14159, 3.14159, 1.79345, 1.22627, 2.59503, 2.0462, 2.67786, 2.69798, 3.14159, 3.14159, 1.83085, 0.95336, 2.682, 2.34535, 1.75072, 2.072, 0.50744, 2.49427, 3.14159, 0.0, 2.95744, 1.47, 2.67065, 3.10201, 2.31403, 2.44717, 0.94448, 1.98761, 1.95807, 0.95813, 0.70665, 2.65986, 2.93345])




# model = gaussian_process(training_data, quantum_kernel, kernel_parameters = classical_parameters)
# predictions, sigmas = model.predict(predictions_x)



# plot_model_performance(training_data, testing_data, predictions_x, predictions, sigmas, "Classical Gaussian Process Predictions", False, True)
# sampling_pipeline(training_data, model, 20, [0.0, 80.0], True)
# window_size_exploration_pipeline(classical_kernel, classical_parameters, 1000, 50, 5, 50, True)




# model, parameters, model_fitness, dataset, training_data, testing_data, predictions, sigma = predicton_pipeline(
#     kernel_function = quantum_kernel,
#     is_quantum = True,
#     initial_parameters = quantum_parameters,
#     parameter_minimum = 0.0,
#     parameter_maximum = np.pi,
#     kernel_parameter_count = 12,
#     quantum_kernel_layer_count = 4,
#     quantum_circuit_type = "reupload",
#     quantum_circuit_subtype = "11",
#     run_evolution = False,
#     evolution_generations = 20,
#     evolution_population_size = 100,
#     evolution_parent_survivorship = 5,
#     evolution_mutation_rate = 0.5,
#     evolution_crossover_rate = 0.25,
#     fitness_granularity = 0.25,
#     file_path = "", 
#     day_count = 12, 
#     initial_value = 50.0, 
#     training_window = 10, 
#     dataset_seed = "basf-qcp",
#     prediction_granularity = 0.1,
#     log_runtime = False,
#     log_evolution = False,
#     log_fitness = False,
#     plot_performance = False,
#     plot_limits = [32.5, 57.5]
# )




quantum_circuit_function = lambda parameters, angle_scaling_minimum, angle_scaling_maximum: data_reupload_circuit(parameters, "11", angle_scaling_minimum, angle_scaling_maximum)

data = generate_data(12, 50.0, "basf-qcp")
training_data, testing_data = format_data(data, 10)

# colours = __plot_init([14, 6], True)

# fig, ax = plt.subplots(1, 1)

# fig.text(0.525, 0.02, 'Time', ha='center')
# fig.text(0.025, 0.5, 'Value', va='center', rotation='vertical')
# plt.subplots_adjust(left = 0.1, right = 0.95, top = 0.9, bottom = 0.15)

# plt.suptitle("Classical Kernel vs Quantum Kernel    >>    Seed 'basf-qcp'")
# plt.ylim([42.5, 56.0])

# dataset_seeds = ["basf-qcp", "basf", "qcp", "team-a", "lmu"]
# dataset_starts = [50.0, 47.0, 53.0, 48.5, 51.5]
# colour_indices = [-1, 1, 2, 3, -2]
predictions_x = np.arange(0.0, 11.0 + 0.1, 0.1)

kmodel = gaussian_process(training_data, classical_kernel, kernel_parameters = classical_parameters)
qmodel = gaussian_process(training_data, quantum_kernel, quantum_circuit_function, 0.0, 11.0, quantum_parameters)

kmodel_normalised_covmat = (kmodel.covariance_matrix - np.min(kmodel.covariance_matrix)) / (np.max(kmodel.covariance_matrix) - np.min(kmodel.covariance_matrix))

# plot_covariance_matrix(kmodel.covariance_matrix, "Classical Covariance Matrix", True)
# plot_covariance_matrix(kmodel_normalised_covmat, "Classical Covariance Matrix Normalised", True)
# plot_covariance_matrix(qmodel.covariance_matrix, "Quantum Covariance Matrix", True)
    
# x_train = training_data["time"].to_numpy()
# y_train = training_data["value"].to_numpy()
# x_test  = testing_data["time"].to_numpy()
# y_test  = testing_data["value"].to_numpy()
# x_pred = predictions_x
# y_pred_k, sigmas_k = kmodel.predict(predictions_x)
# y_pred_q, sigmas_q = qmodel.predict(predictions_x)
    
# ax.plot(x_pred, y_pred_k, color = colours[2], label = "classic predictions")
# ax.fill_between(
#     x_pred,
#     np.subtract(y_pred_k, sigmas_k),
#     np.add(y_pred_k, sigmas_k),
#     color = colours[2],
#     alpha = 0.33
# )

# ax.plot(x_pred, y_pred_q, color = colours[3], label = "quantum predictions")
# ax.fill_between(
#     x_pred,
#     np.subtract(y_pred_q, sigmas_q),
#     np.add(y_pred_q, sigmas_q),
#     color = colours[3],
#     alpha = 0.33
# )

# ax.plot(x_train, y_train, 'o', color = colours[-1], label = "training data")
# ax.plot(x_test, y_test, 'o', color = colours[1], label = "testing data")

# legend = plt.legend(loc = "upper left", frameon = True)
# frame = legend.get_frame()
# frame.set_facecolor("#444444")

# fig.savefig('test.png', transparent = True)
