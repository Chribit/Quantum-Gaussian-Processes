from data import generate_data, format_data, build_prediction_timepoints
from plot import plot_dataset, plot_prediction
from model import gaussian_process
from evolution import evolve
from classical_kernel import combined_kernel_1
import numpy as np




data = generate_data(365, 100, "stein")
plot_dataset(data, "Generated Dataset (Seed: 'stein')", True, "dataset_stein")

training_data, testing_data = format_data(data, 300)
prediction_x = build_prediction_timepoints(0.0, 365.0, 0.1)

classical_parameters = np.array([0.00001, 1.64014, 2.10621, 50.0, 1.33718, 0.143265, 12.0])

model = gaussian_process(
    training_data,
    combined_kernel_1,
    kernel_parameters = classical_parameters
)

x_train = training_data["time"].to_numpy()
y_train = training_data["value"].to_numpy()
x_test  = testing_data["time"].to_numpy()
y_test  = testing_data["value"].to_numpy()
x_pred = prediction_x
y_pred, sigmas = model.predict(prediction_x)

plot_prediction("Model Prediction Performance", x_train, y_train, x_test, y_test, x_pred, y_pred, sigmas, [290, 310], False, True, "classical_prediction_performance")
