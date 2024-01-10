from data import generate_data, format_data, build_prediction_timepoints
from plot import plot_dataset




data = generate_data(20, 100, "stein")
plot_dataset(data, "Generated Dataset (Seed: 'stein')", True, "dataset_stein")

training_data, testing_data = format_data(data, 15)
prediction_x = build_prediction_timepoints(0.0, 20.0, 0.1)

print(training_data)
print(testing_data)
print(prediction_x)