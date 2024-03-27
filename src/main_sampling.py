from data import generate_data, format_data, build_prediction_timepoints
from plot import plot_dataset, plot_samples
from model import gaussian_process
from kernel import classical_kernel_1
import numpy as np




days = 365
seed = "vn96e5t40wr3aijis"

data = generate_data(days, 69, seed)
plot_dataset(data, "Generated Dataset (Seed: '" + seed + "')", True, "sampling/dataset")

training_window = 360
prediction_granularity = 0.25

training_data, testing_data = format_data(data, training_window)
prediction_x = build_prediction_timepoints(0.0, float(days), prediction_granularity)




model = gaussian_process(
    training_data,
    classical_kernel_1,
    np.array([0.12554, 7.6746, 1.80564, 85.04183, 10.0, 1.33497, 38.42345])
)

training_x, training_y = model.get_training_data()
prior_mean = np.average(training_y)

sample_count = 10

prior_samples = model.sample_prior(sample_count, prior_mean)
posterior_samples = model.sample_posterior(sample_count)

plot_samples(training_x, prior_samples, [-50, 175], True, "sampling/prior")
plot_samples(training_x, posterior_samples, [-50, 175], True, "sampling/posterior")
