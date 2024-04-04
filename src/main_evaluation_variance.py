from data import generate_data, format_data, generate_seed, build_prediction_timepoints
from model import gaussian_process
from kernel import classical_kernel_2
from plot import plot_variances, plot_prediction
import numpy as np




days = 20
training_window = 15
granularity = 0.25
sample_count = 100




smoothing_parameters = np.arange(0.1, 4.1, 0.1)
variances = []




for smoothing in smoothing_parameters:
    
    samples = []
    
    for sample in range(sample_count):
        
        print("%.1f" % smoothing, "-", sample)
        
        seed = generate_seed(10)
        data = generate_data(days, 200, seed)
        training_data, testing_data = format_data(data, training_window)
        prediction_x = build_prediction_timepoints(training_window, float(days), granularity)
        model = gaussian_process(
            training_data,
            classical_kernel_2,
            np.array([smoothing, 1.0])
        )
        predictions, sigmas = model.predict(prediction_x)
        average_variance = np.average(sigmas)
        samples.append(average_variance)
        
    variances.append(np.average(samples))




plot_variances(variances, smoothing_parameters, True, "evaluation/variance/variance_smoothings")
