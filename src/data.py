import numpy as np
import pandas as pd
import sys




def generate_data (days, initial_value, seed = "42"):
    
    if ( not isinstance(seed, str) ):
        print("\nERROR: generate_data requires a string as a seed.\n")
        sys.exit(1)
    
    seed = hash(seed)
    
    current = fractal_brownian_motion(seed, 0)
    full_value = (1.0 / current) * initial_value
      
    output = pd.DataFrame({"time" : np.arange(days)})
    output["value"] = output["time"].apply(lambda day : float("%.2f" % (fractal_brownian_motion(seed, day) * full_value)))
        
    return output

def fractal_brownian_motion (seed, x):
    
    normalised_squared_seed = (seed % 1000000000) / 1000000000
    base_amplitude = normalised_squared_seed * 0.5
    
    amplitude_1 = base_amplitude * 0.8
    offset_1 = normalised_squared_seed * 5.1234
    frequency_1 = normalised_squared_seed * 0.01
 
    amplitude_2 = base_amplitude * 0.2
    offset_2 = normalised_squared_seed * 2.45182
    frequency_2 = normalised_squared_seed * 0.04
    
    amplitude_3 = base_amplitude * 0.1
    offset_3 = normalised_squared_seed * 8.45182
    frequency_3 = normalised_squared_seed * 0.16
    
    amplitude_4 = base_amplitude * 0.05
    offset_4 = normalised_squared_seed * 3.85810
    frequency_4 = normalised_squared_seed * 0.64
    
    amplitude_5 = base_amplitude * 0.04
    offset_5 = normalised_squared_seed * 10.06847
    frequency_5 = normalised_squared_seed * 2.56
    
    amplitude_6 = base_amplitude * 0.02
    offset_6 = normalised_squared_seed * 7.395812
    frequency_6 = normalised_squared_seed * 10.24
    
    output = 0.5
    output += np.sin( x * frequency_1 + offset_1 ) * amplitude_1
    output += np.sin( x * frequency_2 + offset_2 ) * amplitude_2
    output += np.sin( x * frequency_3 + offset_3 ) * amplitude_3
    output += np.sin( x * frequency_4 + offset_4 ) * amplitude_4
    output += np.sin( x * frequency_5 + offset_5 ) * amplitude_5
    output += np.sin( x * frequency_6 + offset_6 ) * amplitude_6
    
    return output

def format_data (data, training_window):
    
    training_data = data.head(training_window)
    testing_data = data.tail( len(data["time"]) - training_window )

    return (training_data, testing_data)

def angle_scaling (data, minimum, maximum, maximum_angle = np.pi):

    # TODO: method can be more adaptible if arctan or similar function used to normalise values

    data_range = maximum - minimum
    
    if data_range == 0.0:
        return np.full(len(data), 0.0)
    
    embedder = np.vectorize(lambda value: ((value - minimum) / data_range) * maximum_angle)
    
    return embedder(data)

def train_test_split (data, test_samples):
    
    days = len(data["time"])
    
    features = data["time"].values.reshape(days, 1)
    targets = data["value"].values.reshape(days, 1)
    
    if test_samples == 0:
        test_samples = -1 * days
    
    x_train = features[: -1 * test_samples]
    y_train = targets[: -1 * test_samples]
    x_test = features[-1 * test_samples :]
    y_test = targets[-1 * test_samples :]
    
    return x_train, y_train, x_test, y_test
