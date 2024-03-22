import numpy as np
import pandas as pd
import sys


    

def generate_data (days, initial_value, seed = "42"):
    
    if ( not isinstance(seed, str) ):
        print("\nERROR: generate_data requires a string as a seed.\n")
        sys.exit(1)
    
    seed = hash(seed)
    
    parameters = generate_parameters(seed)
    
    current = fractal_brownian_motion(parameters, 0)
    full_value = (1.0 / current) * initial_value
      
    output = pd.DataFrame({"time" : np.arange(days)})
    output["value"] = output["time"].apply(lambda day : float("%.2f" % (fractal_brownian_motion(parameters, day) * full_value)))
    
    return output

def generate_parameters (seed):
    
    modulised_seed = seed % (2 ** 32)
    normalised_seed = modulised_seed / (2 ** 32)

    amplitude_1 = normalised_seed * (np.e ** (-1 * 0.5))
    offset_1 = normalised_seed * (modulised_seed / (2 ** 10))
    frequency_1 = normalised_seed * (0.01 * (2.5 ** 0))
 
    amplitude_2 = normalised_seed * (np.e ** (-2 * 0.5))
    offset_2 = normalised_seed * (modulised_seed / (2 ** 9))
    frequency_2 = normalised_seed * (0.01 * (2.5 ** 1))
    
    amplitude_3 = normalised_seed * (np.e ** (-3 * 0.5))
    offset_3 = normalised_seed * (modulised_seed / (2 ** 8))
    frequency_3 = normalised_seed * (0.01 * (2.5 ** 2))
    
    amplitude_4 = normalised_seed * (np.e ** (-4 * 0.5))
    offset_4 = normalised_seed * (modulised_seed / (2 ** 7))
    frequency_4 = normalised_seed * (0.01 * (2.5 ** 3))
    
    amplitude_5 = normalised_seed * (np.e ** (-5 * 0.5))
    offset_5 = normalised_seed * (modulised_seed / (2 ** 6))
    frequency_5 = normalised_seed * (0.01 * (2.5 ** 4))
    
    amplitude_6 = normalised_seed * (np.e ** (-6 * 0.5))
    offset_6 = normalised_seed * (modulised_seed / (2 ** 5))
    frequency_6 = normalised_seed * (0.01 * (2.5 ** 5))
    
    amplitude_7 = normalised_seed * (np.e ** (-7 * 0.5))
    offset_7 = normalised_seed * (modulised_seed / (2 ** 4))
    frequency_7 = normalised_seed * (0.01 * (2.5 ** 6))
    
    amplitude_8 = normalised_seed * (np.e ** (-8 * 0.5))
    offset_8 = normalised_seed * (modulised_seed / (2 ** 3))
    frequency_8 = normalised_seed * (0.01 * (2.5 ** 7))
    
    amplitude_9 = normalised_seed * (np.e ** (-9 * 0.5))
    offset_9 = normalised_seed * (modulised_seed / (2 ** 2))
    frequency_9 = normalised_seed * (0.01 * (2.5 ** 8))
    
    amplitude_10 = normalised_seed * (np.e ** (-10 * 0.5))
    offset_10 = normalised_seed * (modulised_seed / (2 ** 1))
    frequency_10 = normalised_seed * (0.01 * (2.5 ** 9))
    
    return [
        [amplitude_1, offset_1, frequency_1],
        [amplitude_2, offset_2, frequency_2],
        [amplitude_3, offset_3, frequency_3],
        [amplitude_4, offset_4, frequency_4],
        [amplitude_5, offset_5, frequency_5],
        [amplitude_6, offset_6, frequency_6],
        [amplitude_7, offset_7, frequency_7],
        [amplitude_8, offset_8, frequency_8],
        [amplitude_9, offset_9, frequency_9],
        [amplitude_10, offset_10, frequency_10]
    ]

def fractal_brownian_motion (parameters, x):
    
    output = 0.5
    
    for layer_index in range(len(parameters)):
        output += np.sin( parameters[layer_index][2] * (x + parameters[layer_index][1])) * parameters[layer_index][0]

    return output

def format_data (data, training_window):
    
    training_data = data.head(training_window)
    testing_data = data.tail( len(data["time"]) - training_window )

    return (training_data, testing_data)

def build_prediction_timepoints (start = 0.0, end = 10.0, step = 1.0):

    return np.arange(start, (end - 1.0) + step, step)

def angle_scaling (data, minimum, maximum, maximum_angle = np.pi):

    # TODO: method can be more adaptible if arctan or similar function used to normalise values --> no more minimum and maximum

    data_range = maximum - minimum
    
    if data_range == 0.0:
        return np.full(len(data), 0.0)
    
    embedder = np.vectorize(lambda value: ((value - minimum) / data_range) * maximum_angle)
    
    return embedder(data)

def invert_matrix (matrix):
    
    # 1. convert matrix to a numpy array if not already the case
    matrix = np.array(matrix)
    
    # 2. get dimensions of provided matrix
    dimension_x, dimension_y = matrix.shape
    
    # 3. if matrix dimensions are not equal and therefor the matrix isn't square
    if dimension_x != dimension_y:
        
        # 1. throw error
        raise ValueError("ERROR: Only square matrices are invertible.")

    # 4. get an identity matrix of the determined dimensions
    identity = np.eye(dimension_x, dimension_x)
    
    # 5. return the least squares solution to the linear matrix equation of input matrix and identity
    return np.linalg.lstsq(matrix, identity, rcond = None)[0]
