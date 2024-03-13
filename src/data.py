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
    
    # full_value = 1.0
      
    output = pd.DataFrame({"time" : np.arange(days)})
    output["value"] = output["time"].apply(lambda day : float("%.2f" % (fractal_brownian_motion(seed, day) * full_value)))
        
    return output

def fractal_brownian_motion (seed, x):
    
    # output = np.sin(x) + 1.0
    # return output
    
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
