# SOURCES:
# https://www.cs.toronto.edu/~duvenaud/cookbook/
# https://peterroelants.github.io/posts/gaussian-process-kernels/ 
# https://distill.pub/2019/visual-exploration-gaussian-processes/




import numpy as np




def constant (x1, x2, parameters):
    amplitude = parameters[0]
    return amplitude

def linear (x1, x2, parameters):
    variance = parameters[0]
    certainty = parameters[1]
    offset = parameters[2]
    return (certainty ** 2) + (variance ** 2) * (x1 - offset) * (x2 - offset)

def sine_squared (x1, x2, parameters):
    variance = parameters[0]
    length = parameters[1]
    periodicity = parameters[2]
    return (variance ** 2) * np.exp(-1 * ( (np.sin( (np.pi * np.abs(x1 - x2)) / periodicity ) ** 2) / (length ** 2) ))

def white_noise (x1, x2, parameters):
    variance = parameters[0]
    if (x1 == x2):
        return variance ** 2
    else:
        return 0

def rbf (x1, x2, parameters):
    variance = parameters[0]
    length = parameters[1]
    return (variance ** 2) * np.exp(-1 * ( ((x1 - x2) ** 2) / (2 * (length ** 2)) ))

def rational_quadratic (x1, x2, parameters):
    variance = parameters[0]
    length = parameters[1]
    weighting = parameters[2]
    return (variance ** 2) * (1 + ( ((x1 - x2) ** 2) / (2 * weighting * (length ** 2)) )) ** (-1 * weighting)




def classical_kernel_1 (x1, x2, parameters):
    
    output = white_noise(x1, x2, parameters[:1])
    output += rbf(x1, x2, parameters[1:3])
    output += constant(x1, x2, parameters[3:4])
    output += sine_squared(x1, x2, parameters[4:])
    
    # inverse to generated target function for target slopes in manhatten distance approach --> use to test accurate fitness scores 
    # output = 1.0 - np.exp(1.0 / ((((x1 - x2) * 0.3) ** 2) + (1.0 / np.log(2)))) + 1.0
    
    return output

def classical_kernel_2 (x1, x2, parameters):
    
    return (np.exp(1.0 / ((((x1 - x2) * parameters[0]) ** 2) + (1.0 / np.log(2)))) - 1.0) * parameters[1]




def quantum_kernel_1 (x1, x2, circuit):
    
    output = circuit(x1, x2)[:, 0] * 1000.0
    return output
