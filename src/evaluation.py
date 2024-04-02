import numpy as np
from scipy import integrate, interpolate
import sys
from plot import plot_fitness




def build_fitness_target_AUC (training_x, training_y, prediction_x, granularity):
    
    target_y = interpolate.CubicSpline(training_x, training_y)(prediction_x)
    target_aucs = []
    
    for index in range(len(training_x[:-1])):
        
        target_auc = 0
        for iteration in range(granularity):
            
            prediction_index = index * granularity + iteration    
            target_auc += integrate.simpson(
                    target_y[prediction_index : prediction_index + 2],
                    prediction_x[prediction_index : prediction_index + 2]
                )
        target_aucs.append(target_auc)
        
    return (target_y, np.array(target_aucs))

def build_fitness_target_SMD (training_y):
    
    matrix_dimension = len(training_y)
    diagonal_smoothing = 0.3
    
    targets = []
    for x2 in range(matrix_dimension):
        targets.append(
            np.exp(1.0 / (((-x2 * diagonal_smoothing) ** 2) + (1.0 / np.log(2)))) - 1.0
        )
    targets = np.array(targets)
    targets = np.gradient(targets)
    
    return targets

def fitness (model, granularity, training_x, prediction_x, target_aucs = None, target_slopes = None, plot = False, filepath = "fitness_evaluation", training_y = False, target_y = False):

    # Slope Manhatten Distance based approach
    if target_aucs is None:
        
        if target_slopes is None:
            print("ERROR: fitness() requires the target_slopes built via the build_firntess_target_KLD() function if no target_AUCs are provided.")
            sys.exit()  
        
        row_slopes = np.gradient(model.covariance_matrix[0])
        manhatten_distance = np.sum(np.abs(np.subtract(row_slopes, target_slopes)))
        
        return 0.5 * (1.0 + np.tanh(manhatten_distance - np.pi))
    
    # AUC based approach
    else:
        
        prediction_y, sigmas = model.predict(prediction_x)

        steps = len(training_x[:-1])
              
        x = 0
        
        for index in range(steps):
            
            prediction_auc = 0
            for iteration in range(granularity):
                
                prediction_index = index * granularity + iteration            
                prediction_auc += integrate.simpson(
                        prediction_y[prediction_index : prediction_index + 2],
                        prediction_x[prediction_index : prediction_index + 2]
                    )
            
            x += abs(prediction_auc - target_aucs[index])
            
        x = x / steps
            
        if plot:
            
            if type(training_y) is not np.ndarray or type(target_y) is not np.ndarray:
                print("ERROR: fitness() requires the y coordinates of the target function built with build_fitness_target_AUC() if a plot should be created.")
                sys.exit()
            
            plot_fitness(
                "Model Fitness",
                training_x,
                training_y,
                prediction_x,
                target_y,
                prediction_x,
                prediction_y,
                sigmas,
                False,
                False,
                True,
                filepath
            )

        return np.e ** -x
    
