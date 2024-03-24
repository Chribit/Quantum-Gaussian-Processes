import numpy as np
from scipy import integrate, interpolate
import math
import sys
from data import build_prediction_timepoints
from plot import plot_fitness
import math




def build_fitness_target (training_x, training_y, prediction_x, granularity):
    
    target_y = interpolate.CubicSpline(training_x, training_y)(prediction_x)
    target_aucs = []
    
    per_step_iterations = int(1 / granularity)
    
    for index in range(len(training_x[:-1])):
        
        target_auc = 0
        for iteration in range(per_step_iterations):
            
            prediction_index = index * per_step_iterations + iteration            
            target_auc += integrate.simpson(
                    target_y[prediction_index : prediction_index + 2],
                    prediction_x[prediction_index : prediction_index + 2]
                )
        target_aucs.append(target_auc)
        
    return (target_y, np.array(target_aucs))

def fitness (model, granularity, training_x, prediction_x, target_aucs, plot = False, filepath = "fitness_evaluation", training_y = False, target_y = False):
        
    # TODO: use covariance matrix instead of curve
        
    prediction_y, sigmas = model.predict(prediction_x)

    per_step_iterations = int(1 / granularity)
    x = 0
    
    for index in range( len(training_x[:-1])):
        
        prediction_auc = 0
        for iteration in range(per_step_iterations):
            
            prediction_index = index * per_step_iterations + iteration            
            prediction_auc += integrate.simpson(
                    prediction_y[prediction_index : prediction_index + 2],
                    prediction_x[prediction_index : prediction_index + 2]
                )
        
        x += abs(prediction_auc - target_aucs[index])
        
    if plot:
        
        if type(training_y) is not np.ndarray or type(target_y) is not np.ndarray:
            print("ERROR: fitness() requires the y coordinates of the target function built with build_fitness_target() if a plot should be created.")
        
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

    return 0.5 * (1.0 - np.tanh(x - np.pi))
    
