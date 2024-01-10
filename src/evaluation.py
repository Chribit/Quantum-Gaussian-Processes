import numpy as np
from scipy import integrate
import math
import sys




def fitness (model, evaluation_granularity):
    
    if (evaluation_granularity > 1.0 or evaluation_granularity < 0.0):
        print("\nERROR: evaluation_granularity in fitness() has to be between 0.0 and 1.0")
        sys.exit(1)
    
    training_x, training_y = model.get_training_data()
    prediction_x = np.arange(training_x[0], training_x[-1] + evaluation_granularity, evaluation_granularity)
    prediction_y = model.predict(prediction_x)[0] # TODO: factor in sigmas -> higher uncertainty = lower fitness = higher value
    # FIXME: fitness may be less accurate if kernel has full period between two training points
    
    if (math.modf(1.0 / evaluation_granularity)[0] != 0):
        print("\nERROR: evaluation_granularity in fitness() has to divide 1 without remainder")
        sys.exit(1)
    
    per_step_iterations = int(1 / evaluation_granularity)
    
    fitness = 0
    
    for index in range(len(training_x[:-1])):
        
        training_auc = integrate.simpson(
                training_y[index : index + 2],
                training_x[index : index + 2]
            )
        
        prediction_auc = 0
        for iteration in range(per_step_iterations):
            
            prediction_index = index * per_step_iterations + iteration
            prediction_auc += integrate.simpson(
                    prediction_y[prediction_index : prediction_index + 2],
                    prediction_x[prediction_index : prediction_index + 2]
                )
        
        fitness += abs(prediction_auc - training_auc)

    return fitness
    
