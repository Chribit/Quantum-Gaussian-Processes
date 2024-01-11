import numpy as np
import sys




class gaussian_process:
    
    def __init__ (self, dataset, kernel, quantum_circuit_callable = None, minimum_prediction_x = None, maximum_prediction_x = None, kernel_parameters = []):
        
        self.kernel = kernel
        self.x_train = dataset["time"].to_numpy()
        self.y_train = dataset["value"].to_numpy()
        self.quantum_circuit_callable = quantum_circuit_callable
        self.is_quantum = not isinstance(self.quantum_circuit_callable, type(None))
        
        if self.is_quantum and isinstance(minimum_prediction_x, type(None)):
            print("\nERROR: Quantum Gaussian Process models require an intended minimum for prediction x coordinates.")
            sys.exit(1)
        
        if self.is_quantum and isinstance(maximum_prediction_x, type(None)):
            print("\nERROR: Quantum Gaussian Process models require an intended maximum for prediction x coordinates.")
            sys.exit(1)
        
        if self.is_quantum:
            self.minimum_prediction_x = float(minimum_prediction_x) 
            self.maximum_prediction_x = float(maximum_prediction_x)
        
        self.set_kernel_parameters(kernel_parameters)
    
    def get_training_data (self):
        
        return (self.x_train, self.y_train)
    
    def set_kernel_parameters (self, parameters):
        
        self.kernel_parameters = parameters
        
        if self.is_quantum:
            self.circuit = self.quantum_circuit_callable(self.kernel_parameters, self.minimum_prediction_x, self.maximum_prediction_x)
            
        self.__build_covariance_matrix()
    
    def __build_covariance_matrix (self):
        
        cov_mat = []
        cov_mat_dimensions = len( self.x_train )
        
        if self.is_quantum:
            
            x1 = np.repeat(self.x_train, cov_mat_dimensions)
            x2 = np.tile(self.x_train, cov_mat_dimensions)
            
            matrix_cells = self.kernel(x1, x2, self.circuit)
            
            cov_mat = np.reshape(matrix_cells, (cov_mat_dimensions, cov_mat_dimensions))
            
            # cov_mat += np.identity(cov_mat_dimensions) * 0.001
            
        else:
            
            for row in range(cov_mat_dimensions):
                matrix_row = []
                for column in range(cov_mat_dimensions):
                    
                    matrix_row.append(
                        self.kernel(
                            self.x_train[row],
                            self.x_train[column],
                            self.kernel_parameters
                        ))
                    
                cov_mat.append(matrix_row)

        self.covariance_matrix = np.array(cov_mat)
        
    def sample_prior (self, samples, mean):
            
        output = np.random.multivariate_normal(
                mean = np.full(len(self.covariance_matrix), mean),
                cov = self.covariance_matrix, 
                size = samples
            )
            
        return output
    
    def sample_posterior (self, samples):
        
        mean = self.predict(self.x_train)[0]
            
        output = np.random.multivariate_normal(
                mean = mean,
                cov = self.covariance_matrix, 
                size = samples
            )
            
        return output
    
    def __invert_matrix (self, matrix):
        
        matrix = np.array(matrix)
        
        a, b = matrix.shape
        if a != b:
            raise ValueError("Only square matrices are invertible.")

        i = np.eye(a, a)
        return np.linalg.lstsq(matrix, i, rcond=None)[0]
        
    def predict (self, predicted_time_points):
        
        inv_cov_mat = self.__invert_matrix( self.covariance_matrix )
        applied_mat = inv_cov_mat.dot( self.y_train )

        predictions = []
        sigmas = []

        if self.is_quantum:
            
            x1 = np.repeat(predicted_time_points, len(self.x_train))
            x2 = np.tile(self.x_train, len(predicted_time_points))

            ks = self.kernel(x1, x2, self.circuit)            
            ks = np.reshape(ks, (-1, len(self.x_train)))
            
            predictions = np.apply_along_axis(lambda k: np.dot(k, applied_mat), 1, ks)
            
            self_similarity = self.kernel(predicted_time_points, predicted_time_points, self.circuit)
            summand = np.apply_along_axis(lambda k: np.dot(k, inv_cov_mat).dot(k), 1, ks)
            sigmas = np.abs(self_similarity - summand)
            
        else:
            
            for current_x in predicted_time_points:
                
                k = [self.kernel(current_x, train_x, self.kernel_parameters) for train_x in self.x_train]
                
                predictions.append(
                    np.dot(k, applied_mat)
                )
                sigmas.append(
                    abs(
                        self.kernel(current_x, current_x, self.kernel_parameters) - np.dot(k, inv_cov_mat).dot(k)
                    )
                )

        return (predictions, sigmas)
