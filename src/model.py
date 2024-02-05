import pennylane as qml
# from pennylane import numpy as np
import numpy as np
from data import angle_scaling




class gaussian_process:
    
    # 1. constructor; recieves formatted training data, a callable kernel function, parameters for that kernel function
    def __init__ (self, training_data, kernel, kernel_parameters = np.array([]), is_quantum = False):
        
        # 1. set internal training x and y
        self.training_x = training_data["time"].to_numpy()
        self.training_y = training_data["value"].to_numpy()
        
        # 2. store kernel function
        self.kernel = kernel
        
        # 3. set is_quantum flag
        self.is_quantum = is_quantum
        
        # 4. if model is quantum
        if self.is_quantum:
            
            # 1. set wire count to be used in the quantum circuit
            self.circuit_wire_count = 9
            
            # 2. set parameter count per layer
            self.layer_parameter_count = 2 * ((4 ** 4) - 1)
        
        # 5. set provided kernel parameters and build covariance matrix
        self.set_kernel_parameters(kernel_parameters)
            
    # 2. updates kernel_parameters field in model and builds the covariance matrix on update
    def set_kernel_parameters (self, kernel_parameters = np.array([])):
        
        # 1. store provided kernel parameters
        self.kernel_parameters = kernel_parameters
        
        # 2. if kernel_parameters is an empty array
        if self.kernel_parameters.size == 0:
            
            # 1. prevent further execution
            return
        
        # 3. if model is quantum
        if self.is_quantum:
            
            # 1. rebuild quantum circuit
            self.build_quantum_circuit()
        
        # 4. build covariance matrix
        self.build_covariance_matrix()
    
    # 3. builds the gaussian process covariance matrix using models internal state
    def build_covariance_matrix (self):
        
        # 1. initialise empty covariance matrix
        covariance_matrix = []
        
        # 2. fetch the covariance matrix dimensions from the training data
        matrix_dimensions = len(self.training_x)
        
        # 3. if model is quantum
        if self.is_quantum:
            
            # 1. generate a repeating sequence of training x values "matrix dimension" amount of times
            x1 = np.repeat(self.training_x, matrix_dimensions)
            
            # 2. generate a sequence of each training x value with "matrix dimension" amount
            x2 = np.tile(self.training_x, matrix_dimensions)
            
            # 3. generate the matrix cell values using the quantum kernel function - providing a built circuit
            matrix_cells = self.kernel(x1, x2, self.quantum_circuit)
            
            # 4. store a reshaped version of the 1D result as a 2D numpy array
            covariance_matrix = np.reshape(matrix_cells, (matrix_dimensions, matrix_dimensions))
        
        # 4. if model is classical
        else:
            
            # 1. iterate over each row in the matrix
            for row in range(matrix_dimensions):
                
                # 1. initialise empty matrix row
                matrix_row = []
                
                # 2. iterate over each column in the matrix
                for column in range(matrix_dimensions):
                    
                    # 1. calculate a matrix cell using the kernel function; append to matrix row
                    matrix_row.append(
                        self.kernel(
                            self.training_x[row],
                            self.training_x[column],
                            self.kernel_parameters
                        )
                    )
                
                # 3. append built matrix row to new covariance matrix
                covariance_matrix.append(matrix_row)

        # 5. store the new covariance matrix in numpy array form within the model
        self.covariance_matrix = np.array(covariance_matrix)
    
    # 4. builds a quantum reupload circuit with SU intermediates for the model
    def build_quantum_circuit (self):
        
        # 1. if model isn't quantum
        if not self.is_quantum:
            
            # 1. return early
            return
        
        # 2. reshape kernel parameters to be split into subarrays for each layer
        circuit_parameters = np.reshape(self.kernel_parameters, (-1, self.layer_parameter_count))
        
        # 3. define a qml function to build the quantum circuit
        @qml.qnode(qml.device('lightning.qubit', self.circuit_wire_count))
        def circuit (x1, x2):
            
            # 1. add a hadamard gate on the measurement wire
            qml.Hadamard(0)
            
            # 2. concatenate provided x values to be compared in the kernel circuit
            fused = np.concatenate((x1, x2))
            
            # 3. apply angle scaling to concatenated values --> provide range of x values to train on
            scaled = angle_scaling(fused, 0.0, 11.0)
            
            # 4. overwrite inputs with scaled versions
            x1 = scaled[:len(x1)]
            x2 = scaled[len(x1):]
            
            # 5. iterate over circuit parameter subarrays
            for parameters in circuit_parameters:
                
                # 1. split parameters into x1 and x2 subarrays
                split_parameters = np.split(parameters, 2)
                
                # 2. add rx gates for x1 value reupload layer
                qml.RX(x1, 1)
                qml.RX(x1, 2)
                qml.RX(x1, 3)
                qml.RX(x1, 4)
                
                # 3. add a special unitary for x1
                qml.SpecialUnitary(split_parameters[0], [1, 2, 3, 4])
                
                # 4. add rx gates for x2 value reupload layer
                qml.RX(x2, 5)
                qml.RX(x2, 6)
                qml.RX(x2, 7)
                qml.RX(x2, 8)
                
                # 5. add a special unitary for x2
                qml.SpecialUnitary(split_parameters[1], [5, 6, 7, 8])
                
                # 6. add barrier visual along all reupload wires for cleaner circuit appearance
                qml.Barrier([1, 2, 3, 4, 5, 6, 7, 8], only_visual = True)

            # 6. add cswap gates from the measurement wire to each pair of reupload wires
            qml.CSWAP([0, 1, 5])
            qml.CSWAP([0, 2, 6])
            qml.CSWAP([0, 3, 7])
            qml.CSWAP([0, 4, 8])

            # 7. add a hadamard gate on the measurement wire
            qml.Hadamard(0)

            # 8. measure qubit on measurement wire and return the probabilities
            return qml.probs(0)
        
        # 4. store circuit builder function in model
        self.quantum_circuit = circuit
    
    # 5. plots a quantum circuit 
    def plot_quantum_circuit (self):
        
        # 1. if model isn't quantum
        if not self.is_quantum:
            
            # 1. return early
            return
        
        # 2. fetch diagram caller function via pennylane draw_mpl function
        drawer = qml.draw_mpl(self.quantum_circuit, show_all_wires = True, decimals = 2, fontsize = 18, style = "solarized_light")
        
        # 3. call drawer function with throwaway parameters --> same parameter signature as circuit function in self.quantum_circuit
        fig, ax = drawer([0], [1])
        
        # 4. return figure
        return fig
    
    # def sample_prior (self, samples, mean):
            
    #     output = np.random.multivariate_normal(
    #             mean = np.full(len(self.covariance_matrix), mean),
    #             cov = self.covariance_matrix, 
    #             size = samples
    #         )
            
    #     return output
    
    # def sample_posterior (self, samples):
        
    #     mean = self.predict(self.x_train)[0]
            
    #     output = np.random.multivariate_normal(
    #             mean = mean,
    #             cov = self.covariance_matrix, 
    #             size = samples
    #         )
            
    #     return output
    
    # def __invert_matrix (self, matrix):
        
    #     matrix = np.array(matrix)
        
    #     a, b = matrix.shape
    #     if a != b:
    #         raise ValueError("Only square matrices are invertible.")

    #     i = np.eye(a, a)
    #     return np.linalg.lstsq(matrix, i, rcond=None)[0]
        
    # def predict (self, predicted_time_points):
        
    #     inv_cov_mat = self.__invert_matrix( self.covariance_matrix )
    #     applied_mat = inv_cov_mat.dot( self.y_train )

    #     predictions = []
    #     sigmas = []

    #     if self.is_quantum:
            
    #         x1 = np.repeat(predicted_time_points, len(self.x_train))
    #         x2 = np.tile(self.x_train, len(predicted_time_points))

    #         ks = self.kernel(x1, x2, self.circuit)            
    #         ks = np.reshape(ks, (-1, len(self.x_train)))
            
    #         predictions = np.apply_along_axis(lambda k: np.dot(k, applied_mat), 1, ks)
            
    #         self_similarity = self.kernel(predicted_time_points, predicted_time_points, self.circuit)
    #         summand = np.apply_along_axis(lambda k: np.dot(k, inv_cov_mat).dot(k), 1, ks)
    #         sigmas = np.abs(self_similarity - summand)
            
    #     else:
            
    #         for current_x in predicted_time_points:
                
    #             k = [self.kernel(current_x, train_x, self.kernel_parameters) for train_x in self.x_train]
                
    #             predictions.append(
    #                 np.dot(k, applied_mat)
    #             )
    #             sigmas.append(
    #                 abs(
    #                     self.kernel(current_x, current_x, self.kernel_parameters) - np.dot(k, inv_cov_mat).dot(k)
    #                 )
    #             )

    #     return (predictions, sigmas)
