import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import sys

from data import angle_scaling



 
class swap_test_circuit:

    def __init__ (self, circuit_parameters, circuit_type, angle_scaling_minimum, angle_scaling_maximum):
        
        if (circuit_type != "5" and circuit_type != "11"):
            print("\nERROR: Circuit type can only be a string with the value '5' or '11'.")
            sys.exit(1)
        
        self.angle_scaling_minimum = angle_scaling_minimum
        self.angle_scaling_maximum = angle_scaling_maximum
        self.state_qubits = 4
        self.ancilla_qubit = 1
        self.n_qubits = self.state_qubits * 2 + self.ancilla_qubit
        
        circuit = None
        if (circuit_type == "5"):
            
            if (len(circuit_parameters) % 28 != 0):
                print("\nERROR: Circuit 5 parameters must be a multiple of 28 in number.")
                sys.exit(1)
            
            circuit_parameters = circuit_parameters.reshape(-1, 28)
            circuit = self.build_quantum_circuit_5
            
        elif (circuit_type == "11"):
            
            if (len(circuit_parameters) % 12 != 0):
                print("\nERROR: Circuit 11 parameters must be a multiple of 12 in number.")
                sys.exit(1)
            
            circuit_parameters = circuit_parameters.reshape(-1, 12)
            circuit = self.build_quantum_circuit_11
        
        @qml.qnode( qml.device('lightning.qubit', 9) )
        def kernel (x1, x2):
            
            qml.Hadamard(0)
            
            fused = np.concatenate((x1, x2))
            scaled = angle_scaling(fused, self.angle_scaling_minimum, self.angle_scaling_maximum)
            
            x1 = scaled[:len(x1)]
            x2 = scaled[len(x1):]
            
            for parameters in circuit_parameters:
                
                qml.RX(x1, 1)
                qml.RX(x1, 2)
                qml.RX(x1, 3)
                qml.RX(x1, 4)
                
                circuit([4, 3, 2, 1], parameters)
                
                qml.RX(x2, 5)
                qml.RX(x2, 6)
                qml.RX(x2, 7)
                qml.RX(x2, 8)
                
                circuit([8, 7, 6, 5], parameters)

            for i in range(4):
                qml.CSWAP([0, i + 1, i + 5])

            qml.Hadamard(0)

            return qml.probs(0)
         
        self.kernel = kernel
    
    def build_quantum_circuit_5 (self, wires, parameters):

        index = 0
        for i in wires:
            qml.RX(parameters[index], wires = i)
            index += 1
            # qml.RZ(parameters[index], wires = i)
            qml.RY(parameters[index], wires = i)
            index += 1

        for j in wires:
            for i in wires:
                if j != i:
                    # qml.CRZ(parameters[index], wires = [j, i])
                    qml.CRY(parameters[index], wires = [j, i])
                    index += 1
            
        for k in wires:
            qml.RX(parameters[index], wires = k)
            index += 1
            # qml.RZ(parameters[index], wires = k)
            qml.RY(parameters[index], wires = k)
            index += 1

    def build_quantum_circuit_11 (self, wires, parameters):

        index = 0
        for i in wires:
            qml.RX(parameters[index], wires = i)
            index += 1
            # qml.RZ(parameters[index], wires = i)
            qml.RY(parameters[index], wires = i)
            index += 1

        qml.CNOT(wires = [wires[-2], wires[-1]])
        qml.CNOT(wires = [wires[-3], wires[-4]])

        qml.RX(parameters[index], wires = wires[-2])
        index += 1
        qml.RX(parameters[index], wires = wires[-3])
        index += 1
        # qml.RZ(parameters[index], wires = wires[-2])
        qml.RY(parameters[index], wires = wires[-2])
        index += 1
        # qml.RZ(parameters[index], wires = wires[-3])
        qml.RY(parameters[index], wires = wires[-3])
        
        qml.CNOT(wires = [wires[-3], wires[-2]])

    def __call__ (self, x1, x2):
        
        results = self.kernel(x1, x2)
        
        if results.shape == (2,):
            results = [results[1]]
        else:
            results = results[:, 1]
        
        results = np.clip(results, 0.0, 0.5)
        results = 1.0 - 2.0 * results
        
        return results
        
    def plot (self):
        
        qml.draw_mpl(self.kernel, show_all_wires = True, decimals = 2)([1], [1])
        plt.show()
    
    
    

# FIXME: redo angle embedding with manual implementation
class inversion_test_circuit:

    def __init__ (self, circuit_parameters, circuit_type, angle_scaling_minimum, angle_scaling_maximum):
        
        if (len(circuit_parameters) == 28):
            print("\nERROR: Circuit 5 parameters must be exactly 28 in number.")
            sys.exit(1)
            
        self.angle_scaling_minimum = angle_scaling_minimum  
        self.angle_scaling_maximum = angle_scaling_maximum
        self.n_qubits = 4
        dev_kernel = qml.device('lightning.qubit', wires = self.n_qubits)
        
        self.params = circuit_parameters
        n = circuit_parameters.reshape(-1, 28).shape[0]
        self.n = n
        circuit = self.build_quantum_circuit_5

        @qml.qnode(dev_kernel, interface='autograd')
        def kernel(x1, x2):

            # embedding
            for i in range(n):
                circuit(x1, wires=[3, 2, 1, 0])
                
            # reverse embedding
            for i in range(n):
                qml.adjoint(circuit)(x2, wires=[3, 2, 1, 0])
                
            # return the probabilities for measuring |0...0>, |0...1>, ... , |1...1>
            return qml.probs(wires = range(self.n_qubits))

        self.kernel = kernel

    def build_quantum_circuit_5 (self, x, wires):

        qml.AngleEmbedding(features = [x] * 4, wires = wires)
        
        index = 0
        for i in wires:
            qml.RX(self.params[index], wires = i)
            index += 1
            # qml.RZ(self.params[index], wires = i)
            qml.RY(self.params[index], wires = i)
            index += 1

        for j in wires:
            for i in wires:
                if j != i:
                    # qml.CRZ(self.params[index], wires = [j, i])
                    qml.CRY(self.params[index], wires = [j, i])
                    index += 1
            
        for i in wires:
            qml.RX(self.params[index], wires = i)
            index += 1
            # qml.RZ(self.params[index], wires = i)
            qml.RY(self.params[index], wires = i)
            index += 1

    def __call__ (self, x1, x2):
   
        return self.kernel(x1, x2)[0]
       
    def plot (self):
        
        qml.draw_mpl(self.kernel, show_all_wires = True, decimals = 2)([1] * 4, [1] * 4)
        plt.show()




class data_reupload_circuit:

    def __init__ (self, circuit_parameters, circuit_type, angle_scaling_minimum, angle_scaling_maximum):
        
        if (circuit_type != "11"):
            print("\nERROR: Circuit type can only be a string with the value '11'.")
            sys.exit(1)
        
        self.angle_scaling_minimum = angle_scaling_minimum
        self.angle_scaling_maximum = angle_scaling_maximum
        self.state_qubits = 4
        self.ancilla_qubit = 1
        self.n_qubits = self.state_qubits * 2 + self.ancilla_qubit
        
        circuit = None      
        if (circuit_type == "11"):
            
            if (len(circuit_parameters) % 12 != 0):
                print("\nERROR: Circuit 11 parameters must be a multiple of 12 in number.")
                sys.exit(1)
            
            circuit_parameters = circuit_parameters.reshape(-1, 12)
            circuit = self.build_quantum_circuit_11
        
        @qml.qnode( qml.device('lightning.qubit', 9) )
        def kernel (x1, x2):
            
            qml.Hadamard(0)
            
            fused = np.concatenate((x1, x2))
            scaled = angle_scaling(fused, self.angle_scaling_minimum, self.angle_scaling_maximum)
            
            x1 = scaled[:len(x1)]
            x2 = scaled[len(x1):]
            
            for parameters in circuit_parameters:
                
                qml.RX(x1, 1)
                qml.RX(x1, 2)
                qml.RX(x1, 3)
                qml.RX(x1, 4)
                
                qml.RX(x2, 5)
                qml.RX(x2, 6)
                qml.RX(x2, 7)
                qml.RX(x2, 8)
                
                circuit(parameters)

            for i in range(4):
                qml.CSWAP([0, i + 1, i + 5])

            qml.Hadamard(0)

            return qml.probs(0)
         
        self.kernel = kernel
    
    def build_quantum_circuit_11 (self, parameters):
        
        qml.RY(parameters[0], 1)
        qml.RY(parameters[1], 2)
        qml.RY(parameters[2], 3)
        qml.RY(parameters[3], 4)
        qml.RY(parameters[0], 5)
        qml.RY(parameters[1], 6)
        qml.RY(parameters[2], 7)
        qml.RY(parameters[3], 8)

        qml.RZ(parameters[4], 1)
        qml.RZ(parameters[5], 2)
        qml.RZ(parameters[6], 3)
        qml.RZ(parameters[7], 4)
        qml.RZ(parameters[4], 5)
        qml.RZ(parameters[5], 6)
        qml.RZ(parameters[6], 7)
        qml.RZ(parameters[7], 8)
        
        qml.CNOT([2, 1])
        qml.CNOT([4, 3])
        qml.CNOT([6, 5])
        qml.CNOT([8, 7])
        
        qml.RY(parameters[8], 2)
        qml.RY(parameters[9], 3)
        qml.RY(parameters[8], 6)
        qml.RY(parameters[9], 7)
        
        qml.RZ(parameters[10], 2)
        qml.RZ(parameters[11], 3)
        qml.RZ(parameters[10], 6)
        qml.RZ(parameters[11], 7)
        
        qml.CNOT([3, 2])
        qml.CNOT([7, 6])
        
        qml.Barrier([1, 2, 3, 4, 5, 6, 7, 8], only_visual = True)

    def __call__ (self, x1, x2):
        
        results = self.kernel(x1, x2)
        
        if results.shape == (2,):
            results = [results[1]]
        else:
            results = results[:, 1]
        
        results = np.clip(results, 0.0, 0.5)
        results = 1.0 - 2.0 * results
        
        return results
        
    def plot (self):
        
        qml.draw_mpl(self.kernel, show_all_wires = True, decimals = 2, fontsize = 7)([1], [1])
        plt.show()
    