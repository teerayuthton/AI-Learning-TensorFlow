# This code demonstrates a simple forward propagation in a neural network.
# It calculates the output of a neural network with one hidden layer.
# The network has two input nodes, two hidden nodes, and one output node.
# The weights are defined for each node, and the output is calculated by multiplying the inputs with the weights.
import numpy as np

input_data = np.array([2, 3])

weights = {'node_0': np.array([1, 1]),
           'node_1': np.array([-1, 1]),
           'output': np.array([2, -1])}

node_0_value = (input_data * weights['node_0']).sum()
node_1_value = (input_data * weights['node_1']).sum()

hidden_layer_values = np.array([node_0_value, node_1_value])

output = (hidden_layer_values * weights['output']).sum()

print("Output:", output)