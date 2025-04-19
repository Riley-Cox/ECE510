import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function for gradient calculation
def sigmoid_derivative(x):
    return x * (1 - x)

# Perceptron learning rule
def train_perceptron(inputs, targets, learning_rate, epochs):
    # Initialize weights and bias randomly
    weights = np.random.randn(inputs.shape[1])
    bias = np.random.randn()
    
    # Training loop
    for epoch in range(epochs):
        total_error = 0
        for i in range(inputs.shape[0]):
            x = inputs[i]
            t = targets[i]
            
            # Compute the weighted sum
            weighted_sum = np.dot(x, weights) + bias
            
            # Apply the sigmoid activation function
            y = sigmoid(weighted_sum)
            
            # Calculate the error (target - output)
            error = t - y
            total_error += error ** 2
            
            # Update weights and bias using the perceptron rule
            weights += learning_rate * error * x * sigmoid_derivative(y)
            bias += learning_rate * error * sigmoid_derivative(y)
        
        # Print the total error for the epoch (optional)
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Total Error: {total_error}')
    
    return weights, bias

# Function to test the trained neuron
def test_perceptron(inputs, weights, bias):
    outputs = []
    for x in inputs:
        weighted_sum = np.dot(x, weights) + bias
        output = sigmoid(weighted_sum)
        outputs.append(output)
    return np.array(outputs)

# NAND function
nand_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
nand_targets = np.array([1, 1, 1, 0])

# XOR function
xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_targets = np.array([0, 1, 1, 0])
# Set learning rate and number of epochs
learning_rate = 0.1
epochs = 10000

# Train and test the NAND neuron
print("Training for NAND function...")
nand_weights, nand_bias = train_perceptron(nand_inputs, nand_targets, learning_rate, epochs)
nand_output = test_perceptron(nand_inputs, nand_weights, nand_bias)
print("\nNAND Outputs:", nand_output)

# Train and test the XOR neuron
print("\nTraining for XOR function...")
xor_weights, xor_bias = train_perceptron(xor_inputs, xor_targets, learning_rate, epochs)
xor_output = test_perceptron(xor_inputs, xor_weights, xor_bias)
print("\nXOR Outputs:", xor_output)


