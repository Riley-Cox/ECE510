import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# XOR dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0], [1], [1], [0]])

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(output):
    return output * (1 - output)

# Initialize weights and biases
np.random.seed(1)
input_size = 2
hidden_size = 2
output_size = 1

weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.5
bias_hidden = np.zeros((1, hidden_size))

weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.5
bias_output = np.zeros((1, output_size))

# Training config
epochs = 10000
learning_rate = 0.1
interval = 200  # Save every 200 epochs

# Store predictions for animation
predictions_history = []

for epoch in range(epochs):
    # Forward pass
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(final_input)

    # Save frames for GIF
    if epoch % interval == 0:
        predictions_history.append(predicted_output.copy())
        loss = np.mean(np.square(y - predicted_output))
        print(f"Epoch {epoch} - Loss: {loss:.5f}")

    # Backpropagation
    error = y - predicted_output
    d_output = error * sigmoid_derivative(predicted_output)

    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Update weights and biases
    weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    weights_input_hidden += X.T.dot(d_hidden) * learning_rate
    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

# Create plot
fig, ax = plt.subplots()

def update(frame):
    ax.clear()
    ax.set_title(f"Epoch: {frame * interval}")
    preds = predictions_history[frame]
    colors = ['red' if p < 0.5 else 'blue' for p in preds.flatten()]
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=200)
    for i, txt in enumerate(np.round(preds.flatten(), 2)):
        ax.annotate(txt, (X[i, 0] + 0.05, X[i, 1] + 0.05), fontsize=12)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")

anim = FuncAnimation(fig, update, frames=len(predictions_history), interval=200)

# Save to GIF
anim.save("xor_training.gif", writer=PillowWriter(fps=5))
print("✅ GIF saved as xor_training.gif")

