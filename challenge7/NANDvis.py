import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(output):
    return output * (1 - output)

# NAND dataset
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([1, 1, 1, 0])

# Training parameters
learning_rate = 0.1
epochs = 200
history = []

# Initialize weights and bias
weights = np.random.randn(2)
bias = np.random.randn()

# Train and store weights/bias after each epoch
for epoch in range(epochs):
    for i in range(len(inputs)):
        x = inputs[i]
        t = targets[i]
        weighted_sum = np.dot(x, weights) + bias
        y = sigmoid(weighted_sum)
        error = t - y
        weights += learning_rate * error * x * sigmoid_derivative(y)
        bias += learning_rate * error * sigmoid_derivative(y)
    history.append((weights.copy(), bias))

# Set up plot
fig, ax = plt.subplots()
colors = ['red' if t == 0 else 'green' for t in targets]
scatter = ax.scatter(inputs[:, 0], inputs[:, 1], c=colors, s=100)
line, = ax.plot([], [], 'b--')

ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.set_title("Learning NAND - Decision Boundary")

# Update function for animation
def update(frame):
    w, b = history[frame]
    x_vals = np.array(ax.get_xlim())
    if w[1] != 0:
        y_vals = -(w[0] * x_vals + b) / w[1]
    else:
        y_vals = np.full_like(x_vals, -b)
    line.set_data(x_vals, y_vals)
    ax.set_title(f"Epoch {frame}")
    return line,

# Animate
ani = FuncAnimation(fig, update, frames=len(history), interval=100)

# Save as GIF
ani.save("nand_learning.gif", writer=PillowWriter(fps=10))

