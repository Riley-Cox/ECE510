import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Step activation function
def step_activation(x):
    return 1 if x >= 0 else 0

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.randn(input_size)  # Initialize weights randomly
        self.bias = np.random.randn()  # Initialize bias randomly
        self.lr = learning_rate
        self.snapshots = []  # Store snapshots of decision boundary during training

    def forward(self, x):
        # Weighted sum + bias
        z = np.dot(x, self.weights) + self.bias
        return step_activation(z)

    def update(self, X, y):
        total_error = 0
        for i in range(len(X)):
            x_i = X[i]
            y_i = y[i]

            # Prediction
            prediction = self.forward(x_i)
            error = y_i - prediction

            # Perceptron update rule
            self.weights += self.lr * error * x_i
            self.bias += self.lr * error

            total_error += error ** 2  # Track total error (for logging)
        return total_error

    def train(self, X, y, epochs=1000, snapshot_interval=50):
        for epoch in range(epochs):
            total_error = self.update(X, y)

            # Capture decision boundary snapshot
            if epoch % snapshot_interval == 0:
                self.snapshots.append(self.create_decision_boundary(X))

            if epoch % 100 == 0:  # Print progress every 100 epochs
                print(f"Epoch {epoch} - Total Error: {total_error}")

    def predict(self, x):
        return self.forward(x)

    def create_decision_boundary(self, X):
        # Create a meshgrid for decision boundary visualization
        h = 0.01
        x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
        y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Evaluate predictions over grid
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = np.array([self.forward(p) for p in grid_points])
        Z = Z.reshape(xx.shape)
        return xx, yy, Z

# NAND logic gate truth table
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_nand = np.array([1, 1, 1, 0])  # NAND output

# Initialize perceptron
perceptron = Perceptron(input_size=2, learning_rate=0.1)

# Train the perceptron and capture snapshots of decision boundary
perceptron.train(X, y_nand, epochs=1000, snapshot_interval=50)

# Set up the plot for animation
fig, ax = plt.subplots(figsize=(6, 5))
title = ax.set_title("Learning NAND with Perceptron")

def update(frame):
    ax.clear()
    xx, yy, Z = perceptron.snapshots[frame]
    contour = ax.contourf(xx, yy, Z, levels=50, cmap="coolwarm", alpha=0.8)

    # Plot training points
    for i, point in enumerate(X):
        ax.scatter(*point, c='k' if y_nand[i] == 0 else 'w', edgecolors='k', s=100)

    ax.set_title(f"Epoch {frame * 50}")

# Create animation
ani = FuncAnimation(fig, update, frames=len(perceptron.snapshots), interval=50)

# Save the animation as a .gif
ani.save('nand_perceptron_learning.gif', writer='imagemagick', fps=20)

plt.show()

