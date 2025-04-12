import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Sigmoid activation and derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Single Neuron Model for NAND
def train_sigmoid_nand(X, y, lr=0.1, epochs=1000, snapshot_interval=10):
    np.random.seed(0)
    weights = np.random.randn(2)
    bias = np.random.randn(1)
    snapshots = []

    for epoch in range(epochs):
        for i in range(len(X)):
            x_i = X[i]
            y_i = y[i]
            z = np.dot(weights, x_i) + bias
            output = sigmoid(z)
            error = y_i - output
            grad = error * sigmoid_derivative(z)
            weights += lr * grad * x_i
            bias += lr * grad

        if epoch % snapshot_interval == 0:
            snapshots.append((weights.copy(), bias.copy(), epoch))

    return weights, bias, snapshots

# NAND data
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([1, 1, 1, 0])

# Train
weights, bias, snapshots = train_sigmoid_nand(X, y, lr=0.5, epochs=1000, snapshot_interval=10)

# Animate decision boundary
fig, ax = plt.subplots(figsize=(6, 5))
xx, yy = np.meshgrid(np.linspace(-0.2, 1.2, 200), np.linspace(-0.2, 1.2, 200))
grids = np.c_[xx.ravel(), yy.ravel()]

sc = None

def update(i):
    ax.clear()
    w, b, epoch = snapshots[i]
    zz = sigmoid(np.dot(grids, w) + b)
    zz = zz.reshape(xx.shape)
    contour = ax.contourf(xx, yy, zz, levels=50, cmap="coolwarm", alpha=0.8)
    for j, point in enumerate(X):
        ax.scatter(*point, c='k' if y[j] == 0 else 'w', edgecolors='k', s=100)
    ax.set_title(f"Epoch {epoch}")
    return contour

ani = FuncAnimation(fig, update, frames=len(snapshots), interval=30)
ani.save("sigmoid_nand.gif", writer=PillowWriter(fps=30))

plt.show()

