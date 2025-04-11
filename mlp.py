import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# MLP Class
class MLP_XOR:
    def __init__(self, learning_rate=0.1):
        self.lr = learning_rate
        self.hidden_size = 2

        # Init weights and biases
        self.W1 = np.random.randn(self.hidden_size, 2)
        self.b1 = np.random.randn(self.hidden_size)
        self.W2 = np.random.randn(1, self.hidden_size)
        self.b2 = np.random.randn(1)

    def forward(self, x):
        self.z1 = np.dot(self.W1, x) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.W2, self.a1) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, x, y):
        error = y - self.a2
        d_output = error * sigmoid_derivative(self.z2)

        error_hidden = np.dot(self.W2.T, d_output)
        d_hidden = error_hidden * sigmoid_derivative(self.z1)

        self.W2 += self.lr * np.outer(d_output, self.a1)
        self.b2 += self.lr * d_output
        self.W1 += self.lr * np.outer(d_hidden, x)
        self.b1 += self.lr * d_hidden

    def train_with_snapshots(self, X, y, epochs=10000, snapshot_interval=100):
        snapshots = []
        for epoch in range(epochs):
            for i in range(len(X)):
                x_i = X[i]
                y_i = y[i]
                self.forward(x_i)
                self.backward(x_i, y_i)
            if epoch % snapshot_interval == 0:
                snapshots.append(self.snapshot_decision_boundary(X))
        return snapshots

    def snapshot_decision_boundary(self, X):
        h = 0.01
        x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
        y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = np.array([self.forward(p)[0] for p in grid_points])
        Z = Z.reshape(xx.shape)
        return xx, yy, Z

    def predict(self, x):
        return self.forward(x)

# Logic gates data
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# NAND training
print("Training on NAND")
y_nand = np.array([
    [1],
    [1],
    [1],
    [0]
])
mlp_nand = MLP_XOR(learning_rate=0.5)
snapshots_nand = mlp_nand.train_with_snapshots(X, y_nand, epochs=5000, snapshot_interval=100)

fig1, ax1 = plt.subplots(figsize=(6, 5))
def update_nand(frame):
    ax1.clear()
    xx, yy, Z = snapshots_nand[frame]
    ax1.contourf(xx, yy, Z, levels=50, cmap="coolwarm", alpha=0.8)
    for i, point in enumerate(X):
        ax1.scatter(*point, c='k' if y_nand[i] == 0 else 'w', edgecolors='k', s=100)
    ax1.set_title(f"NAND - Epoch {frame * 100}")
ani_nand = FuncAnimation(fig1, update_nand, frames=len(snapshots_nand), interval=100)

# XOR training
print("Training on XOR")
y_xor = np.array([
    [0],
    [1],
    [1],
    [0]
])
mlp_xor = MLP_XOR(learning_rate=0.5)
snapshots_xor = mlp_xor.train_with_snapshots(X, y_xor, epochs=5000, snapshot_interval=100)

fig2, ax2 = plt.subplots(figsize=(6, 5))
def update_xor(frame):
    ax2.clear()
    xx, yy, Z = snapshots_xor[frame]
    ax2.contourf(xx, yy, Z, levels=50, cmap="coolwarm", alpha=0.8)
    for i, point in enumerate(X):
        ax2.scatter(*point, c='k' if y_xor[i] == 0 else 'w', edgecolors='k', s=100)
    ax2.set_title(f"XOR - Epoch {frame * 100}")
ani_xor = FuncAnimation(fig2, update_xor, frames=len(snapshots_xor), interval=100)

ani_nand.save("mlp_nand_training.gif",writer=PillowWriter(fps=10))
print("Saved as mlp_nand_training.gif")
ani_xor.save("mlp_xor_training.gif",writer=PillowWriter(fps=10))
print("Saved as mlp_xor_training.gif")

plt.show()
