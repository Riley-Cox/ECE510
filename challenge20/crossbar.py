import numpy as np
import matplotlib.pyplot as plt

def simulate_crossbar(V_in, R_matrix):
    V_in = np.array(V_in).reshape(-1, 1)              # Shape: M×1
    G_matrix = 1 / np.array(R_matrix, dtype=float)    # Conductance matrix M×N
    I_out_expected = (V_in.T @ G_matrix).flatten()    # Shape: N
    return I_out_expected, G_matrix

def simulate_actual_outputs_mock(I_out_expected, noise_level=0.05):
    # Adds small random noise to expected results (mock real-world/simulated data)
    noise = np.random.normal(0, noise_level * np.abs(I_out_expected))
    return I_out_expected + noise

def compare_and_visualize(V_in, G_matrix, I_expected, I_actual):
    fig, axs = plt.subplots(1, 4, figsize=(20, 4))

    axs[0].bar(range(len(V_in)), V_in, color='skyblue')
    axs[0].set_title("Input Voltages")
    axs[0].set_xlabel("Row Index")
    axs[0].set_ylabel("Voltage (V)")

    im = axs[1].imshow(G_matrix, cmap='viridis', aspect='auto')
    axs[1].set_title("Conductance Matrix (S)")
    axs[1].set_xlabel("Column Index")
    axs[1].set_ylabel("Row Index")
    plt.colorbar(im, ax=axs[1])

    width = 0.35
    x = np.arange(len(I_expected))
    axs[2].bar(x - width/2, I_expected, width, label='Expected', color='lightgreen')
    axs[2].bar(x + width/2, I_actual, width, label='Actual', color='salmon')
    axs[2].set_title("Output Currents Comparison")
    axs[2].set_xlabel("Column Index")
    axs[2].set_ylabel("Current (A)")
    axs[2].legend()

    # Error bar
    error = np.abs(I_expected - I_actual)
    axs[3].bar(x, error, color='orange')
    axs[3].set_title("Absolute Error")
    axs[3].set_xlabel("Column Index")
    axs[3].set_ylabel("Error (A)")

    plt.tight_layout()
    plt.show()

# Example usage
V_in = [0.5, 0.3, 0.7, 1.0]

R_matrix = [
    [1e3, 2e3, 1e3, 4e3],
    [2e3, 2e3, 1e3, 1e3],
    [1e3, 1e3, 2e3, 2e3],
    [4e3, 1e3, 2e3, 1e3],
]

I_expected, G_matrix = simulate_crossbar(V_in, R_matrix)
I_actual = simulate_actual_outputs_mock(I_expected, noise_level=0.02)  # 2% variation

compare_and_visualize(V_in, G_matrix, I_expected, I_actual)

