import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV
df = pd.read_csv('timing.csv')

# Compute exponent for labels
df['exp'] = np.log2(df['N']).astype(int)

# Bar plot
bar_width = 0.35
x = np.arange(len(df))

plt.figure(figsize=(10, 6))

# Total time bars
plt.bar(x - bar_width/2, df['TotalTime_ms'], bar_width, label='Total Time (ms)', color='skyblue')

# Kernel-only time bars
plt.bar(x + bar_width/2, df['KernelTime_ms'], bar_width, label='Kernel Time (ms)', color='orange')

# Labels and ticks
plt.xlabel('Matrix Size (N = 2^exp)')
plt.ylabel('Time (ms, log scale)')
plt.title('CUDA SAXPY: Total Time vs Kernel Time (Log Scale)')
plt.xticks(x, df['exp'])
plt.yscale('log')  # Log scale here
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7, which='both')

plt.tight_layout()
plt.show()

