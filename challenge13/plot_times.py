import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV
data = pd.read_csv('execution_times.csv')

# Plot
plt.figure(figsize=(10,6))
plt.bar(data['MatrixSize'], data['ExecutionTime_ms'], color='skyblue')
plt.xlabel('Matrix Size (power of 2)')
plt.ylabel('Execution Time (ms)')
plt.title('SAXPY Execution Time vs Matrix Size')
plt.xticks(data['MatrixSize'])  # show exact powers
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

