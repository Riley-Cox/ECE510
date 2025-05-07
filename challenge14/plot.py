import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read times.csv
df = pd.read_csv('times.csv')
print(df)

# Plot CPU vs GPU times (kernel and total)
plt.figure(figsize=(10,6))
plt.plot(df['N'], df['CPU_ms'], label='CPU (Sequential)', marker='o')
plt.plot(df['N'], df['GPU_kernel_ms'], label='GPU Kernel Only', marker='o')
plt.plot(df['N'], df['GPU_memcpy_ms'], label='GPU Memory Copy', marker='o')
plt.plot(df['N'], df['GPU_total_ms'], label='GPU Total (Kernel + Copy)', marker='o')
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('N (number of Fibonacci numbers)')
plt.ylabel('Time (ms)')
plt.title('CPU vs GPU Fibonacci Execution Time (Detailed)')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()

# Plot speedup (CPU time / GPU total time)
plt.figure(figsize=(8,5))
speedup = df['CPU_ms'] / df['GPU_total_ms']
plt.plot(df['N'], speedup, label='Speedup (CPU / GPU Total)', marker='o', color='green')
plt.xscale('log', base=2)
plt.xlabel('N (number of Fibonacci numbers)')
plt.ylabel('Speedup Factor')
plt.title('GPU Speedup over CPU')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()

