import matplotlib.pyplot as plt
import pandas as pd
import math

# Load CSV
data = pd.read_csv('timing.csv')

# Plot
plt.figure(figsize=(10,6))
plt.bar(data['N'].astype(str), data['Time_ms'], color='skyblue')
plt.xlabel('Matrix Size (N)')
plt.ylabel('Execution Time (ms)')
plt.title('SAXPY Execution Time vs Matrix Size')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
