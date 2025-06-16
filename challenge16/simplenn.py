import torch
import time

# Model definition (same as before)
class SimpleFeedforwardNN(torch.nn.Module):
    def __init__(self):
        super(SimpleFeedforwardNN, self).__init__()
        self.fc1 = torch.nn.Linear(4, 5)
        self.fc2 = torch.nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model
model = SimpleFeedforwardNN()
model.eval()  # Set model to eval mode for inference

# Optional: move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Create dummy input batch (e.g., 1000 samples, 4 features each)
input_batch = torch.randn(1000, 4).to(device)

# Warm-up (important for fair GPU timing)
for _ in range(10):
    _ = model(input_batch)

# Measure inference time
start = time.time()
with torch.no_grad():
    for _ in range(1000):  # Run multiple forward passes
        _ = model(input_batch)
end = time.time()

total_time = end - start
avg_time_per_pass = total_time / 1000

print(f"Total inference time for 1000 runs: {total_time:.6f} seconds")
print(f"Average time per inference: {avg_time_per_pass * 1e3:.6f} ms")
