import numpy as np 

def save_mem(data, filename):
  with open(filename, 'w') as f:
    flat = data.flatten()
    for val in flat:
      f.write(f"{val & 0xFF:02x}\n")


image = np.load("image_input.npy")
kernel1 = np.load("kernel1.npy")
bias1 = np.load("bias1.npy")
kernel2 = np.load("kernel2.npy")
bias2 = np.load("bias2.npy")
kernel3 = np.load("kernel3.npy")
bias3 = np.load("bias3.npy")

save_mem(image, "image_input.mem")
save_mem(kernel1, "kernel1.mem")
save_mem(bias1, "bias1.mem")
save_mem(kernel2, "kernel2.mem")
save_mem(bias2, "bias2.mem")
save_mem(kernel3, "kernel3.mem")
save_mem(bias3, "bias3.mem")
