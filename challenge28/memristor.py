import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# === Memristor Parameters (HP + Biolek) ===
R_on = 100       # Ohms
R_off = 16000    # Ohms
D = 10e-9        # Memristor thickness (meters)
mu_v = 1e-14     # Ion mobility (m^2/s/V)
p = 2            # Biolek window function exponent

# === Input Voltage: Low Frequency Sinusoid ===
V_amp = 1.0       # Voltage amplitude (V)
freq = 1.0        # Frequency in Hz
omega = 2 * np.pi * freq

# === Time Range ===
t = np.linspace(0, 4, 2000)  # 4 seconds

def voltage(t):
    """Sinusoidal driving voltage."""
    return V_amp * np.sin(omega * t)

def biolek_window(w, i):
    """Biolek nonlinear window function."""
    w_norm = w / D
    if i >= 0:
        return 1 - (2*w_norm - 1)**(2 * p)
    else:
        return 1 - (2*w_norm - 1)**(2 * p)

def memristor_model(w, t):
    """ODE for memristor state variable using Biolek window."""
    v = voltage(t)
    R = R_on * (w/D) + R_off * (1 - w/D)
    i = v / R
    dw_dt = mu_v * R_on * i / D * biolek_window(w, i)
    # Clamp w to [0, D] to avoid numerical issues
    if w <= 0 and dw_dt < 0:
        return 0
    elif w >= D and dw_dt > 0:
        return 0
    return dw_dt

# === Initial State ===
w0 = D / 2

# === Solve ODE ===
w = odeint(memristor_model, w0, t).flatten()

# === Compute Voltage, Resistance, Current ===
v_t = voltage(t)
R_t = R_on * (w/D) + R_off * (1 - w/D)
i_t = v_t / R_t

# === Plot and Save I–V Curve ===
plt.figure(figsize=(8, 6))
plt.plot(v_t, i_t, label="I–V curve")
plt.title("Memristor Pinched Hysteresis Loop (Biolek Window)")
plt.xlabel("Voltage (V)")
plt.ylabel("Current (A)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("memristor_iv_curve.png", dpi=300)
plt.show()

