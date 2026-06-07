#State space formulation with forward euler (1st order integration)
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[0.0,    1.0],
              [-400.0, -4.0]])
dt = 0.0001
total_time = 1
steps = int(total_time // dt) 

# Initial state: [displacement, velocity]
x_prev = np.array([1.0, 0.0])

# --- Pre-allocate Lists for History ---
t_hist = [0.0]
x_hist = [x_prev[0]]  # Save initial displacement

# --- Simulation Loop ---
for i in range(steps):
    # State-space forward step
    x = x_prev + dt * (A @ x_prev)
    
    # Update time and save tracking data
    current_time = (i + 1) * dt
    t_hist.append(current_time)
    x_hist.append(x[0])          # x[0] is displacement
    
    # Critical: March state forward for next iteration
    x_prev = x

# --- Convert History to Arrays ---
t_numerical = np.array(t_hist)
x_numerical = np.array(x_hist)

# --- Compute Analytical Solution ---
# Formula: 1.0050 * e^(-2 * t) * cos(19.900 * t - 0.10017)
x_analytical = 1.0050 * np.exp(-2.0 * t_numerical) * np.cos(19.900 * t_numerical - 0.10017)

# --- Plotting Results ---
plt.figure(figsize=(10, 5))
plt.plot(t_numerical, x_numerical, label="Numerical (Forward Euler SS)", color="crimson", linewidth=2)
plt.plot(t_numerical, x_analytical, label="Analytical (Exact)", color="black", linestyle="--", linewidth=1.5)

plt.title("State-Space Numerical vs. Analytical Solution")
plt.xlabel("Time (seconds)")
plt.ylabel("Displacement")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.show()