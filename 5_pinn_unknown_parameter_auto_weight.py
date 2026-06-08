import os
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. Hardware & Reproducibility Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

print(f"\n[SYSTEM] Active Execution Device: {device.type.upper()}")
if device.type == "cuda":
    print(f"[SYSTEM] GPU Name: {torch.cuda.get_device_name(0)}")

# Ensure output directory exists
os.makedirs("graphs", exist_ok=True)

# 2. Define PINN Architecture with Inverse Problem Parameter
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        # Randomly initialized trainable parameter for the damping coefficient
        self.c = nn.Parameter(torch.randn(1) * 5.0) 

    def forward(self, t):
        return self.net(t)

# Initialize model
model = PINN().to(device)

# 3. Setup Optimizer with Explicit Parameter Groups
optimizer = optim.Adam([
    {"params": model.net.parameters(), "lr": 1e-3},
    {"params": [model.c], "lr": 1e-3}  # Dedicated group for physical parameter c
])

# 4. Generate Synthetic Noisy Dataset (t = 0 to 0.5)
num_data_points = 20
t_data_np = np.linspace(0.0, 0.5, num_data_points, dtype=np.float32)

# Analytical function to evaluate target values (True c = 4.0)
def analytical_solution(t_arr):
    return (
        1.0050 
        * np.exp(-2.0 * t_arr) 
        * np.cos(19.900 * t_arr - 0.10017)
    )

# Compute true displacement and add Gaussian noise (std dev = 0.05)
x_true = analytical_solution(t_data_np)
noise = np.random.normal(0, 0.05, size=x_true.shape).astype(np.float32)
x_data_np = x_true + noise

# Convert dataset to PyTorch tensors
t_data = torch.from_numpy(t_data_np).view(-1, 1).to(device)
x_data = torch.from_numpy(x_data_np).view(-1, 1).to(device)

# 5. Domain Tensors for Physics Evaluation (t = 0 to 1)
epochs = 20000
plot_every = 1000

t_physics = torch.linspace(0, 1, 200, dtype=torch.float32).view(-1, 1).to(device)
t_physics.requires_grad = True

# Analytical reference for visual diagnostics
t_eval = torch.linspace(0, 1, 200, dtype=torch.float32).view(-1, 1).to(device)
t_eval_np = t_eval.cpu().flatten().numpy()
x_analytical = analytical_solution(t_eval_np)

# History tracking for parameter estimation
c_history = []
epoch_history = []
lambda_history = []  # Track dynamic weight evolution
true_c = 4.0

# Dynamic Weighting State Variables (EMA)
alpha_ema = 0.9  # Decay factor for weight update smoothing
lambda_val = 1.0  # Initial weight

print("Training Loop Initiated with Dynamic Gradient Weighting")

model.train()
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()

    # --- Step 1: Forward Pass for Data Loss ---
    x_pred_data = model(t_data)
    loss_data = torch.mean((x_pred_data - x_data) ** 2)

    # --- Step 2: Forward Pass for Physics Loss ---
    x_p = model(t_physics)
    dxdt = torch.autograd.grad(
        x_p,
        t_physics,
        grad_outputs=torch.ones_like(x_p),
        create_graph=True,
    )[0]
    d2xdt2 = torch.autograd.grad(
        dxdt,
        t_physics,
        grad_outputs=torch.ones_like(dxdt),
        create_graph=True,
    )[0]

    ode_residual = d2xdt2 + model.c * dxdt + 400 * x_p
    loss_physics = torch.mean(ode_residual**2)

    # --- Step 3: Dynamic Weight Calculation via Functional Autograd ---
    # We directly compute gradients with respect to model.net weights without mutating parameter state (.grad)
    net_params = list(model.net.parameters())
    
    grads_data = torch.autograd.grad(loss_data, net_params, retain_graph=True, allow_unused=True)
    grads_physics = torch.autograd.grad(loss_physics, net_params, retain_graph=True, allow_unused=True)

    # Compute mean absolute gradients of both objectives across all parameter vectors
    # Non-empty gradient checks safeguard against edge cases with disconnected graphs
    if any(g is not None for g in grads_data) and any(g is not None for g in grads_physics):
        mean_grad_data = torch.mean(torch.cat([g.abs().view(-1) for g in grads_data if g is not None])).item()
        mean_grad_physics = torch.mean(torch.cat([g.abs().view(-1) for g in grads_physics if g is not None])).item()
        
        # Calculate target weight (prevent division by zero)
        if mean_grad_data > 1e-8:
            lambda_target = mean_grad_physics / mean_grad_data
            # Apply EMA smoothing to prevent destabilizing oscillation
            lambda_val = alpha_ema * lambda_val + (1.0 - alpha_ema) * lambda_target

    # --- Step 4: Weighted Optimization Step ---
    # Apply detached lambda_val so backpropagation treats it as a constant scale factor
    total_loss = lambda_val * loss_data + loss_physics
    total_loss.backward()
    optimizer.step()

    # Log parameter history
    c_history.append(model.c.item())
    epoch_history.append(epoch)
    lambda_history.append(lambda_val)

    # --- Diagnostics & Visualizations ---
    if epoch % plot_every == 0 or epoch == 1:
        print(
            f"Epoch: {epoch:5d} | "
            f"Total Loss: {total_loss.item():.6e} | "
            f"Data Loss: {loss_data.item():.6e} | "
            f"ODE Res: {loss_physics.item():.6e} | "
            f"Weight (lambda): {lambda_val:.2f} | "
            f"Estimated c: {model.c.item():.4f}"
        )

        model.eval()
        with torch.no_grad():
            x_pinn = model(t_eval).cpu().numpy()
        model.train()

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(
            t_eval_np,
            x_analytical,
            "g-",
            label="Analytical Reference",
            alpha=0.7,
            linewidth=2,
        )
        ax.scatter(
            t_data_np,
            x_data_np,
            color="blue",
            marker="o",
            s=30,
            label="Noisy Training Data (t ∈ [0, 0.5])",
            alpha=0.6,
            zorder=3
        )
        ax.plot(
            t_eval_np,
            x_pinn,
            "r--",
            label=f"PINN (Epoch {epoch})",
            linewidth=2,
        )
        ax.set_title(
            f"PINN vs Analytical (Epoch: {epoch} | Est. c: {model.c.item():.3f} | λ: {lambda_val:.1f})"
        )
        ax.set_xlabel("Time (t)")
        ax.set_ylabel("Displacement (x)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        
        plt.savefig(f"graphs/pinn_data_{epoch}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

# 7. Generate Parameter Convergence Plot (including Weight History)
print("\nGenerating parameter estimation history plot...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot parameter c
ax1.plot(epoch_history, c_history, "b-", linewidth=2, label="Estimated Damping Coefficient (c)")
ax1.axhline(y=true_c, color="r", linestyle="--", linewidth=2, label=f"True Value (c = {true_c})")
ax1.set_title("Damping Coefficient ($c$) Convergence History")
ax1.set_ylabel("Value of $c$")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend()

# Plot weighting history
ax2.plot(epoch_history, lambda_history, "purple", linewidth=2, label="Weight Scale factor ($\lambda$)")
ax2.set_title("Dynamic Weight Scale Factor ($\lambda$) Evolution")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Scale $\lambda$")
ax2.set_yscale("log")  # Log scale since lambda can vary across orders of magnitude
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
parameter_plot_path = "graphs/parameter_estimation_convergence.png"
plt.savefig(parameter_plot_path, dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"Convergence and weight evolution graph saved to: {parameter_plot_path}")
print(f"Total Iterations: {epochs}")
print(f"Final Total Loss: {total_loss.item():.6e}")
print(f"Final Estimated c: {model.c.item():.6f} (True Value: {true_c})")