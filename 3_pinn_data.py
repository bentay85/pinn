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

# 2. Define PINN Architecture
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

    def forward(self, t):
        return self.net(t)

# Initialize model
model = PINN().to(device)

# 3. Setup Optimizer
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 4. Generate Synthetic Noisy Dataset (t = 0 to 0.5)
num_data_points = 20
t_data_np = np.linspace(0.0, 0.5, num_data_points, dtype=np.float32)

# Analytical function to evaluate target values
def analytical_solution(t_arr):
    # Matches: 1.0050 * exp(-2*t) * cos(19.900 * t - 0.10017)
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
epochs = 15000
plot_every = 1000

t_physics = torch.linspace(0, 1, 200, dtype=torch.float32).view(-1, 1).to(device)
t_physics.requires_grad = True

# Analytical reference for visual diagnostics
t_eval = torch.linspace(0, 1, 200, dtype=torch.float32).view(-1, 1).to(device)
t_eval_np = t_eval.cpu().flatten().numpy()
x_analytical = analytical_solution(t_eval_np)

# 6. Training Loop
print("Training Loop Initiated")

model.train()
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()

    # --- Data Pass (No gradient tracking needed for independent t_data) ---
    x_pred_data = model(t_data)
    loss_data = torch.mean((x_pred_data - x_data) ** 2)

    # --- Physics Domain Pass (Requires higher-order automatic differentiation) ---
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

    # Residual calculation for: d2x/dt2 + 4*dx/dt + 400*x = 0
    ode_residual = d2xdt2 + 4 * dxdt + 400 * x_p
    loss_physics = torch.mean(ode_residual**2)
    
    # The data loss is multiplied by a factor of 5000, this is a hyper parameter that needs to be tuned. 
    total_loss = 5000*loss_data + loss_physics

    # Backpropagation
    total_loss.backward()
    optimizer.step()

    # --- Diagnostics & Visualizations ---
    if epoch % plot_every == 0 or epoch == 1:
        print(
            f"Epoch: {epoch:5d} | "
            f"Total Loss: {total_loss.item():.6e} | "
            f"Data Loss: {loss_data.item():.6e} | "
            f"ODE Res: {loss_physics.item():.6e}"
        )

        # Plot current model predictions against ground truth and noisy points
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
            f"PINN vs Analytical (Epoch: {epoch} | Loss: {total_loss.item():.4e})"
        )
        ax.set_xlabel("Time (t)")
        ax.set_ylabel("Displacement (x)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        
        plt.savefig(f"graphs/pinn_data_{epoch}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

print(f"\nTotal Iterations: {epochs}")
print(f"Final Total Loss: {total_loss.item():.6e}")