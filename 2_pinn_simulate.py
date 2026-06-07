import os
import matplotlib
# Use non-interactive backend to prevent GUI initialization and rendering overhead
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Hardware & Reproducibility Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# Print execution device immediately to verify hardware acceleration
print(f"\n[SYSTEM] Active Execution Device: {device.type.upper()}")
if device.type == "cuda":
    print(f"[SYSTEM] GPU Name: {torch.cuda.get_device_name(0)}")
print("-" * 65)

# Ensure the output directory for plots exists
os.makedirs("graphs", exist_ok=True)

# 2. Define PINN Architecture
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        # 50 neurons per layer provides adequate capacity for stiff ODE landscapes
        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

    def forward(self, t):
        return self.net(t)

# Initialize model
model = PINN().to(device)

# 3. Setup Adam Optimizer
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 4. Setup Training and Domain Tensors
epochs = 15000  # Adam requires more iterations than L-BFGS to converge on stiff landscapes
plot_every = 1000

# Boundary condition anchors at t=0
t_zero = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True).to(device)

# Unlabeled time points spanning the domain for physics evaluation
t_physics = torch.linspace(0, 1, 200, dtype=torch.float32).view(-1, 1).to(device)
t_physics.requires_grad = True

# --- PRECOMPUTE ANALYTICAL REFERENCE FOR IN-LOOP PLOTTING ---
t_eval = torch.linspace(0, 1, 200, dtype=torch.float32).view(-1, 1).to(device)
t_eval_np = t_eval.cpu().flatten().numpy()
x_analytical = (
    (
        1.0050
        * torch.exp(-2 * t_eval.cpu().flatten())
        * torch.cos(19.900 * t_eval.cpu().flatten() - 0.10017)
    )
    .numpy()
)

# 5. Training Loop
print(f"Optimizing stiff landscape via Adam Engine...\n{'='*65}")

model.train()
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()

    # --- Boundary Tracking Pass ---
    x_zero = model(t_zero)
    dx_dt_zero = torch.autograd.grad(
        x_zero,
        t_zero,
        grad_outputs=torch.ones_like(x_zero),
        create_graph=True,
    )[0]

    # --- Physics Domain Pass ---
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

    # --- Compute Weighted Losses ---
    loss_ic1 = (x_zero - 1.0).pow(2).sum()
    loss_ic2 = dx_dt_zero.pow(2).sum()

    # Residual calculation for: d2x/dt2 + 4*dx/dt + 400*x = 0
    ode_residual = d2xdt2 + 4 * dxdt + 400 * x_p
    loss_physics = torch.mean(ode_residual**2)

    total_loss = 5000 * (loss_ic1 + loss_ic2) + loss_physics

    # Backpropagation
    total_loss.backward()
    optimizer.step()

    # --- Diagnostics & Visualizations ---
    if epoch % plot_every == 0 or epoch == 1:
        print(
            f"Epoch: {epoch:5d} | "
            f"Total Loss: {total_loss.item():.6e} | "
            f"ODE Res: {loss_physics.item():.6e} | "
            f"IC Loss: {(loss_ic1+loss_ic2).item():.6e}"
        )

        # Plot current state vs analytical reference
        model.eval()
        with torch.no_grad():
            x_pinn = model(t_eval).cpu().numpy()
        model.train()

        # Generate a headless figure context
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(
            t_eval_np,
            x_analytical,
            "g-",
            label="Analytical Reference",
            alpha=0.7,
            linewidth=2,
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
        
        # Save frame directly to disk
        plt.savefig(f"graphs/pinn_simulate_{epoch}.png", dpi=150, bbox_inches='tight')
        
        # Free memory associated with the figure context
        plt.close(fig)

print(f"\nTotal Iterations: {epochs}")
print(f"Final Total Loss: {total_loss.item():.6e}\n{'='*65}")