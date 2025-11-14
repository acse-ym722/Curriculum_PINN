import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import copy
import loss_landscapes
import loss_landscapes.metrics

# --- 0. Set Random Seeds for Reproducibility ---
def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

set_seed(42)

# --- 1. Hyperparameters & Configuration ---
LEARNING_RATE = 0.005
BATCH_SIZE = 64
TOTAL_EPOCHS = 20000

# Curriculum Learning Stage Definitions
STAGE1_END = 2000
STAGE2_END = 5000

# Trajectory tracking configuration
TRACK_INTERVAL = 100

# Check for available GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# --- 2. Target Function Definitions ---
def f_simple(x):
    """Simple function: a low-frequency sine wave."""
    return np.sin(2 * np.pi * x)

def f_complex(x):
    """Complex function: a sum of low and high-frequency waves."""
    return np.sin(2 * np.pi * x) + 0.5 * np.sin(20 * np.pi * x)


# --- 3. Neural Network Definition ---
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)

# --- 4. Helper function to get flattened parameters ---
def get_flat_params(model):
    """Get model parameters as a flat numpy array."""
    params = []
    for param in model.parameters():
        params.append(param.data.cpu().numpy().flatten())
    return np.concatenate(params)

# --- 5. Loss Landscape Visualization Function ---
def plot_loss_landscape(model, criterion, data_x, data_y, ax, title, 
                       trajectory_points=None, show_3d=False):
    """
    Generates and plots the loss landscape for a given model.
    """
    print(f"\n--- Generating Loss Landscape for: {title} ---")
    
    model.cpu()
    data_x, data_y = data_x.cpu(), data_y.cpu()
    
    metric = loss_landscapes.metrics.Loss(
        criterion,
        data_x,
        data_y
    )

    STEPS = 40
    
    print(f"Calculating loss landscape with {STEPS}x{STEPS} grid... (This may take a while)")
    
    loss_data = loss_landscapes.random_plane(
        model=model,
        metric=metric,
        distance=1.0,
        steps=STEPS,
        normalization='filter',
        deepcopy_model=True
    )
    
    print("✓ Calculation complete.")

    loss_values = loss_data.copy()
    loss_values[loss_values <= 0] = 1e-10
    
    x_coords = np.linspace(-1.0, 1.0, STEPS)
    y_coords = np.linspace(-1.0, 1.0, STEPS)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    if show_3d:
        surf = ax.plot_surface(X, Y, np.log(loss_values), cmap='viridis', 
                              alpha=0.8, edgecolor='none')
        ax.set_zlabel('Log(Loss)', fontsize=12)
        ax.view_init(elev=25, azim=45)
    else:
        contour = ax.contourf(X, Y, np.log(loss_values), levels=50, cmap='viridis')
        plt.colorbar(contour, ax=ax, label='Log(Loss)')
    
    if show_3d:
        z_center = np.log(metric(model))
        ax.scatter([0], [0], [z_center], c='red', s=200, marker='X', 
                  edgecolors='darkred', linewidths=2, label='Final Model', zorder=10)
    else:
        ax.plot([0], [0], 'rX', markersize=12, markeredgewidth=2, 
               label='Final Model Parameters')
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Direction α', fontsize=12)
    ax.set_ylabel('Direction β', fontsize=12)
    ax.legend()
    
    if not show_3d:
        ax.set_aspect('equal', adjustable='box')

# --- 6. Function to plot training trajectory ---
def plot_trajectory_2d(ax, trajectory_coords, title, color='blue'):
    """
    Plot 2D training trajectory on the loss landscape.
    """
    if len(trajectory_coords) < 2:
        return
    
    trajectory_coords = np.array(trajectory_coords)
    
    ax.plot(trajectory_coords[:, 0], trajectory_coords[:, 1], 
           color=color, linewidth=2, alpha=0.7, label='Training Path', zorder=5)
    
    ax.scatter(trajectory_coords[0, 0], trajectory_coords[0, 1], 
              c='green', s=150, marker='^', edgecolors='darkgreen', 
              linewidths=2, label='Start', zorder=10)
    ax.scatter(trajectory_coords[-1, 0], trajectory_coords[-1, 1], 
              c='red', s=150, marker='v', edgecolors='darkred', 
              linewidths=2, label='End', zorder=10)
    
    n_arrows = min(5, len(trajectory_coords) - 1)
    arrow_indices = np.linspace(0, len(trajectory_coords) - 2, n_arrows, dtype=int)
    
    for idx in arrow_indices:
        dx = trajectory_coords[idx + 1, 0] - trajectory_coords[idx, 0]
        dy = trajectory_coords[idx + 1, 1] - trajectory_coords[idx, 1]
        ax.arrow(trajectory_coords[idx, 0], trajectory_coords[idx, 1], 
                dx * 0.3, dy * 0.3, head_width=0.05, head_length=0.05, 
                fc=color, ec=color, alpha=0.6, zorder=8)

# --- 7. Project trajectory onto 2D plane ---
def project_trajectory_to_plane(trajectory_states, final_model, direction1, direction2):
    """
    Project the parameter trajectory onto a 2D plane defined by two directions.
    """
    final_params = get_flat_params(final_model)
    coords = []
    
    for state in trajectory_states:
        temp_model = NeuralNetwork()
        temp_model.load_state_dict(state)
        temp_params = get_flat_params(temp_model)
        
        displacement = temp_params - final_params
        
        x = np.dot(displacement, direction1) / np.linalg.norm(direction1)**2
        y = np.dot(displacement, direction2) / np.linalg.norm(direction2)**2
        
        coords.append([x, y])
    
    return coords

# --- 8. Generate random orthonormal directions ---
def generate_random_directions(model):
    """Generate two random orthonormal directions in parameter space."""
    params = []
    for param in model.parameters():
        params.append(param.data.cpu().numpy().flatten())
    params_flat = np.concatenate(params)
    
    n_params = len(params_flat)
    
    direction1 = np.random.randn(n_params)
    direction2 = np.random.randn(n_params)
    
    direction1 = direction1 / np.linalg.norm(direction1)
    direction2 = direction2 - np.dot(direction2, direction1) * direction1
    direction2 = direction2 / np.linalg.norm(direction2)
    
    return direction1, direction2

# --- Model and Optimizer Initialization ---
cl_model = NeuralNetwork().to(DEVICE)
dt_model = NeuralNetwork().to(DEVICE)

dt_model.load_state_dict(copy.deepcopy(cl_model.state_dict()))

initial_state = copy.deepcopy(cl_model.state_dict())

criterion = nn.MSELoss()

cl_optimizer = optim.Adam(cl_model.parameters(), lr=LEARNING_RATE)
dt_optimizer = optim.Adam(dt_model.parameters(), lr=LEARNING_RATE)

# --- Initialize trajectory tracking ---
cl_trajectory_states = [copy.deepcopy(initial_state)]
dt_trajectory_states = [copy.deepcopy(initial_state)]

# ✨ NEW: Track stage-specific models
stage1_model_state = None  # Will store model at end of Stage 1
stage2_model_state = None  # Will store model at end of Stage 2 (transition)

# --- Training Loop ---
cl_losses = []
dt_losses = []

for epoch in tqdm(range(TOTAL_EPOCHS), desc="Training Progress"):
    # --- Curriculum Learning (CL) Training Step ---
    cl_model.train()
    
    if epoch < STAGE1_END:
        alpha = 0.0
    elif epoch < STAGE2_END:
        alpha = (epoch - STAGE1_END) / (STAGE2_END - STAGE1_END)
    else:
        alpha = 1.0

    x_train = torch.rand(BATCH_SIZE, 1).to(DEVICE)
    
    y_simple_tensor = torch.tensor(f_simple(x_train.cpu().numpy()), dtype=torch.float32).to(DEVICE)
    y_complex_tensor = torch.tensor(f_complex(x_train.cpu().numpy()), dtype=torch.float32).to(DEVICE)
    y_true_cl = (1 - alpha) * y_simple_tensor + alpha * y_complex_tensor

    cl_optimizer.zero_grad()
    cl_output = cl_model(x_train)
    cl_loss = criterion(cl_output, y_true_cl)
    cl_loss.backward()
    cl_optimizer.step()
    
    cl_losses.append(cl_loss.item())

    # --- Direct Training (DT) Training Step ---
    dt_model.train()
    
    y_true_dt = torch.tensor(f_complex(x_train.cpu().numpy()), dtype=torch.float32).to(DEVICE)

    dt_optimizer.zero_grad()
    dt_output = dt_model(x_train)
    dt_loss = criterion(dt_output, y_true_dt)
    dt_loss.backward()
    dt_optimizer.step()
    
    dt_losses.append(dt_loss.item())
    
    # --- Track trajectory at intervals ---
    if (epoch + 1) % TRACK_INTERVAL == 0:
        cl_trajectory_states.append(copy.deepcopy(cl_model.state_dict()))
        dt_trajectory_states.append(copy.deepcopy(dt_model.state_dict()))
    
    # ✨ NEW: Save models at stage boundaries
    if epoch == STAGE1_END - 1:
        stage1_model_state = copy.deepcopy(cl_model.state_dict())
        print(f"\n✓ Saved Stage 1 model at epoch {epoch + 1}")
    
    if epoch == STAGE2_END - 1:
        stage2_model_state = copy.deepcopy(cl_model.state_dict())
        print(f"\n✓ Saved Stage 2 (transition end) model at epoch {epoch + 1}")

# Add final states
cl_trajectory_states.append(copy.deepcopy(cl_model.state_dict()))
dt_trajectory_states.append(copy.deepcopy(dt_model.state_dict()))

print(f"\nTracked {len(cl_trajectory_states)} points along the training trajectory.")

# --- Visualization Part A: Original Plots ---
x_plot = np.linspace(0, 1, 400)
x_plot_tensor = torch.tensor(x_plot, dtype=torch.float32).view(-1, 1).to(DEVICE)

cl_model.eval()
dt_model.eval()

with torch.no_grad():
    y_pred_cl = cl_model(x_plot_tensor).cpu().numpy()
    y_pred_dt = dt_model(x_plot_tensor).cpu().numpy()

y_true_complex = f_complex(x_plot)

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig1.suptitle("Training Results Analysis", fontsize=20)

ax1.plot(x_plot, y_true_complex, 'k-', linewidth=3, label='True Complex Function')
ax1.plot(x_plot, y_pred_cl, 'g-', linewidth=2.5, label='Curriculum Learning (CL)')
ax1.plot(x_plot, y_pred_dt, 'r--', linewidth=2.5, label='Direct Training (DT)')
ax1.set_title('Model Prediction vs. True Function', fontsize=16)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.legend(fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_ylim(-1.7, 1.7)

epochs_range = range(TOTAL_EPOCHS)
ax2.plot(epochs_range, cl_losses, 'g-', linewidth=1.5, label='Curriculum Learning (CL)', alpha=0.9)
ax2.plot(epochs_range, dt_losses, 'r-', linewidth=1.5, label='Direct Training (DT)', alpha=0.7)
ax2.set_yscale('log')
ax2.set_title('Training Loss Comparison (Log Scale)', fontsize=16)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Mean Squared Error (MSE Loss)', fontsize=12)
ax2.grid(True, which="both", linestyle='--', alpha=0.6)
ax2.axvline(x=STAGE1_END, color='blue', linestyle=':', linewidth=2, label='End of Stage 1')
ax2.axvline(x=STAGE2_END, color='purple', linestyle=':', linewidth=2, label='End of Transition')
ax2.legend(fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
plt.show()

# --- ✨ NEW: Part B - Stage-wise Loss Landscape Comparison (2D) ---
N_LANDSCAPE_POINTS = 1000
x_landscape = torch.linspace(0, 1, N_LANDSCAPE_POINTS).view(-1, 1)
y_landscape_complex = torch.tensor(f_complex(x_landscape.numpy()), dtype=torch.float32)

# Generate random directions for projection
set_seed(42)
cl_model.cpu()
dt_model.cpu()
direction1, direction2 = generate_random_directions(cl_model)

# Create stage models
stage1_model = NeuralNetwork()
stage1_model.load_state_dict(stage1_model_state)

stage2_model = NeuralNetwork()
stage2_model.load_state_dict(stage2_model_state)

# Create figure for stage-wise comparison (2D)
fig2, axes = plt.subplots(1, 3, figsize=(24, 7))
fig2.suptitle("Loss Landscape Evolution: CL Stages vs DT (2D)", fontsize=20)

# Plot Stage 1 (Simple function training)
print("\n=== Stage 1: Training on Simple Function ===")
plot_loss_landscape(
    model=copy.deepcopy(stage1_model),
    criterion=criterion,
    data_x=x_landscape,
    data_y=y_landscape_complex,
    ax=axes[0],
    title='CL Stage 1: Simple Function\n(Epoch 0-2000, α=0.0)',
    show_3d=False
)

# Plot Stage 2 (Transition completed)
print("\n=== Stage 2: End of Transition ===")
plot_loss_landscape(
    model=copy.deepcopy(stage2_model),
    criterion=criterion,
    data_x=x_landscape,
    data_y=y_landscape_complex,
    ax=axes[1],
    title='CL Stage 2: Transition End\n(Epoch 2000-5000, α→1.0)',
    show_3d=False
)

# Plot Direct Training
print("\n=== Direct Training (Full Epochs) ===")
plot_loss_landscape(
    model=copy.deepcopy(dt_model),
    criterion=criterion,
    data_x=x_landscape,
    data_y=y_landscape_complex,
    ax=axes[2],
    title='Direct Training\n(Epoch 0-20000, Complex)',
    show_3d=False
)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('loss_landscape_stages_2d.png', dpi=300, bbox_inches='tight')
plt.show()

# --- ✨ NEW: Part C - Stage-wise Loss Landscape Comparison (3D) ---
fig3 = plt.figure(figsize=(24, 7))
fig3.suptitle("Loss Landscape Evolution: CL Stages vs DT (3D)", fontsize=20)

ax1_3d = fig3.add_subplot(131, projection='3d')
ax2_3d = fig3.add_subplot(132, projection='3d')
ax3_3d = fig3.add_subplot(133, projection='3d')

print("\n=== Generating 3D Landscapes ===")

plot_loss_landscape(
    model=copy.deepcopy(stage1_model),
    criterion=criterion,
    data_x=x_landscape,
    data_y=y_landscape_complex,
    ax=ax1_3d,
    title='CL Stage 1: Simple Function\n(Epoch 0-2000)',
    show_3d=True
)

plot_loss_landscape(
    model=copy.deepcopy(stage2_model),
    criterion=criterion,
    data_x=x_landscape,
    data_y=y_landscape_complex,
    ax=ax2_3d,
    title='CL Stage 2: Transition End\n(Epoch 2000-5000)',
    show_3d=True
)

plot_loss_landscape(
    model=copy.deepcopy(dt_model),
    criterion=criterion,
    data_x=x_landscape,
    data_y=y_landscape_complex,
    ax=ax3_3d,
    title='Direct Training\n(Epoch 0-20000)',
    show_3d=True
)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('loss_landscape_stages_3d.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Part D: Original trajectory visualization ---
print("\nProjecting CL trajectory onto 2D plane...")
cl_trajectory_2d = project_trajectory_to_plane(cl_trajectory_states, cl_model, direction1, direction2)

print("Projecting DT trajectory onto 2D plane...")
dt_trajectory_2d = project_trajectory_to_plane(dt_trajectory_states, dt_model, direction1, direction2)

fig4, (ax_cl, ax_dt) = plt.subplots(1, 2, figsize=(20, 8))
fig4.suptitle("Final Loss Landscape with Training Trajectories (2D)", fontsize=20)

plot_loss_landscape(
    model=copy.deepcopy(cl_model),
    criterion=criterion,
    data_x=x_landscape,
    data_y=y_landscape_complex,
    ax=ax_cl,
    title='Curriculum Learning Model (Final)',
    show_3d=False
)
plot_trajectory_2d(ax_cl, cl_trajectory_2d, 'Curriculum Learning', color='yellow')

plot_loss_landscape(
    model=copy.deepcopy(dt_model),
    criterion=criterion,
    data_x=x_landscape,
    data_y=y_landscape_complex,
    ax=ax_dt,
    title='Direct Training Model (Final)',
    show_3d=False
)
plot_trajectory_2d(ax_dt, dt_trajectory_2d, 'Direct Training', color='cyan')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('loss_landscape_2d_trajectory.png', dpi=300, bbox_inches='tight')
plt.show()

# Move models back to original device
cl_model.to(DEVICE)
dt_model.to(DEVICE)

# --- Final Output ---
print(f"\n{'='*80}")
print(f"TRAINING COMPLETE!")
print(f"{'='*80}")
print(f"\n📊 Final Losses:")
print(f"  • Curriculum Learning (CL): {cl_losses[-1]:.6f}")
print(f"  • Direct Training (DT): {dt_losses[-1]:.6f}")
print(f"\n📁 Visualization files saved:")
print(f"  • training_results.png")
print(f"  • loss_landscape_stages_2d.png  ← NEW: Stage-wise comparison (2D)")
print(f"  • loss_landscape_stages_3d.png  ← NEW: Stage-wise comparison (3D)")
print(f"  • loss_landscape_2d_trajectory.png")
print(f"\n🎯 Stage Information:")
print(f"  • Stage 1 (Simple): Epochs 0-{STAGE1_END}")
print(f"  • Stage 2 (Transition): Epochs {STAGE1_END}-{STAGE2_END}")
print(f"  • Stage 3 (Complex): Epochs {STAGE2_END}-{TOTAL_EPOCHS}")
print(f"{'='*80}\n")