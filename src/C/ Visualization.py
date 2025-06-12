import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

# === Settings ===
box_size = 1.0
dp = "ngp"
solver = "periodic"
N_grid = 128

# === File paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
potential_csv = os.path.join(BASE_DIR, f"potential_N{N_grid}.csv")
pos_csv = os.path.join(BASE_DIR, "particle_positions.csv")
log_csv = os.path.join(BASE_DIR, "simulation_output.csv")

# === Load particle positions ===
pos_data = pd.read_csv(pos_csv)
steps = pos_data['step'].unique()
N_particles = pos_data['id'].nunique()

particle_frames = []
for step in steps:
    frame = pos_data[pos_data['step'] == step]
    xy = frame[['x', 'y']].to_numpy()
    particle_frames.append(xy)

# === Load potential data ===
# Each row is a flattened 2D (N_grid x N_grid) potential slice
pot_data = pd.read_csv(potential_csv, sep=r'\s+', header=None, engine="python").values
pot_frames = [pot_data[i].reshape(N_grid, N_grid) for i in range(pot_data.shape[0])]

# === Combined Animation: Particles + Potential ===
def PP_anim(frames, particle_frames, box_size, dp, solver):
    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], extent=[0, box_size, 0, box_size], origin='lower',
                   cmap='viridis', vmin=np.min(frames), vmax=np.max(frames))
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Gravitational Potential")

    scatter = ax.scatter(particle_frames[0][:, 0], particle_frames[0][:, 1], s=1, color='white')

    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = ax.set_title("Particles + Potential Frame 0")

    def update(i):
        im.set_data(frames[i])
        scatter.set_offsets(particle_frames[i])
        title.set_text(f"Particles + Potential Frame {i}")
        return im, scatter, title

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=150, blit=False)
    ani.save(f"PP_anim_{dp}_{solver}.gif", writer=PillowWriter(fps=10))
    plt.show()

# === Plot Energy ===
def plot_energy(log_data):
    KEs = log_data['KE']
    PEs = log_data['PE']
    TEs = log_data['Total']

    plt.figure()
    plt.plot(KEs, label='Kinetic Energy')
    plt.plot(PEs, label='Potential Energy')
    plt.plot(TEs, label='Total Energy')
    plt.xlabel("Step")
    plt.ylabel("Energy")
    plt.title("Energy Evolution")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(TEs - TEs.iloc[0], label='Δ Total Energy')
    plt.xlabel("Step")
    plt.ylabel("ΔE")
    plt.title("Energy Conservation Error")
    plt.legend()
    plt.grid()
    plt.show()

# === Plot Momentum ===
def plot_momentum(log_data):
    P0 = np.array([log_data['Px'][0], log_data['Py'][0], log_data['Pz'][0]])
    delta_P = log_data[['Px', 'Py', 'Pz']].to_numpy() - P0

    plt.figure()
    plt.plot(np.abs(delta_P[:, 0]), label='|ΔP_x|')
    plt.plot(np.abs(delta_P[:, 1]), label='|ΔP_y|')
    plt.plot(np.abs(delta_P[:, 2]), label='|ΔP_z|')
    plt.xlabel("Step")
    plt.ylabel("Momentum Error")
    plt.title("Momentum Conservation Error")
    plt.legend()
    plt.grid()
    plt.show()

# === Main Entry ===
if __name__ == "__main__":
    log_data = pd.read_csv(log_csv)

    # Animate Particles + Potential
    PP_anim(pot_frames, particle_frames, box_size, dp, solver)

    # Plot Diagnostics
    plot_energy(log_data)
    plot_momentum(log_data)