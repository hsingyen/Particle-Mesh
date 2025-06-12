import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

# === Settings ===
box_size = 1.0
dp        = "ngp"
solver    = "periodic"   # or "periodic"
N_grid    = 128

# === File paths ===
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
potential_csv  = os.path.join(BASE_DIR, f"potential_N{N_grid}.csv")
pos_csv        = os.path.join(BASE_DIR, "particle_positions.csv")
log_csv        = os.path.join(BASE_DIR, "simulation_output.csv")

# === Load particle positions ===
pos_data = pd.read_csv(pos_csv)
steps    = np.sort(pos_data['step'].unique())
particle_frames = [
    pos_data[pos_data['step'] == s][['x','y']].to_numpy()
    for s in steps
]

# === Load potential data ===
raw_pot = pd.read_csv(potential_csv, sep=r'\s+', header=None, engine='python').values
n_rows, n_cols = raw_pot.shape

if n_cols == N_grid * N_grid:
    # one long line per step
    n_steps_pot = n_rows
    pot_frames = raw_pot.reshape(n_steps_pot, N_grid, N_grid)
elif n_cols == N_grid and (n_rows % N_grid) == 0:
    # N_grid lines per step
    n_steps_pot = n_rows // N_grid
    pot_frames = raw_pot.reshape(n_steps_pot, N_grid, N_grid)
else:
    raise ValueError(
        f"Unexpected shape for potential data: {raw_pot.shape}. "
        "Should be (steps, N^2) or (steps*N, N)."
    )

# === Sync frame counts ===
n_frames = min(len(pot_frames), len(particle_frames))
pot_frames      = pot_frames[:n_frames]
particle_frames = particle_frames[:n_frames]
steps           = steps[:n_frames]

# === Animation: Potential heatmap + particle scatter ===
def animate_PP(pot_frames, part_frames):
    fig, (ax_pot, ax_part) = plt.subplots(1, 2, figsize=(10,4))

    im = ax_pot.imshow(
        pot_frames[0], origin='lower',
        extent=[0, box_size, 0, box_size],
        vmin=pot_frames.min(), vmax=pot_frames.max(),
        cmap='viridis'
    )
    ax_pot.set_title("Potential")
    ax_pot.set_xlabel("x"); ax_pot.set_ylabel("y")
    cbar = fig.colorbar(im, ax=ax_pot)
    cbar.set_label("Φ")

    scat = ax_part.scatter(
        part_frames[0][:,0], part_frames[0][:,1],
        s=5, c='white', edgecolors='black', linewidths=0.2
    )
    ax_part.set_xlim(0,box_size); ax_part.set_ylim(0,box_size)
    ax_part.set_title("Particles")
    ax_part.set_xlabel("x"); ax_part.set_ylabel("y")

    def update(frame):
        im.set_data(pot_frames[frame])
        scat.set_offsets(part_frames[frame])
        return im, scat

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=200, blit=False
    )
    out_gif = f"PP_anim_{dp}_{solver}.gif"
    ani.save(out_gif, writer=PillowWriter(fps=5))
    print(f"Saved animation: {out_gif}")
    plt.close(fig)

# === Energy diagnostics ===
def plot_energy(log_data):
    plt.figure()
    plt.plot(log_data['step'], log_data['KE'], label='KE')
    plt.plot(log_data['step'], log_data['PE'], label='PE')
    plt.plot(log_data['step'], log_data['Total'], label='Total')
    plt.xlabel("Step"); plt.ylabel("Energy")
    plt.title("Energy Evolution"); plt.legend(); plt.grid()
    plt.show()

    plt.figure()
    plt.plot(
        log_data['step'],
        log_data['Total'] - log_data['Total'].iloc[0],
        label='Δ Total'
    )
    plt.xlabel("Step"); plt.ylabel("ΔE")
    plt.title("Total‐Energy Drift"); plt.grid()
    plt.show()

# === Momentum diagnostics ===
def plot_momentum(log_data):
    P0 = log_data.loc[0, ['Px','Py','Pz']].to_numpy()
    dP = log_data[['Px','Py','Pz']].to_numpy() - P0
    plt.figure()
    plt.plot(log_data['step'], np.abs(dP[:,0]), label='|ΔPx|')
    plt.plot(log_data['step'], np.abs(dP[:,1]), label='|ΔPy|')
    plt.plot(log_data['step'], np.abs(dP[:,2]), label='|ΔPz|')
    plt.xlabel("Step"); plt.ylabel("|ΔP|")
    plt.title("Momentum Conservation Error"); plt.legend(); plt.grid()
    plt.show()

# === Main ===
if __name__ == "__main__":
    log_data = pd.read_csv(log_csv)
    animate_PP(pot_frames, particle_frames)
    plot_energy(log_data)
    plot_momentum(log_data)
