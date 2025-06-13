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
mode = "expand"

# === File paths ===
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
potential_csv  = os.path.join(BASE_DIR, f"potential_N{N_grid}.csv")
pos_csv        = os.path.join(BASE_DIR, "Result/particle_positions_ngpperiodicdkd_expand.csv")#
log_csv        = os.path.join(BASE_DIR, "Result/simulation_output_ngpperiodicdkd_expand.csv")#自己改黨名

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
# === Animation: only inside this block use dark style ===
def animate_PP(pot_frames, part_frames, box_size, dp, solver):
    # 只在這裡套用 dark_background，離開 with 區塊後會自動還原
    with plt.style.context('dark_background'):
        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        # 再明確設黑底以保險
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        scat = ax.scatter(
            part_frames[0][:,0], part_frames[0][:,1],
            s=5, c='white', edgecolors='none'
        )
        ax.set_xlim(0, box_size)
        ax.set_ylim(0, box_size)

        # 白色字、刻度
        ax.set_title("Particles", color='white')
        ax.set_xlabel("x", color='white')
        ax.set_ylabel("y", color='white')
        ax.tick_params(colors='white')

        def update(i):
            scat.set_offsets(part_frames[i])
            return scat,

        ani = animation.FuncAnimation(
            fig, update, frames=range(0, len(part_frames), 5), interval=200, blit=True
        )
        out_gif = f"PP_anim_{dp}_{solver}_{mode}.gif"
        ani.save(out_gif, writer=PillowWriter(fps=5))
        print(f"Saved animation: {out_gif}")
        plt.close(fig)

# === Energy diagnostics: 依舊是白底風格 ===
def plot_energy(log_data):
    # 因為我們沒在這裡改 style，所以預設就是白底
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

# === Momentum diagnostics: 同上 ===
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

# === Momentum diagnostics: 同上 ===
def plot_momentum(log_data):

    # 全局设置：使用论文常见的衬线字体（Times New Roman）
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'dejavuserif',  # 数学公式也用衬线
        'font.size': 16
    })

    # 计算 ΔP
    P0 = log_data.loc[0, ['Px', 'Py', 'Pz']].to_numpy()
    dP = log_data[['Px', 'Py', 'Pz']].to_numpy() - P0

    # 创建 figure & axes
    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘图
    ax.plot(log_data['step'], (np.abs(dP[:, 0])), linewidth=2, label=r'$|\Delta P_x|$')
    ax.plot(log_data['step'], (np.abs(dP[:, 1])), linewidth=2, label=r'$|\Delta P_y|$')
    ax.plot(log_data['step'], (np.abs(dP[:, 2])), linewidth=2, label=r'$|\Delta P_z|$')
    ax.set_yscale('log')
    ax.set_ylim(1e-16, 1e0)


    # 标签与标题
    ax.set_xlabel("Step", fontsize=16)
    ax.set_ylabel(r"$|\Delta P|$", fontsize=16)
    ax.set_title("Momentum Conservation Error", fontsize=18)

    # 图例 & 网格
    ax.legend(fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)

    # 刻度字体
    ax.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    plt.show()

# === Main ===
if __name__ == "__main__":
    log_data = pd.read_csv(log_csv)
    animate_PP(pot_frames, particle_frames,1.0,dp,solver)
    plot_energy(log_data)
    plot_momentum(log_data)
