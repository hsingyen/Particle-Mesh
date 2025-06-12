import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
#from new_mass_deposition import deposit_ngp, deposit_cic, deposit_tsc
from new_orbit_integrator import kdk_step, dkd_step,rk4_step
from new_orbit_integrator import compute_phi, compute_acceleration, compute_grid_acceleration, interpolate_to_particles
from utils import Timer  # optional
from mpl_toolkits.mplot3d import Axes3D
from jeans_initial import create_particles
from poisson_solver import poisson_solver_periodic, poisson_solver_isolated,poisson_solver_isolated_green

# === Simulation parameters ===
N = 256  # Grid size: N x N x N
box_size = 1.0
N_particles =  10 #10000
center = N // 2
dt = 0.001
n_steps = 200  #200
dp = 'ngp'  # 'ngp', 'cic', or 'ts
solver = 'isolated' # 'isolated', 'periodic ,'periodic_safe'(softening = 0 equal to periodic)
integrator = 'kdk'         # 'kdk' or 'dkd' or 'rk4' or 'hermite_individual'   or 'hermite_fixed'
self_force = False          # True or False
softening = 0.0 
a = 0.001

def create_test_rho(N=128, box_size=1.0):
    """
    Create a rho field with 4 particles of mass 1/4 at center region.

    Returns
    -------
    rho : ndarray, shape (N, N, N)
    """
    rho = np.zeros((N, N, N))

    dx = box_size / N

    # 中心格點
    center = box_size / 2
    center_index = int(center / dx)

    # 四個附近格點
    offsets = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)]
    mass_per_particle = 1.0 / 4

    for dx_off, dy_off, dz_off in offsets:
        i = (center_index + dx_off) % N
        j = (center_index + dy_off) % N
        k = (center_index + dz_off) % N
        rho[i, j, k] += mass_per_particle / dx**3   # 注意: 質量密度 = 質量 / 體積

    return rho

# === fixed potential ===
GM = 1.0
epsilon = 0.0001
center = np.array([0.5, 0.5, 0.5])

# grid spacing
dx = box_size / N

# generate grid (cell-centered)
x = (np.arange(N) +0.5) * dx
y = (np.arange(N) + 0.5) * dx
z = (np.arange(N) + 0.5) * dx
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# displacement from center
dx_grid = X - center[0]
dy_grid = Y - center[1]
dz_grid = Z - center[2]

# compute distance r
r = np.sqrt(dx_grid**2 + dy_grid**2 + dz_grid**2)

# avoid division by zero at center (softening automatically handled)
r_safe = r + epsilon

# potential (for reference)
phi = -GM / r_safe
# acceleration components (analytical)
ax = -GM * dx_grid / (r_safe**3)
ay = -GM * dy_grid / (r_safe**3)
az = -GM * dz_grid / (r_safe**3)

# total acceleration magnitude (optional)
a_mag = np.sqrt(ax**2 + ay**2 + az**2)

center = np.array([0.5, 0.5, 0.5])
position = np.array([0.50390822,0.46797252,0.65835382]) #np.random.rand(3) * box_size
mass = 1.0
# 計算與中心距離
disp = position - center
r = np.sqrt(np.sum(disp**2))
r_safe = r + epsilon

print(f"Particle position: {position}")

#acc, phi = compute_acceleration(position, mass, N, box_size, dp, solver, subtract_self=self_force, soft_len=0.0)
#grad_phi = compute_grid_acceleration(phi, N, box_size)
dx = box_size / N

# 找到粒子所在的 grid index
i = int(position[0] / dx) % N
j = int(position[1] / dx) % N
k = int(position[2] / dx) % N

# 取出該格點的三個分量加速度
ax = (phi[i-1, j, k] - phi[i+1, j, k])/(2*dx)
ay = (phi[i, j-1, k] - phi[i, j+1, k])/(2*dx)
az = (phi[i, j, k-1] - phi[i, j, k+1])/(2*dx)

acc_particle = np.array([ax, ay, az])



# 解析解位勢
phi_an = -GM / r_safe

# 解析解加速度
acc_an = -GM * disp / (r_safe**3)

# 印出
#print(f"PM solver computed phi: {phi}")
#print(f"Analytical phi: {phi_an}")
#print()
print(f"PM solver computed acc: {acc_particle}")
print(f"Analytical acc: {acc_an}")


def create_test_rho(N=128, box_size=1.0):
    rho = np.zeros((N, N, N))
    dx = box_size / N
    center = box_size / 2
    center_index = int(center / dx)

    offsets = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)]
    mass_per_particle = 1.0 / 4

    for dx_off, dy_off, dz_off in offsets:
        i = (center_index + dx_off) % N
        j = (center_index + dy_off) % N
        k = (center_index + dz_off) % N
        rho[i, j, k] += mass_per_particle / dx**3

    return rho

# 理論位能 (point mass)
def phi_analytic(x, y, z, center=np.array([0.5, 0.5, 0.5]), G=1.0, M=1.0, epsilon=0.001):
    dx = x - center[0]
    dy = y - center[1]
    dz = z - center[2]
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    return - G * M / (r + epsilon)

# 主測試
def test_poisson_solver():
    N = 128
    box_size = 1.0
    G = 1.0
    epsilon = 0.0001

    # 產生 rho
    rho = create_test_rho(N, box_size)


    phi = poisson_solver_isolated(rho, box_size, G)
    phi = phi+10

    # 建立網格
    grid = np.linspace(0, box_size, N, endpoint=False)
    X, Y, Z = np.meshgrid(grid, grid, grid, indexing='ij')

    # 計算理論解
    phi_true = phi_analytic(X, Y, Z, center=np.array([0.5, 0.5, 0.5]), G=G, M=1.0, epsilon=epsilon)

    # 比較誤差
    diff = phi - phi_true

    # 繪圖 (隨便取個切面 z=middle)
    mid = N // 2

    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)
    plt.title("Phi (Numerical)")
    plt.imshow(phi[:, :, mid], origin='lower')
    plt.colorbar()

    plt.subplot(1,3,2)
    plt.title("Phi (Analytic)")
    plt.imshow(phi_true[:, :, mid], origin='lower')
    plt.colorbar()

    plt.subplot(1,3,3)
    masked_diff = np.where(np.abs(diff[:, :, mid]) < 0.6, np.nan, diff[:, :, mid])
    plt.imshow(masked_diff, origin='lower')
    plt.title("Error (Phi_num - Phi_analytic)")
    plt.colorbar()

    plt.tight_layout()
    plt.show()
    print("Max error:", np.max(np.abs(diff)))
    print("Mean error:", np.mean(np.abs(diff)))
    print("Min error", np.min(np.abs(diff)))


test_poisson_solver()


