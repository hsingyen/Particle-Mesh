# Particle Mesh Simulation for 3D Self-Gravitating Systems

This project implements a 3D Particle Mesh (PM) simulation for modeling the gravitational evolution of particles in a periodic cubic domain. It is developed as part of the final project for a Computational Astrophysics course. The code uses FFT-based solvers to compute the gravitational potential, supports multiple mass deposition schemes, and includes basic symplectic integrators for evolving particle orbits.

---

## Features

- Periodic boundary conditions using FFT-based Poisson solver
- Mass deposition schemes:
  - Nearest Grid Point (NGP)
  - Cloud-In-Cell (CIC)
  - Triangular Shaped Cloud (TSC)
- Orbit integrators:
  - Kick-Drift-Kick (KDK)
  - Drift-Kick-Drift (DKD)
- Time evolution of particles in 3D
- Animations of:
  - Mid-plane potential field
  - Projected particle motion
  - Density field evolution
  - 3D particle clustering (rotating view)
  - Combined particle + potential visualization
- Physical validation through energy conservation

---

## Directory Structure

ComputationalAstrophysics_FinalProject/
│
├── src/                 # Main simulation code
│   ├── main.py
│   ├── poisson_solver.py
│   ├── mass_deposition.py
│   ├── orbit_integrator.py
│   └── utils.py
│
├── animations/          # Saved .mp4 or .gif animations
├── results/             # Plots (e.g. energy, momentum)
├── data/                # (Optional) Raw output arrays
├── notebook/            # Jupyter notebooks (optional)
├── requirements.txt     # Python dependencies
├── .gitignore           # Ignore .pyc, outputs, cache
└── README.md            # Project overview
