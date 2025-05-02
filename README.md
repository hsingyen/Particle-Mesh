# Particle Mesh Simulation for 3D Self-Gravitating Systems

This project implements a 3D Particle Mesh (PM) simulation for modeling the gravitational evolution of particles in a periodic cubic domain. The code uses FFT-based solvers to compute the gravitational potential, supports multiple mass deposition schemes, and includes basic symplectic integrators for evolving particle orbits.

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
├── src/                   # Main simulation code  
│    ├── main.py  
│    ├── poisson_solver.py  
│    ├── mass_deposition.py  
│    ├── orbit_integrator.py  
│    └── utils.py  
├── animations/            # Output animations (.mp4 or .gif)  
├── results/               # Energy plots, momentum plots, etc.  
├── data/                  # Optional: saved simulation state  
├── notebook/              # Jupyter notebooks (optional)  
├── requirements.txt       # Python dependencies  
├── .gitignore             # Files to exclude from Git  
└── README.md              # Project description  
