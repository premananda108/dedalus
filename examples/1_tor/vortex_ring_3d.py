"""
3D incompressible Navier–Stokes in a fully periodic box with a toroidal vortex ring
initial condition constructed from a vector potential A = A_theta(rho,z) e_theta.

Run (serial):
    python3 vortex_ring_3d.py

Run (MPI, e.g. 4 ranks):
    mpiexec -n 4 python3 vortex_ring_3d.py

Output:
    snapshots/*.h5  (speed, vorticity magnitude, pressure)
"""

import os
# Disable internal threading for numpy/numexpr to avoid performance degradation in Dedalus
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"

import numpy as np
import dedalus.public as d3
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Папка для кадров
output_dir = "examples/1_tor/frames_vortex"
os.makedirs(output_dir, exist_ok=True)

# -----------------
# Parameters
# -----------------
Lx = Ly = Lz = 2*np.pi
Nx = Ny = Nz = 64          # lowered to 64 for much faster simulation preview
dealias = 3/2

Re = 2e4                   # "долгоживучесть" ↑ при большом Re (меньше вязкость)
nu = 1 / Re

# Vortex ring geometry (in nondimensional units)
R0 = 1.5                   # major radius (distance from z-axis to ring centerline)
a  = 0.25                  # core size (Gaussian thickness)
A0 = 0.35                  # vector potential amplitude (controls circulation/strength)

timestepper = d3.RK222
stop_sim_time = 8.0
max_timestep = 5e-3

dtype = np.float64

# -----------------
# Bases / fields
# -----------------
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist   = d3.Distributor(coords, dtype=dtype)

xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)
bases = (xbasis, ybasis, zbasis)

p = dist.Field(name='p', bases=bases)
u = dist.VectorField(coords, name='u', bases=bases)
tau_p = dist.Field(name='tau_p')     # spatially constant Lagrange multiplier (gauge helper)

# Grids
x, y, z = dist.local_grids(xbasis, ybasis, zbasis)

# -----------------
# Problem (IVP)
# -----------------
problem = d3.IVP([u, p, tau_p], namespace=locals())
problem.add_equation("dt(u) + grad(p) - nu*lap(u) = - u@grad(u)")
problem.add_equation("div(u) + tau_p = 0")   # removes the degenerate mean constraint
problem.add_equation("integ(p) = 0")         # pressure gauge (fix mean pressure)

solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# -----------------
# Initial condition: toroidal vortex ring via vector potential
# A = A_theta(rho,z) e_theta,  u = curl(A)
# In cylindrical coords:
#   u_rho = -∂Aθ/∂z
#   u_z   = (1/ρ) ∂(ρ Aθ)/∂ρ
# -----------------
rho = np.sqrt(x**2 + y**2)
theta = np.arctan2(y, x)

s2 = (rho - R0)**2 + z**2
A_theta = A0 * np.exp(-s2 / a**2)

# derivatives
dA_dz   = A_theta * (-2*z / a**2)
dA_drho = A_theta * (-2*(rho - R0) / a**2)

u_rho = -dA_dz

rho_safe = np.where(rho == 0.0, 1e-14, rho)
u_z = (A_theta + rho * dA_drho) / rho_safe

# cylindrical -> cartesian
u_x = u_rho * np.cos(theta)
u_y = u_rho * np.sin(theta)

u['g'][0] = u_x
u['g'][1] = u_y
u['g'][2] = u_z

# Optional: remove tiny mean drift (sometimes helps aesthetics in periodic box)
# umean = np.mean(u['g'], axis=(1,2,3), keepdims=True)
# u['g'] -= umean

# -----------------
# Diagnostics / output
# -----------------
omega = d3.Curl(u)
speed = np.sqrt(u@u)
wmag  = np.sqrt(omega@omega)

snapshots = solver.evaluator.add_file_handler("snapshots", sim_dt=0.1, max_writes=200)
snapshots.add_task(speed, name="speed")
snapshots.add_task(wmag,  name="vorticity_mag")
snapshots.add_task(p,     name="pressure")

# CFL control
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.3,
             threshold=0.1, max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Simple logging
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(speed, name='speed')

# -----------------
# Main loop
# -----------------
try:
    logger.info("Starting main loop")
    frame_idx = 0
    while solver.proceed:
        dt = CFL.compute_timestep()
        solver.step(dt)
        if (solver.iteration - 1) % 20 == 0:
            logger.info(f"it={solver.iteration:6d}, t={solver.sim_time:10.5e}, dt={dt:9.2e}, max|u|={flow.max('speed'):.3e}")
            
            # --- Встроенная визуализация ---
            # Работает просто при запуске в 1 поток (serial)
            if dist.comm.size == 1:
                w_eval = wmag.evaluate()
                w_eval.change_scales(1)
                
                # Ищем индексы локальной сетки: y=0 (XZ срез) и z=0 (XY срез)
                y1d = y[0, :, 0]
                iy0 = np.argmin(np.abs(y1d))
                z1d = z[0, 0, :]
                iz0 = np.argmin(np.abs(z1d))
                
                # Берем двумерные срезы завихренности
                slice_xz = w_eval['g'][:, iy0, :]
                slice_xy = w_eval['g'][:, :, iz0]
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                
                # Вид сверху (XY плоскость)
                c1 = ax1.contourf(x[:, 0, 0], y[0, :, 0], slice_xy.T, levels=40, cmap='magma')
                ax1.set_title(f"Top view (XY) t={solver.sim_time:.2f}")
                ax1.set_xlabel("x")
                ax1.set_ylabel("y")
                fig.colorbar(c1, ax=ax1)

                # Вид сбоку (XZ плоскость)
                c2 = ax2.contourf(x[:, 0, 0], z[0, 0, :], slice_xz.T, levels=40, cmap='magma')
                ax2.set_title("Side view (XZ)")
                ax2.set_xlabel("x")
                ax2.set_ylabel("z")
                fig.colorbar(c2, ax=ax2)
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/frame_{frame_idx:04d}.png", dpi=100)
                plt.close()
                frame_idx += 1

except Exception:
    logger.exception("Exception raised, triggering end of main loop.")
    raise
finally:
    solver.log_stats()
