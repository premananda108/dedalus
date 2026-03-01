"""
3D incompressible Navier–Stokes in a fully periodic box with TWO interacting vortex rings.
Initial condition constructed from superposition of two vector potentials.

Run (serial):
    python3 two_vortex_rings_3d.py
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
output_dir = "examples/1_tor/frames_two_rings"
os.makedirs(output_dir, exist_ok=True)

# -----------------
# Parameters
# -----------------
Lx = Ly = Lz = 2*np.pi
Nx = Ny = Nz = 64          # Разрешение сетки (64 для скорости, 96/128 для качества)
dealias = 3/2

Re = 2e4                   # Высокое число Рейнольдса (малая вязкость)
nu = 1 / Re

# Геометрия вихревых колец
R0 = 1.0                   # Начальный радиус (сделали чуть меньше, чтобы было место для расширения)
a  = 0.25                  # Толщина ядра вихря (core size)

# Кольцо 1 (снизу)
z1 = -1.0
A0_1 = 0.35                # Положительная амплитуда = летит вверх (+z)

# Кольцо 2 (сверху)
z2 = 1.0
A0_2 = -0.35               # Отрицательная амплитуда = летит вниз (-z)
# ! ПОПРОБУЙТЕ ПОТОМ СДЕЛАТЬ A0_2 = 0.35, и они будут лететь в одну сторону (эффект "leap-frogging")

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
tau_p = dist.Field(name='tau_p')

# Grids
x, y, z = dist.local_grids(xbasis, ybasis, zbasis)

# -----------------
# Problem (IVP)
# -----------------
problem = d3.IVP([u, p, tau_p], namespace=locals())
problem.add_equation("dt(u) + grad(p) - nu*lap(u) = - u@grad(u)")
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("integ(p) = 0")

solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# -----------------
# Суперпозиция двух гауссовых вихревых колец (Gaussian spherical vortex)
# Строго бездивергентное поле без сингулярностей (1/rho). 
# Формирует мощный центральный джет, заставляющий кольца физично двигаться.
# -----------------
rho2 = x**2 + y**2

# Эффективный радиус кольца (он же толщина):
R_vortex = 0.8 

# Кольцо 1 (снизу, z1 = -1.0, летит ВВЕРХ, A0_1 = 0.4)
z1 = -1.0
A0_1 = 0.4
E_1 = np.exp(-(rho2 + (z - z1)**2) / R_vortex**2)
u_x_1 = 2 * A0_1 * x * (z - z1) / (R_vortex**2) * E_1
u_y_1 = 2 * A0_1 * y * (z - z1) / (R_vortex**2) * E_1
u_z_1 = 2 * A0_1 * (1.0 - rho2 / R_vortex**2) * E_1

# Кольцо 2 (сверху, z2 = 1.0, летит ВНИЗ, A0_2 = -0.4)
z2 = 1.0
A0_2 = -0.4
E_2 = np.exp(-(rho2 + (z - z2)**2) / R_vortex**2)
u_x_2 = 2 * A0_2 * x * (z - z2) / (R_vortex**2) * E_2
u_y_2 = 2 * A0_2 * y * (z - z2) / (R_vortex**2) * E_2
u_z_2 = 2 * A0_2 * (1.0 - rho2 / R_vortex**2) * E_2

u['g'][0] = u_x_1 + u_x_2
u['g'][1] = u_y_1 + u_y_2
u['g'][2] = u_z_1 + u_z_2

# -----------------
# Diagnostics / output
# -----------------
omega = d3.Curl(u)
speed = np.sqrt(u@u)
wmag  = np.sqrt(omega@omega)

snapshots = solver.evaluator.add_file_handler("snapshots_two_rings", sim_dt=0.1, max_writes=200)
snapshots.add_task(speed, name="speed")
snapshots.add_task(wmag,  name="vorticity_mag")
snapshots.add_task(p,     name="pressure")

# CFL control
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.3,
             threshold=0.1, max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

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
            if dist.comm.size == 1:
                w_eval = wmag.evaluate()
                w_eval.change_scales(1)
                
                y1d = y[0, :, 0]
                iy0 = np.argmin(np.abs(y1d))
                z1d = z[0, 0, :]
                iz0 = np.argmin(np.abs(z1d))
                
                slice_xz = w_eval['g'][:, iy0, :]
                slice_xy = w_eval['g'][:, :, iz0]
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                
                c1 = ax1.contourf(x[:, 0, 0], y[0, :, 0], slice_xy.T, levels=40, cmap='magma')
                ax1.set_title(f"Top view (XY, z=0) t={solver.sim_time:.2f}")
                ax1.set_xlabel("x")
                ax1.set_ylabel("y")
                fig.colorbar(c1, ax=ax1)

                c2 = ax2.contourf(x[:, 0, 0], z[0, 0, :], slice_xz.T, levels=40, cmap='magma')
                ax2.set_title("Side view (XZ, y=0)")
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
