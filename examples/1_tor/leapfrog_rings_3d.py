"""
3D incompressible Navier–Stokes in a fully periodic box.
TWO co-rotating vortex rings ("Leapfrogging" setup)
"""

import os
# Disable internal threading for numpy/numexpr
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
output_dir = "examples/1_tor/frames_leapfrog"
os.makedirs(output_dir, exist_ok=True)

snap_dir = "examples/1_tor/snapshots_leapfrog"

# -----------------
# Parameters
# -----------------
Lx = Ly = Lz = 2*np.pi
Nx = Ny = Nz = 64          # Разрешение
dealias = 3/2
Re = 2e4                   # Малая вязкость
nu = 1 / Re

timestepper = d3.RK222
stop_sim_time = 12.0       # Даем больше времени, чтобы успели разминуться
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
# Суперпозиция: два вихря летят В ОДНУ СТОРОНУ (вверх по Z)
# -----------------
rho2 = x**2 + y**2
R_vortex = 0.7 

# Кольцо 1 (Сзади)
z1 = -1.5
A0_1 = 0.5   # Положительная амплитуда = летит ВВЕРХ
E_1 = np.exp(-(rho2 + (z - z1)**2) / R_vortex**2)
u_x_1 = 2 * A0_1 * x * (z - z1) / (R_vortex**2) * E_1
u_y_1 = 2 * A0_1 * y * (z - z1) / (R_vortex**2) * E_1
u_z_1 = 2 * A0_1 * (1.0 - rho2 / R_vortex**2) * E_1

# Кольцо 2 (Спереди)
z2 = -0.1
A0_2 = 0.5   # ТАКАЯ ЖЕ амплитуда = тоже летит ВВЕРХ
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

# ── Снапшоты в HDF5 (как в tornado_3d.py) ──────────────────────
snapshots = solver.evaluator.add_file_handler(
    snap_dir, sim_dt=0.1, max_writes=500
)
snapshots.add_task(speed,  name='speed')          # |u|
snapshots.add_task(wmag,   name='vorticity_mag')  # |ω|
snapshots.add_task(p,      name='pressure')       # давление
snapshots.add_task(u,      name='velocity')       # 3D вектор скорости

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


# ═══════════════════════════════════════════════════════
# ЭКСПОРТ VTK + PVD (запускается автоматически после симуляции)
# ═══════════════════════════════════════════════════════
def export_vtk(snap_dir_=snap_dir, lx=Lx, ly=Ly, lz=Lz):
    """
    Конвертирует HDF5-снапшоты leapfrog-симуляции в формат VTK.
    Создаёт gpe_simulation.pvd — открывать в ParaView одним файлом.

    Поля в каждом кадре:
      speed         — |u|  (скорость)
      vorticity_mag — |ω|  (завихрённость, главное для визуализации)
      pressure      — p    (давление)
      velocity      — u    (3D вектор скорости)
    """
    import glob, h5py
    try:
        import pyvista as pv
    except ImportError:
        logger.warning("pyvista не установлена — VTK-экспорт пропущен.")
        logger.warning("Установите: .../dedalus_env/bin/pip install pyvista")
        return

    vtk_dir  = snap_dir_.replace("snapshots_", "vtk_output_")
    pvd_path = os.path.join(vtk_dir, "leapfrog_simulation.pvd")
    os.makedirs(vtk_dir, exist_ok=True)

    h5_files = sorted(glob.glob(os.path.join(snap_dir_, "*.h5")))
    if not h5_files:
        logger.warning(f"Нет .h5 файлов в {snap_dir_} — VTK-экспорт пропущен.")
        return

    logger.info(f"Экспорт VTK → {vtk_dir}/")
    pvd_entries   = []
    frame_counter = 0

    for file_path in h5_files:
        with h5py.File(file_path, 'r') as f:
            times       = f['scales/sim_time'][:]
            _, Nx_, Ny_, Nz_ = f['tasks/speed'].shape
            dx = lx / Nx_
            dy = ly / Ny_
            dz = lz / Nz_

            for i in range(len(times)):
                t = float(times[i])

                grid = pv.ImageData()
                grid.dimensions = (Nx_, Ny_, Nz_)
                grid.spacing    = (dx, dy, dz)
                grid.origin     = (-lx / 2, -ly / 2, -lz / 2)

                grid.point_data["speed"]         = f['tasks/speed'][i].flatten(order="F")
                grid.point_data["vorticity_mag"] = f['tasks/vorticity_mag'][i].flatten(order="F")
                grid.point_data["pressure"]      = f['tasks/pressure'][i].flatten(order="F")

                # Вектор скорости: компоненты (3, Nx, Ny, Nz) → N×3
                uv = f['tasks/velocity'][i]   # shape (3, Nx, Ny, Nz)
                u_vtk = np.stack([
                    uv[0].flatten(order="F"),
                    uv[1].flatten(order="F"),
                    uv[2].flatten(order="F"),
                ], axis=1)
                grid.point_data["velocity"] = u_vtk

                vti_name = f"leapfrog_frame_{frame_counter:04d}.vti"
                grid.save(os.path.join(vtk_dir, vti_name))
                pvd_entries.append((t, vti_name))
                frame_counter += 1

    # ── Записываем .pvd коллекцию ─────────────────────────────────
    with open(pvd_path, "w", encoding="utf-8") as pvd:
        pvd.write('<?xml version="1.0"?>\n')
        pvd.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        pvd.write('  <Collection>\n')
        for t, vti_name in pvd_entries:
            pvd.write(f'    <DataSet timestep="{t:.6f}" group="" part="0" file="{vti_name}"/>\n')
        pvd.write('  </Collection>\n')
        pvd.write('</VTKFile>\n')

    logger.info(f"✅  VTK-экспорт завершён: {frame_counter} кадров")
    logger.info(f"🎬  Открыть в ParaView: {pvd_path}")


export_vtk()
