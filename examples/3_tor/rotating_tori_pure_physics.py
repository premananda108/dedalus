import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"

import numpy as np
import dedalus.public as d3
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Папки вывода ────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(SCRIPT_DIR, "frames_pure_physics")
SNAP_DIR   = os.path.join(SCRIPT_DIR, "snapshots_pure_physics") # Папка для HDF5
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(SNAP_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════
# §1. ПАРАМЕТРЫ
# ═══════════════════════════════════════════════════════
Lx = Ly = 4 * np.pi
Lz = 8 * np.pi  
Nx = Ny = 64
Nz = 128
dealias = 3 / 2

Re  = 5000.0
nu  = 1.0 / Re

# ── Параметры торов ─────────────────────────────────────
R_ring = 1.2       
a_core = 0.55      

d_sep = 3.0 * R_ring
pos_A = np.array([0.0, 0.0, +d_sep / 2])   
pos_B = np.array([0.0, 0.0, -d_sep / 2])   
vel_A = np.array([0.0, 0.0, 0.0])           
vel_B = np.array([0.0, 0.0, 0.0])

# ── Скорости вращения ──────────────────────────────────
Omega_tor = 10.0   
Omega_pol = 3.0    
sign_A = +1
sign_B = -1        # Контр-вращение -> отталкивание

# ── Физика взаимодействия ──────────────────────────────
tau_relax = 0.1    
m_eff     = 15.0   

stop_sim_time = 3.0
max_timestep  = 4e-3
dtype = np.float64

# ═══════════════════════════════════════════════════════
# §2. БАЗИСЫ И ПОЛЯ
# ═══════════════════════════════════════════════════════
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist   = d3.Distributor(coords, dtype=dtype)

xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)
bases  = (xbasis, ybasis, zbasis)

p     = dist.Field(name='p', bases=bases)
u     = dist.VectorField(coords, name='u', bases=bases)
tau_p = dist.Field(name='tau_p')

gamma_field = dist.Field(name='gamma_field', bases=bases)
u_target    = dist.VectorField(coords, name='u_target', bases=bases)

x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
dV = (Lx/Nx) * (Ly/Ny) * (Lz/Nz) 

# ═══════════════════════════════════════════════════════
# §3. ФУНКЦИИ РАСЧЁТА
# ═══════════════════════════════════════════════════════
def torus_mask(cx, cy, cz, x_g, y_g, z_g, R=R_ring, a=a_core):
    dx = x_g - cx
    dy = y_g - cy
    dz = z_g - cz
    r_cyl = np.sqrt(dx**2 + dy**2)
    s2 = (r_cyl - R)**2 + dz**2
    return np.exp(-s2 / a**2)

def torus_velocity(Omega_t, Omega_p, sign, cx, cy, cz, x_g, y_g, z_g, R=R_ring, a=a_core):
    dx = x_g - cx
    dy = y_g - cy
    dz = z_g - cz
    r_cyl  = np.sqrt(dx**2 + dy**2)
    r_safe = np.maximum(r_cyl, 1e-12)
    s2 = (r_cyl - R)**2 + dz**2
    envelope = np.exp(-s2 / a**2)

    u_phi = sign * Omega_t * envelope
    ux = -u_phi * (dy / r_safe)
    uy =  u_phi * (dx / r_safe)

    s = np.sqrt(s2 + 1e-24)
    dr = r_cyl - R
    vpol = sign * Omega_p * (2.0 * s / a) * np.exp(-s2 / a**2)
    s_safe = np.maximum(s, 1e-12)
    ux += -vpol * dz * (dx / r_safe) / s_safe
    uy += -vpol * dz * (dy / r_safe) / s_safe
    uz  =  vpol * dr / s_safe

    return ux, uy, uz

def update_fields(pA, pB, vA, vB):
    gamma_field.change_scales(1)
    u_target.change_scales(1)
    
    mA = torus_mask(pA[0], pA[1], pA[2], x, y, z)
    mB = torus_mask(pB[0], pB[1], pB[2], x, y, z)
    gamma_field['g'] = (mA + mB) / tau_relax

    uxA, uyA, uzA = torus_velocity(Omega_tor, Omega_pol, sign_A, pA[0], pA[1], pA[2], x, y, z)
    uxB, uyB, uzB = torus_velocity(Omega_tor, Omega_pol, sign_B, pB[0], pB[1], pB[2], x, y, z)

    u_target['g'][0] = (uxA + mA * vA[0]) + (uxB + mB * vB[0])
    u_target['g'][1] = (uyA + mA * vA[1]) + (uyB + mB * vB[1])
    u_target['g'][2] = (uzA + mA * vA[2]) + (uzB + mB * vB[2])

def compute_hydrodynamic_force(u_field, pos, vel, sign_t):
    cx, cy, cz = pos
    mask = torus_mask(cx, cy, cz, x, y, z)
    u_field.change_scales(1)

    ux_rot, uy_rot, uz_rot = torus_velocity(Omega_tor, Omega_pol, sign_t, cx, cy, cz, x, y, z)
    
    ux_target = ux_rot + mask * vel[0]
    uy_target = uy_rot + mask * vel[1]
    uz_target = uz_rot + mask * vel[2]

    diff_x = u_field['g'][0] - ux_target
    diff_y = u_field['g'][1] - uy_target
    diff_z = u_field['g'][2] - uz_target

    force_x = np.sum(mask * diff_x) * dV / tau_relax
    force_y = np.sum(mask * diff_y) * dV / tau_relax
    force_z = np.sum(mask * diff_z) * dV / tau_relax

    return np.array([force_x, force_y, force_z])

update_fields(pos_A, pos_B, vel_A, vel_B)
u['g'][0] = u_target['g'][0].copy()
u['g'][1] = u_target['g'][1].copy()
u['g'][2] = u_target['g'][2].copy()

# ═══════════════════════════════════════════════════════
# §4. УРАВНЕНИЯ
# ═══════════════════════════════════════════════════════
problem = d3.IVP([u, p, tau_p], namespace=locals())
problem.add_equation("dt(u) + grad(p) - nu*lap(u) = -u@grad(u) + gamma_field*(u_target - u)")
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("integ(p) = 0")

solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = stop_sim_time

# ═══════════════════════════════════════════════════════
# §5. ДИАГНОСТИКА
# ═══════════════════════════════════════════════════════
omega_field = d3.Curl(u)
speed       = np.sqrt(u @ u)
wmag        = np.sqrt(omega_field @ omega_field)

def _save_frame(frame_idx, t, wmag_xz, speed_xz, x1d, z1d, pA, pB, track_A, track_B):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor='#08081a')
    ext_xz = [x1d.min(), x1d.max(), z1d.min(), z1d.max()]
    kw_xz = dict(origin='lower', aspect='equal', extent=ext_xz, interpolation='bilinear')

    axes[0].imshow(wmag_xz.T, cmap='inferno', vmin=0, vmax=np.max(wmag_xz)+1e-5, **kw_xz)
    axes[0].set_title("Завихренность |ω|", color='white')
    
    tA, tB = np.array(track_A), np.array(track_B)
    if len(tA) > 1:
        axes[0].plot(tA[:, 0], tA[:, 2], '-', color='#00d4ff', lw=2)
        axes[0].plot(tB[:, 0], tB[:, 2], '-', color='#ff6b6b', lw=2)
    axes[0].plot(pA[0], pA[2], 'o', color='#00d4ff', ms=8)
    axes[0].plot(pB[0], pB[2], 'o', color='#ff6b6b', ms=8)

    axes[1].imshow(speed_xz.T, cmap='magma', vmin=0, vmax=np.max(speed_xz)+1e-5, **kw_xz)
    axes[1].set_title("Скорость потока |u|", color='white')

    for ax in axes:
        ax.set_facecolor('#08081a')
        ax.tick_params(colors='white')

    plt.tight_layout()
    fig.savefig(os.path.join(FRAMES_DIR, f"frame_{frame_idx:04d}.png"), dpi=100, facecolor='#08081a')
    plt.close(fig)

# ═══════════════════════════════════════════════════════
# §8. ГЛАВНЫЙ ЦИКЛ
# ═══════════════════════════════════════════════════════
if __name__ == '__main__':
    
    # ── НАСТРОЙКА СОХРАНЕНИЯ В HDF5 ──────────────────────────
    # sim_dt=0.1 означает, что файл будет сохраняться каждые 0.1 единиц симуляционного времени
    # max_writes=50 означает, что в одном .h5 файле будет максимум 50 кадров (потом создастся новый файл)
    snapshots = solver.evaluator.add_file_handler(SNAP_DIR, sim_dt=0.1, max_writes=50)
    snapshots.add_task(u, name='velocity')
    snapshots.add_task(p, name='pressure')
    snapshots.add_task(wmag, name='vorticity_mag')
    snapshots.add_task(gamma_field, name='torus_mask') # Маска торов (чтобы видеть их как твердые тела)
    # ─────────────────────────────────────────────────────────

    CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.25, threshold=0.1, max_change=1.4, min_change=0.5, max_dt=max_timestep)
    CFL.add_velocity(u)

    x1d, z1d = x[:, 0, 0], z[0, 0, :]
    iy0 = int(Ny // 2)

    track_A, track_B = [pos_A.copy()], [pos_B.copy()]
    frame_idx = 0
    executor = ThreadPoolExecutor(max_workers=2)

    logger.info("Запуск ЧЕСТНОЙ физической симуляции с сохранением в HDF5...")

    try:
        while solver.proceed:
            t = solver.sim_time
            dt_sim = CFL.compute_timestep()

            F_A = compute_hydrodynamic_force(u, pos_A, vel_A, sign_A)
            F_B = compute_hydrodynamic_force(u, pos_B, vel_B, sign_B)

            vel_A += (F_A / m_eff) * dt_sim
            vel_B += (F_B / m_eff) * dt_sim

            pos_A += vel_A * dt_sim
            pos_B += vel_B * dt_sim

            update_fields(pos_A, pos_B, vel_A, vel_B)

            solver.step(dt_sim)

            if (solver.iteration - 1) % 5 == 0:
                track_A.append(pos_A.copy())
                track_B.append(pos_B.copy())

            if (solver.iteration - 1) % 20 == 0:
                logger.info(f"it={solver.iteration:5d} | t={t:6.3f} | zA={pos_A[2]:+.3f} | zB={pos_B[2]:+.3f} | vA_z={vel_A[2]:+.3f}")
                
                w_ev = wmag.evaluate(); w_ev.change_scales(1)
                s_ev = speed.evaluate(); s_ev.change_scales(1)
                
                executor.submit(
                    _save_frame, frame_idx, t,
                    np.array(w_ev['g'])[:, iy0, :].copy(), 
                    np.array(s_ev['g'])[:, iy0, :].copy(),
                    x1d, z1d, pos_A.copy(), pos_B.copy(), list(track_A), list(track_B)
                )
                frame_idx += 1

    except Exception:
        logger.exception("Ошибка.")
    finally:
        executor.shutdown(wait=True)
        
        # Сохраняем траектории центров торов в отдельный текстовый файл для удобства
        np.savetxt(os.path.join(SCRIPT_DIR, "trajectories.txt"), 
                   np.column_stack((np.array(track_A), np.array(track_B))), 
                   header="Ax Ay Az Bx By Bz")
                   
        logger.info(f"Готово! HDF5 файлы сохранены в папку: {SNAP_DIR}")