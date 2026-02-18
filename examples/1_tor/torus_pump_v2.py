import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"

import numpy as np
import dedalus.public as d3
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

output_dir = 'frames_torus'
os.makedirs(output_dir, exist_ok=True)

# ==============================================
# ПАРАМЕТРЫ
# ==============================================
Lx, Ly, Lz = 24, 24, 24
Nx, Ny, Nz = 32, 32, 32
R_torus = 5.0
g = 5.0
omega_trap = 2.0
DESIRED_PEAK = 1.0
GAS_FRACTION = 0.30   # газ = 30% от пика тора (виден на линейной шкале)

# ==============================================
# БАЗИС
# ==============================================
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist   = d3.Distributor(coords, dtype=np.complex128)

xbasis = d3.ComplexFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=3/2)
ybasis = d3.ComplexFourier(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2), dealias=3/2)
zbasis = d3.ComplexFourier(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=3/2)

psi    = dist.Field(name='psi',    bases=(xbasis, ybasis, zbasis))
V_trap = dist.Field(name='V_trap', bases=(xbasis, ybasis, zbasis))

x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
r_cyl   = np.sqrt(x**2 + y**2)

V_trap['g'] = 0.5 * omega_trap**2 * ((r_cyl - R_torus)**2 + z**2)

# Координаты осей (для срезов)
x1d = x[:, 0, 0]
y1d = y[0, :, 0]
z1d = z[0, 0, :]
iz0 = np.argmin(np.abs(z1d))
iy0 = np.argmin(np.abs(y1d))

logger.info(f"Midplane slices: z={z1d[iz0]:.3f} (iz0={iz0}), y={y1d[iy0]:.3f} (iy0={iy0})")

# ==============================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================================
def normalize(psi, target=DESIRED_PEAK):
    psi.change_scales(1)
    cur = np.max(np.abs(psi['g'])**2)
    if cur > 1e-30:
        psi['g'] *= np.sqrt(target / cur)

def plot_state(psi, label, fname):
    psi.change_scales(1)
    dens = np.abs(psi['g'])**2
    peak = float(np.max(dens))
    vmax = peak if peak > 0 else 1.0

    dxy = dens[:, :, iz0]
    dxz = dens[:, iy0, :]

    vmin_log = max(vmax * 1e-4, 1e-10)
    dxy_log  = np.clip(dxy, vmin_log, None)
    dxz_log  = np.clip(dxz, vmin_log, None)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"{label}   peak={peak:.4f}", fontsize=14)

    ext_xy = [-Lx/2, Lx/2, -Ly/2, Ly/2]
    ext_xz = [-Lx/2, Lx/2, -Lz/2, Lz/2]

    # Верхний ряд: линейная шкала
    im = axes[0,0].imshow(dxy.T, extent=ext_xy, cmap='inferno',
                           origin='lower', vmin=0, vmax=vmax)
    axes[0,0].set_title("XY  linear"); axes[0,0].set_xlabel("x"); axes[0,0].set_ylabel("y")
    plt.colorbar(im, ax=axes[0,0])

    im = axes[0,1].imshow(dxz.T, extent=ext_xz, cmap='inferno',
                           origin='lower', vmin=0, vmax=vmax)
    axes[0,1].set_title("XZ  linear"); axes[0,1].set_xlabel("x"); axes[0,1].set_ylabel("z")
    plt.colorbar(im, ax=axes[0,1])

    # Нижний ряд: логарифмическая шкала (газ виден)
    norm = LogNorm(vmin=vmin_log, vmax=vmax)

    im = axes[1,0].imshow(dxy_log.T, extent=ext_xy, cmap='inferno',
                           origin='lower', norm=norm)
    axes[1,0].set_title("XY  LOG (gas visible)"); axes[1,0].set_xlabel("x"); axes[1,0].set_ylabel("y")
    plt.colorbar(im, ax=axes[1,0])

    im = axes[1,1].imshow(dxz_log.T, extent=ext_xz, cmap='inferno',
                           origin='lower', norm=norm)
    axes[1,1].set_title("XZ  LOG (gas visible)"); axes[1,1].set_xlabel("x"); axes[1,1].set_ylabel("z")
    plt.colorbar(im, ax=axes[1,1])

    plt.tight_layout()
    plt.savefig(fname, dpi=100)
    plt.close()
    logger.info(f"Saved: {fname}  (peak={peak:.4e})")

# ==============================================
# НАЧАЛЬНОЕ УСЛОВИЕ
# ==============================================
psi.change_scales(1)
d_torus = np.sqrt((r_cyl - R_torus)**2 + z**2)
psi['g'] = np.exp(-d_torus**2 / (2 * 1.5**2)).astype(np.complex128)
normalize(psi)
plot_state(psi, "INITIAL", f"{output_dir}/00_initial.png")

# ==============================================
# ФАЗА 1: МНИМОЕ ВРЕМЯ
# ==============================================
logger.info("PHASE 1: Imaginary time relaxation")

problem_relax = d3.IVP([psi], namespace=locals())
problem_relax.add_equation(
    "dt(psi) = 0.5*div(grad(psi)) - V_trap*psi - g*conj(psi)*psi*psi"
)

solver_relax = problem_relax.build_solver(d3.RK222)
solver_relax.stop_sim_time = 3.0
dt_relax = 1e-3

frame_r = 0
while solver_relax.proceed:
    solver_relax.step(dt_relax)
    normalize(psi)

    if not np.all(np.isfinite(psi['g'])):
        logger.error(f"NaN at relax iter={solver_relax.iteration}"); break

    if solver_relax.iteration % 500 == 0:
        t = solver_relax.sim_time
        logger.info(f"  [Relax] tau={t:.3f}")
        plot_state(psi, f"Relax tau={t:.2f}", f"{output_dir}/relax_{frame_r:04d}.png")
        frame_r += 1

normalize(psi)
plot_state(psi, "GROUND STATE", f"{output_dir}/01_ground_state.png")
logger.info("Phase 1 done.")

# ==============================================
# ФАЗА 2: ГАЗ + ПОЛОИДАЛЬНАЯ ФАЗА
# ==============================================
logger.info("PHASE 2: Adding gas + poloidal phase")

psi.change_scales(1)
psi_torus = psi['g'].copy()

d_torus   = np.sqrt((r_cyl - R_torus)**2 + z**2)
theta_pol = np.arctan2(z, r_cyl - R_torus)

# --- Газ ---
# Амплитуда: sqrt(GAS_FRACTION) — чтобы |psi_gas|^2 = GAS_FRACTION
gas_amp = np.sqrt(GAS_FRACTION)

# Равномерный фон с мягкими волнами для наглядности
gas_bg = gas_amp * (
    1.0
    + 0.2 * np.sin(2 * np.pi * x / (Lx / 3))
    + 0.2 * np.cos(2 * np.pi * y / (Ly / 3))
    + 0.15 * np.sin(2 * np.pi * z / (Lz / 4))
)

# Маска: газ присутствует вне тела тора (плавный переход)
gas_mask   = 1.0 - np.exp(-0.5 * d_torus**2 / 1.5**2)
psi_gas    = gas_bg * gas_mask

# --- Полоидальная фаза ---
m_pump      = 1
phase_mask  = np.exp(-0.5 * d_torus**2 / 2.0**2)
phase_field = np.exp(1j * m_pump * theta_pol * phase_mask)

# Суперпозиция тор + газ, с фазой
psi['g'] = (psi_torus + psi_gas) * phase_field

# Нормируем так, чтобы пик тора ≈ 1 (не меняем абсолютный масштаб слишком сильно)
normalize(psi)

logger.info(f"Gas density fraction: {GAS_FRACTION:.2f}")
logger.info(f"Gas amplitude: {gas_amp:.4f}")
plot_state(psi, "Torus + Gas + Phase", f"{output_dir}/02_torus_gas_phase.png")

# ==============================================
# ФАЗА 3: РЕАЛЬНАЯ ДИНАМИКА
# ==============================================
logger.info("PHASE 3: Real-time GPE dynamics")

problem_real = d3.IVP([psi], namespace=locals())
problem_real.add_equation(
    "dt(psi) - 0.5j*div(grad(psi)) = -1j*V_trap*psi - 1j*g*conj(psi)*psi*psi"
)

solver_real = problem_real.build_solver(d3.RK222)
solver_real.stop_sim_time = 5.0
dt_real = 2e-4

frame_idx = 0
while solver_real.proceed:
    solver_real.step(dt_real)

    psi.change_scales(1)
    if not np.all(np.isfinite(psi['g'])):
        logger.error(f"NaN at real t={solver_real.sim_time:.4f}"); break

    if solver_real.iteration % 500 == 0:
        t = solver_real.sim_time
        logger.info(f"  [Real] t={t:.3f}")
        plot_state(psi, f"Real t={t:.2f}", f"{output_dir}/frame_{frame_idx:04d}.png")
        frame_idx += 1

logger.info(f"Done! {frame_idx} frames in {output_dir}/")