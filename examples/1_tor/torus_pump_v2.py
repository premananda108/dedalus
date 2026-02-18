import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"

import numpy as np
import dedalus.public as d3
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

output_dir = 'frames_torus'
os.makedirs(output_dir, exist_ok=True)

# ==============================================
# ПАРАМЕТРЫ
# ==============================================
Lx, Ly, Lz = 24, 24, 24
Nx, Ny, Nz = 32, 32, 32
R_torus    = 5.0
g          = 5.0
omega_trap = 2.0
DESIRED_PEAK = 1.0
GAS_PEAK     = 0.4   # пик плотности газа (40% от тора)

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

x1d = x[:, 0, 0];  y1d = y[0, :, 0];  z1d = z[0, 0, :]
iz0 = np.argmin(np.abs(z1d))
iy0 = np.argmin(np.abs(y1d))

# ==============================================
# УТИЛИТЫ
# ==============================================
def normalize(psi, target=DESIRED_PEAK):
    psi.change_scales(1)
    cur = np.max(np.abs(psi['g'])**2)
    if cur > 1e-30:
        psi['g'] *= np.sqrt(target / cur)

def plot_state(psi, label, fname, gas_vmax=None):
    """
    2 панели: XY и XZ.
    Если gas_vmax задан — шкала сжата чтобы газ был виден.
    """
    psi.change_scales(1)
    dens = np.abs(psi['g'])**2
    peak = float(np.max(dens))
    vmax = gas_vmax if gas_vmax is not None else peak

    dxy = dens[:, :, iz0]
    dxz = dens[:, iy0, :]

    ext_xy = [-Lx/2, Lx/2, -Ly/2, Ly/2]
    ext_xz = [-Lx/2, Lx/2, -Lz/2, Lz/2]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f"{label}   peak={peak:.4f}  vmax={vmax:.3f}", fontsize=13)

    im0 = axes[0].imshow(dxy.T, extent=ext_xy, cmap='inferno',
                          origin='lower', vmin=0, vmax=vmax)
    axes[0].set_title("XY (top view)")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(dxz.T, extent=ext_xz, cmap='inferno',
                          origin='lower', vmin=0, vmax=vmax)
    axes[1].set_title("XZ (side view)")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("z")
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.savefig(fname, dpi=100)
    plt.close()
    logger.info(f"Saved: {fname}")

# ==============================================
# НАЧАЛЬНОЕ УСЛОВИЕ
# ==============================================
psi.change_scales(1)
d_torus = np.sqrt((r_cyl - R_torus)**2 + z**2)
psi['g'] = np.exp(-d_torus**2 / (2 * 1.5**2)).astype(np.complex128)
normalize(psi)

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

while solver_relax.proceed:
    solver_relax.step(dt_relax)
    normalize(psi)
    if not np.all(np.isfinite(psi['g'])):
        logger.error("NaN in relaxation!"); break

normalize(psi)
plot_state(psi, "GROUND STATE", f"{output_dir}/01_ground_state.png")
logger.info("Phase 1 done.")

# ==============================================
# ФАЗА 2: ГАЗ + ПОЛОИДАЛЬНАЯ ФАЗА
# ==============================================
logger.info("PHASE 2: Adding uniform gas + poloidal phase")

psi.change_scales(1)
psi_torus = psi['g'].copy()
d_torus   = np.sqrt((r_cyl - R_torus)**2 + z**2)
theta_pol = np.arctan2(z, r_cyl - R_torus)

# -------------------------------------------------------
# ГАЗ: равномерно по всему пространству с мягкими волнами
# Волны делают газ визуально интересным (видны переливы)
# -------------------------------------------------------
rng = np.random.default_rng(42)

# Базовый равномерный фон
gas = np.ones((Nx, Ny, Nz), dtype=np.complex128)

# Длинноволновые модуляции амплитуды — видны как светлые/тёмные области
gas *= (1.0
        + 0.35 * np.sin(2 * np.pi * x / (Lx / 2))
        + 0.35 * np.cos(2 * np.pi * y / (Ly / 2))
        + 0.25 * np.sin(2 * np.pi * z / (Lz / 3))
        + 0.20 * np.cos(2 * np.pi * (x + y) / (Lx / 2)))

# Случайная фаза в каждой точке — квазитепловой газ
random_phase = rng.uniform(0, 2 * np.pi, (Nx, Ny, Nz))
gas *= np.exp(1j * random_phase)

# Нормируем газ к GAS_PEAK
gas_peak_cur = np.max(np.abs(gas)**2)
gas *= np.sqrt(GAS_PEAK / gas_peak_cur)

logger.info(f"Gas peak density = {np.max(np.abs(gas)**2):.4f}")
logger.info(f"Gas mean density = {np.mean(np.abs(gas)**2):.4f}")

# -------------------------------------------------------
# ПОЛОИДАЛЬНАЯ ФАЗА на торе
# -------------------------------------------------------
m_pump     = 1
phase_mask = np.exp(-0.5 * d_torus**2 / 2.0**2)
psi['g']   = (psi_torus + gas) * np.exp(1j * m_pump * theta_pol * phase_mask)

psi.change_scales(1)
total_peak = float(np.max(np.abs(psi['g'])**2))
logger.info(f"Total peak after mixing = {total_peak:.4f}")

# Сохраняем с двумя шкалами
# 1) Полная шкала — виден тор
plot_state(psi, "Torus + Gas (full scale)", f"{output_dir}/02a_full_scale.png")

# 2) Шкала газа — виден газ (тор насыщен)
plot_state(psi, "Torus + Gas (gas scale)", f"{output_dir}/02b_gas_scale.png",
           gas_vmax=GAS_PEAK * 1.5)

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
        logger.error(f"NaN at t={solver_real.sim_time:.4f}"); break

    if solver_real.iteration % 500 == 0:
        t = solver_real.sim_time
        logger.info(f"  [Real] t={t:.3f}")
        # Шкала газа — чтобы всегда был виден и тор и газ
        plot_state(psi, f"t={t:.2f}",
                   f"{output_dir}/frame_{frame_idx:04d}.png",
                   gas_vmax=GAS_PEAK * 1.5)
        frame_idx += 1

logger.info(f"Done! {frame_idx} frames in {output_dir}/")