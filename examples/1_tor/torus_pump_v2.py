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

output_dir = 'examples/1_tor/frames_torus'
os.makedirs(output_dir, exist_ok=True)

# ==============================================
# ПАРАМЕТРЫ — уменьшена сетка для скорости
# ==============================================
Lx, Ly, Lz = 24, 24, 24
Nx, Ny, Nz = 32, 32, 32   # 32^3 — в 8 раз быстрее чем 64^3
R_torus    = 5.0
g          = 5.0
omega_trap = 2.0
DESIRED_PEAK = 1.0
GAS_PEAK     = 0.5

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

def plot_xy(psi, label, fname, vmax=None):
    """XY (сверху) + XZ (сбоку)"""
    psi.change_scales(1)
    dens = np.abs(psi['g'])**2
    peak = float(np.max(dens))
    if vmax is None:
        vmax = peak

    dxy = dens[:, :, iz0]
    dxz = dens[:, iy0, :]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f"{label}   peak={peak:.3f}", fontsize=13)

    im0 = axes[0].imshow(dxy.T, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2],
                          cmap='inferno', origin='lower', vmin=0, vmax=vmax)
    axes[0].set_title("XY (top view)")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(dxz.T, extent=[-Lx/2, Lx/2, -Lz/2, Lz/2],
                          cmap='inferno', origin='lower', vmin=0, vmax=vmax)
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
# ФАЗА 1: МНИМОЕ ВРЕМЯ (быстрее — меньше шагов)
# ==============================================
logger.info("PHASE 1: Imaginary time relaxation (32^3 grid)")

problem_relax = d3.IVP([psi], namespace=locals())
problem_relax.add_equation(
    "dt(psi) = 0.5*div(grad(psi)) - V_trap*psi - g*conj(psi)*psi*psi"
)

solver_relax = problem_relax.build_solver(d3.RK222)
solver_relax.stop_sim_time = 2.0   # меньше времени — достаточно для сходимости
dt_relax = 2e-3                    # больший шаг — быстрее

while solver_relax.proceed:
    solver_relax.step(dt_relax)
    normalize(psi)
    if not np.all(np.isfinite(psi['g'])):
        logger.error("NaN in relaxation!"); break

normalize(psi)
plot_xy(psi, "GROUND STATE", f"{output_dir}/01_ground_state.png")
logger.info("Phase 1 done.")

# ==============================================
# ФАЗА 2: ДОБАВЛЯЕМ ГАЗ ВНУТРИ ДЫРКИ ТОРА
# ==============================================
logger.info("PHASE 2: Adding gas blobs INSIDE torus hole")

psi.change_scales(1)
psi_torus = psi['g'].copy()
d_torus   = np.sqrt((r_cyl - R_torus)**2 + z**2)
theta_pol = np.arctan2(z, r_cyl - R_torus)

# -------------------------------------------------------
# 3 сгустка газа внутри дырки (r < R_torus - 1)
# расположены несимметрично — интереснее наблюдать движение
# -------------------------------------------------------
def gauss3d(cx, cy, cz, sx=1.0, sy=1.0, sz=1.5):
    return np.exp(-((x-cx)**2/(2*sx**2) + (y-cy)**2/(2*sy**2) + (z-cz)**2/(2*sz**2)))

# Сгустки внутри дырки — r ~ 1..3, z ~ 0
psi_gas = (
      1.0 * gauss3d( 2.0,  0.5, 0.0, sx=1.0, sy=1.0, sz=1.5)
    + 0.8 * gauss3d(-1.5,  2.0, 0.0, sx=1.0, sy=1.0, sz=1.5)
    + 0.9 * gauss3d( 0.5, -2.5, 0.0, sx=1.0, sy=1.0, sz=1.5)
    + 1.0 * gauss3d( 0.0,  0.0, 0.0, sx=1.0, sy=1.0, sz=1.5)  # центр
    + 0.7 * gauss3d(-2.0, -1.0, 2.0, sx=1.0, sy=1.0, sz=1.0)  # выше
).astype(np.complex128)

# Нормируем газ к GAS_PEAK
gas_peak_cur = np.max(np.abs(psi_gas)**2)
psi_gas *= np.sqrt(GAS_PEAK / gas_peak_cur)

logger.info(f"Gas: 3 blobs inside hole, peak={np.max(np.abs(psi_gas)**2):.4f}")

# -------------------------------------------------------
# ПОЛОИДАЛЬНАЯ ФАЗА (насос, m=1)
# -------------------------------------------------------
m_pump     = 1
phase_mask = np.exp(-0.5 * d_torus**2 / 2.0**2)
psi['g']   = (psi_torus + psi_gas) * np.exp(1j * m_pump * theta_pol * phase_mask)

# Сохраняем начальное состояние с газом
# vmax = GAS_PEAK*1.5 — тор насыщен белым, газ виден ярко
plot_xy(psi, "t=0.00  Torus + Gas in hole",
        f"{output_dir}/02_initial_with_gas.png",
        vmax=GAS_PEAK * 1.5)

# ==============================================
# ФАЗА 3: РЕАЛЬНАЯ ДИНАМИКА
# ==============================================
logger.info("PHASE 3: Real-time dynamics — watching gas get entrained!")

problem_real = d3.IVP([psi], namespace=locals())
problem_real.add_equation(
    "dt(psi) - 0.5j*div(grad(psi)) = -1j*V_trap*psi - 1j*g*conj(psi)*psi*psi"
)

solver_real = problem_real.build_solver(d3.RK222)
solver_real.stop_sim_time = 8.0   # достаточно долго чтобы увидеть затягивание
dt_real = 5e-4                    # шаг покрупнее — быстрее

# Сохраняем часто — каждые 100 шагов (dt=5e-4 → каждые 0.05 единиц времени)
frame_idx = 0
save_every = 100

while solver_real.proceed:
    solver_real.step(dt_real)

    psi.change_scales(1)
    if not np.all(np.isfinite(psi['g'])):
        logger.error(f"NaN at t={solver_real.sim_time:.4f}"); break

    if solver_real.iteration % save_every == 0:
        t = solver_real.sim_time
        logger.info(f"  [Real] t={t:.3f}")
        plot_xy(psi, f"t={t:.2f}",
                f"{output_dir}/frame_{frame_idx:04d}.png",
                vmax=GAS_PEAK * 1.5)
        frame_idx += 1

logger.info(f"Done! {frame_idx} frames in {output_dir}/")
logger.info("Tip: make video with:  ffmpeg -r 20 -i frames_torus/frame_%04d.png -vcodec libx264 torus.mp4")