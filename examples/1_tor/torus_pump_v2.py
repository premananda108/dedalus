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

output_dir = 'examples/1_tor/frames_torus_v2'
os.makedirs(output_dir, exist_ok=True)

# ==============================================
# ПАРАМЕТРЫ
# ==============================================
Lx, Ly, Lz = 24, 24, 24
Nx, Ny, Nz = 48, 48, 48
R_torus    = 5.0
g_tor      = 5.0    # нелинейность тора
g_gas      = 1.5    # самовзаимодействие газа — удерживает форму сгустков
omega_trap = 2.0
DESIRED_PEAK = 1.0
GAS_PEAK     = 0.5
V_POL        = 3.0

# ==============================================
# БАЗИС
# ==============================================
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist   = d3.Distributor(coords, dtype=np.complex128)

xbasis = d3.ComplexFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=3/2)
ybasis = d3.ComplexFourier(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2), dealias=3/2)
zbasis = d3.ComplexFourier(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=3/2)

psi_t  = dist.Field(name='psi_t',  bases=(xbasis, ybasis, zbasis))
phi    = dist.Field(name='phi',    bases=(xbasis, ybasis, zbasis))
V_trap = dist.Field(name='V_trap', bases=(xbasis, ybasis, zbasis))
V_tor  = dist.Field(name='V_tor',  bases=(xbasis, ybasis, zbasis))
V_pol  = dist.Field(name='V_pol',  bases=(xbasis, ybasis, zbasis))
V_nl   = dist.Field(name='V_nl',   bases=(xbasis, ybasis, zbasis))

# ВАЖНО: получаем грид с scales=1, чтобы размер был 32x32x32, а не 48x48x48
x, y, z = dist.local_grids(xbasis, ybasis, zbasis, scales=1)
r_cyl   = np.sqrt(x**2 + y**2)

x1d = x[:, 0, 0];  y1d = y[0, :, 0];  z1d = z[0, 0, :]
iz0 = np.argmin(np.abs(z1d))
iy0 = np.argmin(np.abs(y1d))

# ==============================================
# УТИЛИТЫ
# ==============================================
def normalize(field, target=DESIRED_PEAK):
    field.change_scales(1)
    cur = np.max(np.abs(field['g'])**2)
    if cur > 1e-30:
        field['g'] *= np.sqrt(target / cur)

def gauss3d(cx, cy, cz, sx=1.0, sy=1.0, sz=1.5):
    return np.exp(-((x-cx)**2/(2*sx**2) +
                    (y-cy)**2/(2*sy**2) +
                    (z-cz)**2/(2*sz**2)))

def set_field(field, data):
    """Присвоить данные полю с правильным scale=1."""
    field.change_scales(1)
    field['g'] = data

def plot_frame(psi_t, phi, label, fname, gas_vmax=None):
    psi_t.change_scales(1); phi.change_scales(1)
    dens_tor = np.abs(psi_t['g'])**2
    dens_gas = np.abs(phi['g'])**2

    if gas_vmax is None:
        gas_vmax = float(np.max(dens_gas)) * 1.2 or GAS_PEAK

    tor_xy = dens_tor[:, :, iz0]
    gas_xy = dens_gas[:, :, iz0]
    tor_xz = dens_tor[:, iy0, :]
    gas_xz = dens_gas[:, iy0, :]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(label, fontsize=13)

    ext_xy = [-Lx/2, Lx/2, -Ly/2, Ly/2]
    ext_xz = [-Lx/2, Lx/2, -Lz/2, Lz/2]

    tor_vmax = float(np.max(dens_tor)) or 1.0

    def make_rgb(tor, gas, tor_vmax, gas_vmax):
        r     = np.clip(tor / tor_vmax, 0, 1)
        gas_n = np.clip(gas / gas_vmax, 0, 1) ** 0.5  # гамма-коррекция
        b     = gas_n
        g_ch  = gas_n * 0.4
        return np.stack([r.T, g_ch.T, b.T], axis=-1)

    axes[0].imshow(make_rgb(tor_xy, gas_xy, tor_vmax, gas_vmax), extent=ext_xy, origin='lower')
    axes[0].set_title("XY top view  (red=torus, blue=gas)")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

    axes[1].imshow(make_rgb(tor_xz, gas_xz, tor_vmax, gas_vmax), extent=ext_xz, origin='lower')
    axes[1].set_title("XZ side view  (red=torus, blue=gas)")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("z")

    plt.tight_layout()
    plt.savefig(fname, dpi=100)
    plt.close()
    logger.info(f"Saved: {fname}")

# ==============================================
# ИНИЦИАЛИЗАЦИЯ СТАТИЧЕСКИХ ПОЛЕЙ
# ==============================================
set_field(V_trap, 0.5 * omega_trap**2 * ((r_cyl - R_torus)**2 + z**2))

d_torus_2d = np.sqrt((r_cyl - R_torus)**2 + z**2)
pump_mask  = np.exp(-0.5 * d_torus_2d**2 / 3.0**2)
theta_pol  = np.arctan2(z, r_cyl - R_torus)
set_field(V_pol, -V_POL * theta_pol * pump_mask)

set_field(V_nl, np.zeros((Nx, Ny, Nz)))  # пока нулевой

# ==============================================
# ФАЗА 1: РЕЛАКСАЦИЯ ТОРА
# ==============================================
logger.info("PHASE 1: Torus ground state")

d_torus = np.sqrt((r_cyl - R_torus)**2 + z**2)
set_field(psi_t, np.exp(-d_torus**2 / (2 * 1.5**2)).astype(np.complex128))
normalize(psi_t)

problem_relax = d3.IVP([psi_t], namespace=locals())
problem_relax.add_equation(
    "dt(psi_t) = 0.5*div(grad(psi_t)) - V_trap*psi_t - g_tor*conj(psi_t)*psi_t*psi_t"
)

solver_r = problem_relax.build_solver(d3.RK222)
solver_r.stop_sim_time = 2.0
while solver_r.proceed:
    solver_r.step(2e-3)
    normalize(psi_t)
    if not np.all(np.isfinite(psi_t['g'])):
        logger.error("NaN in torus relaxation!"); break

normalize(psi_t)
psi_t.change_scales(1)
torus_profile = psi_t['g'].copy()
logger.info(f"Torus ground state: peak={np.max(np.abs(torus_profile)**2):.4f}")

# ==============================================
# ФАЗА 2: ПОТЕНЦИАЛ ОТ ТОРА
# ==============================================
logger.info("PHASE 2: Building torus potential")
set_field(V_tor, g_tor * np.abs(torus_profile)**2)

# ==============================================
# ФАЗА 3: НАЧАЛЬНОЕ УСЛОВИЕ ДЛЯ ГАЗА
# ==============================================
logger.info("PHASE 3: Setting up gas initial condition")

gas_init = (
      1.0 * gauss3d( 2.0,  0.5,  0.0, sx=1.0, sy=1.0, sz=1.2)
    + 0.9 * gauss3d(-1.5,  1.5,  0.0, sx=1.0, sy=1.0, sz=1.2)
    + 0.8 * gauss3d( 0.5, -2.0,  0.0, sx=1.0, sy=1.0, sz=1.2)
    + 0.7 * gauss3d(-1.0, -1.0,  2.0, sx=0.8, sy=0.8, sz=0.8)
    + 0.7 * gauss3d( 1.5,  1.0, -2.0, sx=0.8, sy=0.8, sz=0.8)
).astype(np.complex128)
gas_init *= np.sqrt(GAS_PEAK / np.max(np.abs(gas_init)**2))
set_field(phi, gas_init)

logger.info(f"Gas initial peak: {np.max(np.abs(phi['g'])**2):.4f}")
plot_frame(psi_t, phi, "t=0.00  Initial state", f"{output_dir}/00_initial.png")

# ==============================================
# ФАЗА 4: ЭВОЛЮЦИЯ ГАЗА (тор заморожен)
# ==============================================
# Уравнение Гросса-Питаевского:
#   i*d_t(phi) = -0.5*Lap(phi) + (V_trap + V_tor + V_pol + V_nl)*phi
# V_nl = g_gas*|phi|^2 обновляется явно перед каждым шагом —
# Dedalus видит только линейный член V_nl*phi.
logger.info("PHASE 4: Gas dynamics (GPE) in frozen torus + poloidal pump")

problem_gas = d3.IVP([phi], namespace=locals())
problem_gas.add_equation(
    "dt(phi) - 0.5j*div(grad(phi)) = -1j*(V_trap + V_tor + V_pol + V_nl)*phi"
)

solver_gas = problem_gas.build_solver(d3.RK222)
solver_gas.stop_sim_time = 10.0
dt_gas = 5e-4

frame_idx = 0
save_every = 100  # каждые 0.05 единиц времени

while solver_gas.proceed:
    # Обновляем нелинейный потенциал ПЕРЕД шагом
    phi.change_scales(1)
    V_nl.change_scales(1)
    V_nl['g'] = g_gas * np.abs(phi['g'])**2

    solver_gas.step(dt_gas)

    phi.change_scales(1)
    if not np.all(np.isfinite(phi['g'])):
        logger.error(f"NaN at t={solver_gas.sim_time:.4f}"); break

    if solver_gas.iteration % save_every == 0:
        t = solver_gas.sim_time
        gas_peak = float(np.max(np.abs(phi['g'])**2))
        logger.info(f"  t={t:.3f}  gas_peak={gas_peak:.4f}")
        plot_frame(psi_t, phi,
                   f"t={t:.2f}  gas_peak={gas_peak:.3f}",
                   f"{output_dir}/frame_{frame_idx:04d}.png",
                   gas_vmax=GAS_PEAK)
        frame_idx += 1

logger.info(f"Done! {frame_idx} frames in {output_dir}/")
logger.info("Make video: ffmpeg -r 20 -i frames_torus/frame_%04d.png -vcodec libx264 torus.mp4")