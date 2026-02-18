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

output_dir = 'examples/1_tor/frames_torus_v3'
os.makedirs(output_dir, exist_ok=True)

# ==============================================
# ПАРАМЕТРЫ
# ==============================================
Lx, Ly, Lz = 24, 24, 24
Nx, Ny, Nz = 32, 32, 32
R_torus    = 5.0
g_tor      = 5.0    # нелинейность тора
g_gas      = 8.0    # самодействие газа (держит сгустки)
omega_trap = 2.0
DESIRED_PEAK = 1.0
GAS_PEAK     = 0.5
V_POL        = 3.0  # скорость полоидального насоса

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
V_wall = dist.Field(name='V_wall', bases=(xbasis, ybasis, zbasis))

x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
r_cyl   = np.sqrt(x**2 + y**2)

x1d = x[:, 0, 0];  y1d = y[0, :, 0];  z1d = z[0, 0, :]
iz0 = np.argmin(np.abs(z1d))
iy0 = np.argmin(np.abs(y1d))

# V_trap нужен только для релаксации тора
V_trap['g'] = 0.5 * omega_trap**2 * ((r_cyl - R_torus)**2 + z**2)

# Мягкая стенка для газа (вместо V_trap — та выталкивает из дырки!)
# Газ свободно движется внутри дырки и через тор
V_wall['g'] = (
    30.0 * np.clip(r_cyl - 10.0, 0, None)**2 +
    30.0 * np.clip(np.abs(z) - 8.0, 0, None)**2
)

# ==============================================
# УТИЛИТЫ
# ==============================================
def normalize(field, target=DESIRED_PEAK):
    field.change_scales(1)
    cur = np.max(np.abs(field['g'])**2)
    if cur > 1e-30:
        field['g'] *= np.sqrt(target / cur)

def renorm_gas(field, target=GAS_PEAK):
    """Сохраняем норму газа — компенсируем численные потери"""
    field.change_scales(1)
    cur = np.max(np.abs(field['g'])**2)
    if cur > 1e-30:
        field['g'] *= np.sqrt(target / cur)

def gauss3d(cx, cy, cz, sx=1.0, sy=1.0, sz=1.5):
    return np.exp(-((x-cx)**2/(2*sx**2) +
                    (y-cy)**2/(2*sy**2) +
                    (z-cz)**2/(2*sz**2)))

def plot_frame(psi_t, phi, label, fname, gas_vmax=GAS_PEAK):
    psi_t.change_scales(1); phi.change_scales(1)
    dens_tor = np.abs(psi_t['g'])**2
    dens_gas = np.abs(phi['g'])**2

    tor_xy = dens_tor[:, :, iz0];  gas_xy = dens_gas[:, :, iz0]
    tor_xz = dens_tor[:, iy0, :];  gas_xz = dens_gas[:, iy0, :]

    def make_rgb(tor, gas, tor_vmax, gvmax):
        r  = np.clip(tor / tor_vmax, 0, 1)
        b  = np.clip(gas / gvmax,    0, 1)
        g_ch = b * 0.3
        return np.stack([r.T, g_ch.T, b.T], axis=-1)

    tor_vmax = float(np.max(dens_tor)) or 1.0
    gas_peak_cur = float(np.max(dens_gas))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f"{label}   gas_peak={gas_peak_cur:.3f}", fontsize=13)

    axes[0].imshow(make_rgb(tor_xy, gas_xy, tor_vmax, gas_vmax),
                   extent=[-Lx/2, Lx/2, -Ly/2, Ly/2], origin='lower')
    axes[0].set_title("XY top view  (red=torus, blue=gas)")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

    axes[1].imshow(make_rgb(tor_xz, gas_xz, tor_vmax, gas_vmax),
                   extent=[-Lx/2, Lx/2, -Lz/2, Lz/2], origin='lower')
    axes[1].set_title("XZ side view  (red=torus, blue=gas)")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("z")

    plt.tight_layout()
    plt.savefig(fname, dpi=100)
    plt.close()
    logger.info(f"Saved: {fname}")

# ==============================================
# ФАЗА 1: РЕЛАКСАЦИЯ ТОРА
# ==============================================
logger.info("PHASE 1: Torus ground state")

psi_t.change_scales(1)
d_torus = np.sqrt((r_cyl - R_torus)**2 + z**2)
psi_t['g'] = np.exp(-d_torus**2 / (2 * 1.5**2)).astype(np.complex128)
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
# ФАЗА 2: ПОТЕНЦИАЛЫ
# ==============================================
logger.info("PHASE 2: Building potentials")

# Отталкивание от тора
V_tor['g'] = g_tor * np.abs(torus_profile)**2

# Полоидальный насос
d_torus_2d = np.sqrt((r_cyl - R_torus)**2 + z**2)
pump_mask  = np.exp(-0.5 * d_torus_2d**2 / 3.0**2)
theta_pol  = np.arctan2(z, r_cyl - R_torus)
V_pol['g'] = -V_POL * theta_pol * pump_mask

# ==============================================
# ФАЗА 3: НАЧАЛЬНОЕ УСЛОВИЕ ГАЗА
# ==============================================
logger.info("PHASE 3: Gas initial condition")

phi.change_scales(1)
phi['g'] = (
      1.0 * gauss3d( 2.0,  0.5,  0.0, sx=1.2, sy=1.2, sz=1.5)
    + 1.0 * gauss3d(-2.0,  0.5,  0.2, sx=1.2, sy=1.2, sz=1.5)
    + 1.0 * gauss3d(-1.5,  1.5,  0.0, sx=1.0, sy=1.0, sz=1.2)
    + 1.0 * gauss3d( 0.5, -2.0,  0.0, sx=1.0, sy=1.0, sz=1.2)
    + 0.9 * gauss3d(-1.0, -1.0,  2.0, sx=0.8, sy=0.8, sz=0.8)
    + 0.9 * gauss3d( 1.5,  1.0, -2.0, sx=0.8, sy=0.8, sz=0.8)
).astype(np.complex128)

gas_cur = np.max(np.abs(phi['g'])**2)
phi['g'] *= np.sqrt(GAS_PEAK / gas_cur)
logger.info(f"Gas initial peak: {np.max(np.abs(phi['g'])**2):.4f}")

plot_frame(psi_t, phi, "t=0.00  Initial", f"{output_dir}/00_initial.png")

# ==============================================
# ФАЗА 4: ДИНАМИКА ГАЗА (тор заморожен)
# ==============================================
# Уравнение газа:
#   i*d_t(phi) = -0.5*Lap(phi) + (V_wall + V_tor + V_pol)*phi + g_gas*|phi|^2*phi
#
# V_wall — мягкая стенка на краях (НЕ V_trap: та выталкивает газ из дырки)
# V_tor  — отталкивание от тора
# V_pol  — полоидальный насос
# g_gas  — самодействие газа (удерживает сгустки компактными)
logger.info("PHASE 4: Gas dynamics")

problem_gas = d3.IVP([phi], namespace=locals())
problem_gas.add_equation(
    "dt(phi) - 0.5j*div(grad(phi)) = "
    "-1j*(V_wall + V_tor + V_pol)*phi - 1j*g_gas*conj(phi)*phi*phi"
)

solver_gas = problem_gas.build_solver(d3.RK222)
solver_gas.stop_sim_time = 20.0
dt_gas     = 2e-4
save_every = 200   # кадр каждые 0.04 ед. времени

frame_idx = 0
while solver_gas.proceed:
    solver_gas.step(dt_gas)

    phi.change_scales(1)
    if not np.all(np.isfinite(phi['g'])):
        logger.error(f"NaN at t={solver_gas.sim_time:.4f}"); break

    # Ренормировка газа каждые 50 шагов — компенсируем численную диссипацию
    if solver_gas.iteration % 50 == 0:
        renorm_gas(phi, GAS_PEAK)

    if solver_gas.iteration % save_every == 0:
        t = solver_gas.sim_time
        gas_peak = float(np.max(np.abs(phi['g'])**2))
        logger.info(f"  t={t:.3f}  gas_peak={gas_peak:.4f}")
        plot_frame(psi_t, phi, f"t={t:.2f}",
                   f"{output_dir}/frame_{frame_idx:04d}.png")
        frame_idx += 1

logger.info(f"Done! {frame_idx} frames in {output_dir}/")
logger.info("Video: ffmpeg -r 20 -i " + output_dir + "/frame_%04d.png -vcodec libx264 torus.mp4")