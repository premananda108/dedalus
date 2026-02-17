import os
# Ограничиваем потоки для скорости
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"

import numpy as np
import dedalus.public as d3
import logging
import matplotlib.pyplot as plt

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

output_dir = 'examples/1_tor/frames_pump_v2'
os.makedirs(output_dir, exist_ok=True)

# ==============================================
# ПАРАМЕТРЫ
# ==============================================
Lx, Ly, Lz = 24, 24, 24
Nx, Ny, Nz = 64, 64, 64
R_torus = 5.0
g = 30.0
omega_trap = 3.0
DESIRED_PEAK = 2.0  # Желаемая пиковая плотность |psi|^2 тора

# Базис
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype=np.complex128)
xbasis = d3.ComplexFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=3/2)
ybasis = d3.ComplexFourier(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2), dealias=3/2)
zbasis = d3.ComplexFourier(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=3/2)

psi = dist.Field(name='psi', bases=(xbasis, ybasis, zbasis))
V_trap = dist.Field(name='V', bases=(xbasis, ybasis, zbasis))

x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
r_cyl = np.sqrt(x**2 + y**2)

# Потенциал ловушки
V_trap['g'] = 0.5 * (omega_trap**2) * ((r_cyl - R_torus)**2 + z**2)

def rescale_to_peak(field, target_peak):
    """Масштабировать psi так, чтобы max(|psi|^2) = target_peak"""
    field.change_scales(1)
    current_peak = np.max(np.abs(field['g'])**2)
    if current_peak > 1e-30:
        field['g'] *= np.sqrt(target_peak / current_peak)
    logger.info(f"  Rescaled: peak |psi|^2 was {current_peak:.6f}, now {target_peak:.2f}")

# ==============================================
# ФАЗА 1: МНИМОЕ ВРЕМЯ — СТАБИЛИЗАЦИЯ ТОРА
# ==============================================
logger.info("=" * 50)
logger.info("PHASE 1: Imaginary time relaxation")
logger.info("=" * 50)

# Начальное приближение
psi.change_scales(1)
psi['g'] = np.sqrt(DESIRED_PEAK) * np.exp(-0.5 * ((r_cyl - R_torus)**2 + z**2))

# GPE в мнимом времени
problem_relax = d3.IVP([psi], namespace=locals())
problem_relax.add_equation(
    "dt(psi) - 0.5*div(grad(psi)) = -V_trap*psi - g*psi*conj(psi)*psi"
)

solver_relax = problem_relax.build_solver(d3.RK222)
solver_relax.stop_sim_time = 2.0
dt_relax = 5e-3

frame_relax = 0
while solver_relax.proceed:
    solver_relax.step(dt_relax)

    # Ренормализация: поддерживаем пиковую плотность
    if solver_relax.iteration % 10 == 0:
        rescale_to_peak(psi, DESIRED_PEAK)

    # Визуализация
    if solver_relax.iteration % 100 == 0:
        t = solver_relax.sim_time
        psi.change_scales(1)
        dens = np.abs(psi['g'])**2
        dens_xy = dens[:, :, Nz//2]
        dens_xz = dens[:, Ny//2, :]
        peak = np.max(dens)
        logger.info(f"  Relax t={t:.2f}, peak density={peak:.4f}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].imshow(dens_xy.T, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2],
                        cmap='inferno', origin='lower', vmin=0, vmax=DESIRED_PEAK)
        axes[0].set_title(f"Relax XY t={t:.2f}, peak={peak:.3f}")
        axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

        axes[1].imshow(dens_xz.T, extent=[-Lx/2, Lx/2, -Lz/2, Lz/2],
                        cmap='inferno', origin='lower', vmin=0, vmax=DESIRED_PEAK)
        axes[1].set_title(f"Relax XZ t={t:.2f}")
        axes[1].set_xlabel("x"); axes[1].set_ylabel("z")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/frame_relax_{frame_relax:04d}.png")
        plt.close()
        frame_relax += 1

rescale_to_peak(psi, DESIRED_PEAK)
logger.info("Phase 1 done: torus stabilized.")

# ==============================================
# ФАЗА 2: ДОБАВЛЕНИЕ ГАЗА + ПОЛОИДАЛЬНАЯ ФАЗА
# ==============================================
logger.info("=" * 50)
logger.info("PHASE 2: Adding gas + poloidal pump phase")
logger.info("=" * 50)

psi.change_scales(1)
psi_torus = psi['g'].copy()

# Расстояние от трубки тора
d_torus = np.sqrt((r_cyl - R_torus)**2 + z**2)

# Газ: 10% от пиковой плотности тора, с модуляциями для визуализации
gas_density_fraction = 0.10
gas_amp = np.sqrt(DESIRED_PEAK * gas_density_fraction)

gas_modulation = (
    1.0
    + 0.4 * np.sin(2 * np.pi * z / Lz * 3)
    + 0.3 * np.cos(2 * np.pi * x / Lx * 2)
    + 0.2 * np.sin(2 * np.pi * y / Ly * 2)
)

# Маска: газ вне тела тора (плавный переход)
gas_mask = np.clip((d_torus - 1.5) / 2.0, 0.0, 1.0)

psi_gas = gas_amp * gas_modulation * gas_mask
psi['g'] = psi_torus + psi_gas

# Полоидальная фаза (насос)
theta_pol = np.arctan2(z, r_cyl - R_torus)
m_pump = 1

# Фаза сильная на торе, затухает вдали
phase_mask = np.exp(-0.5 * d_torus**2 / 4.0**2)
phase_field = m_pump * theta_pol * phase_mask
psi['g'] *= np.exp(1j * phase_field)

# Логируем реальные значения
psi.change_scales(1)
dens = np.abs(psi['g'])**2
logger.info(f"  Peak density after Phase 2: {np.max(dens):.4f}")
logger.info(f"  Mean density: {np.mean(dens):.6f}")
logger.info(f"  Gas amplitude: {gas_amp:.4f} (|psi_gas|^2 ~ {gas_amp**2:.4f})")

# Сохраняем начальное состояние
dens_xy = dens[:, :, Nz//2]
dens_xz = dens[:, Ny//2, :]
phase_xz = np.angle(psi['g'][:, Ny//2, :])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

im0 = axes[0].imshow(dens_xy.T, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2],
                      cmap='inferno', origin='lower', vmin=0, vmax=DESIRED_PEAK)
axes[0].set_title("Initial: Density XY (top)")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
plt.colorbar(im0, ax=axes[0], fraction=0.046)

# Боковой вид: vmax низкий, чтобы видеть газ
side_vmax = DESIRED_PEAK * 0.15
im1 = axes[1].imshow(dens_xz.T, extent=[-Lx/2, Lx/2, -Lz/2, Lz/2],
                      cmap='inferno', origin='lower', vmin=0, vmax=side_vmax)
axes[1].set_title(f"Initial: Density XZ (side, vmax={side_vmax:.2f})")
axes[1].set_xlabel("x"); axes[1].set_ylabel("z")
plt.colorbar(im1, ax=axes[1], fraction=0.046)

im2 = axes[2].imshow(phase_xz.T, extent=[-Lx/2, Lx/2, -Lz/2, Lz/2],
                      cmap='hsv', origin='lower')
axes[2].set_title("Initial: Phase XZ (poloidal winding)")
axes[2].set_xlabel("x"); axes[2].set_ylabel("z")
plt.colorbar(im2, ax=axes[2], fraction=0.046)

plt.tight_layout()
plt.savefig(f"{output_dir}/initial_state.png")
plt.close()
logger.info("  Initial state saved.")

# ==============================================
# ФАЗА 3: РЕАЛЬНАЯ ЭВОЛЮЦИЯ
# ==============================================
logger.info("=" * 50)
logger.info("PHASE 3: Real-time dynamics")
logger.info("=" * 50)

problem_real = d3.IVP([psi], namespace=locals())
problem_real.add_equation(
    "dt(psi) - 0.5*1j*div(grad(psi)) = -1j*V_trap*psi - 1j*g*psi*conj(psi)*psi"
)

solver_real = problem_real.build_solver(d3.RK222)
solver_real.stop_sim_time = 5.0
dt_real = 1e-3

frame_idx = 0

while solver_real.proceed:
    solver_real.step(dt_real)

    if solver_real.iteration % 100 == 0:
        t = solver_real.sim_time
        psi.change_scales(1)
        dens = np.abs(psi['g'])**2

        peak = np.max(dens)
        dens_xy = dens[:, :, Nz//2]
        dens_xz = dens[:, Ny//2, :]

        logger.info(f"  t={t:.3f}, peak={peak:.4f}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Вид сверху
        im1 = axes[0].imshow(dens_xy.T, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2],
                              cmap='inferno', origin='lower',
                              vmin=0, vmax=DESIRED_PEAK)
        axes[0].set_title(f"Top View (XY)  t={t:.2f}  peak={peak:.2f}")
        axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
        plt.colorbar(im1, ax=axes[0], fraction=0.046)

        # Вид сбоку — vmax = 15% от пика, чтобы видеть газ
        side_vmax = DESIRED_PEAK * 0.15
        im2 = axes[1].imshow(dens_xz.T, extent=[-Lx/2, Lx/2, -Lz/2, Lz/2],
                              cmap='inferno', origin='lower',
                              vmin=0, vmax=side_vmax)
        axes[1].set_title(f"Side View (XZ)  t={t:.2f}")
        axes[1].set_xlabel("x"); axes[1].set_ylabel("z")
        plt.colorbar(im2, ax=axes[1], fraction=0.046)

        # Стрелка потока
        axes[1].annotate("", xy=(0, -4), xytext=(0, 4),
                          arrowprops=dict(arrowstyle="->", color='cyan',
                                         lw=2, alpha=0.7))

        plt.tight_layout()
        plt.savefig(f"{output_dir}/frame_{frame_idx:04d}.png")
        plt.close()
        frame_idx += 1

logger.info("DONE! Frames saved to: " + output_dir)
