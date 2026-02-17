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
Lx, Ly, Lz = 24, 24, 24       # Большая область, чтобы газ не упирался в границы
Nx, Ny, Nz = 64, 64, 64       # Разрешение
R_torus = 5.0                  # Большой радиус тора
g = 30.0                       # Нелинейность (умеренная)
omega_trap = 3.0               # Частота ловушки (сильная)
N_particles = 1.0              # Нормировка

# Базис: 3D периодические (Фурье)
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype=np.complex128)
xbasis = d3.ComplexFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=3/2)
ybasis = d3.ComplexFourier(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2), dealias=3/2)
zbasis = d3.ComplexFourier(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=3/2)

# Поля
psi = dist.Field(name='psi', bases=(xbasis, ybasis, zbasis))
V_trap = dist.Field(name='V', bases=(xbasis, ybasis, zbasis))

# Координатные сетки
x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
r_cyl = np.sqrt(x**2 + y**2)  # Цилиндрический радиус

# Потенциал ловушки (удерживает тор)
V_trap['g'] = 0.5 * (omega_trap**2) * ((r_cyl - R_torus)**2 + z**2)

# Нормализация волновой функции
def normalize_psi(field):
    """Нормализовать |psi|^2 -> N_particles"""
    norm = d3.Integrate(field * d3.conj(field)).evaluate()['g'][0, 0, 0]
    if np.real(norm) > 1e-10:
        field['g'] /= np.sqrt(np.real(norm))
        field['g'] *= np.sqrt(N_particles)

# ==============================================
# ФАЗА 1: МНИМОЕ ВРЕМЯ — СТАБИЛИЗАЦИЯ ТОРА
# ==============================================
logger.info("=" * 50)
logger.info("ФАЗА 1: Мнимое время — поиск основного состояния тора")
logger.info("=" * 50)

# Начальное приближение: гауссов тор + шум
psi.change_scales(1)
psi['g'] = np.exp(-0.5 * ((r_cyl - R_torus)**2 + z**2)) \
         + 0.05 * np.random.randn(*psi['g'].shape)

# GPE в мнимом времени: dt(psi) = 0.5*Lap(psi) - V*psi - g*|psi|^2*psi
problem_relax = d3.IVP([psi], namespace=locals())
problem_relax.add_equation(
    "dt(psi) - 0.5*div(grad(psi)) = -V_trap*psi - g*psi*conj(psi)*psi"
)

solver_relax = problem_relax.build_solver(d3.RK222)
solver_relax.stop_sim_time = 2.0  # Достаточно для релаксации
dt_relax = 5e-3

frame_relax = 0
while solver_relax.proceed:
    solver_relax.step(dt_relax)

    # Ренормализация каждые 10 шагов (в мнимом времени норма теряется)
    if solver_relax.iteration % 10 == 0:
        normalize_psi(psi)

    # Визуализация прогресса релаксации
    if solver_relax.iteration % 100 == 0:
        t = solver_relax.sim_time
        logger.info(f"  Relaxation t={t:.2f}")

        psi.change_scales(1)
        dens_xy = np.abs(psi['g'][:, :, Nz//2])**2
        dens_xz = np.abs(psi['g'][:, Ny//2, :])**2

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(dens_xy.T, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2],
                        cmap='inferno', origin='lower')
        axes[0].set_title(f"Relaxation XY t={t:.2f}")
        axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

        axes[1].imshow(dens_xz.T, extent=[-Lx/2, Lx/2, -Lz/2, Lz/2],
                        cmap='inferno', origin='lower')
        axes[1].set_title(f"Relaxation XZ t={t:.2f}")
        axes[1].set_xlabel("x"); axes[1].set_ylabel("z")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/frame_relax_{frame_relax:04d}.png")
        plt.close()
        frame_relax += 1

normalize_psi(psi)
logger.info("Фаза 1 завершена: тор стабилизирован.")

# ==============================================
# ФАЗА 2: ДОБАВЛЕНИЕ ГАЗА + ПОЛОИДАЛЬНАЯ ФАЗА
# ==============================================
logger.info("=" * 50)
logger.info("ФАЗА 2: Добавление распределённого газа + полоидальная фаза насоса")
logger.info("=" * 50)

psi.change_scales(1)

# Сохраняем стабилизированный тор
psi_torus = psi['g'].copy()

# Расстояние от центральной линии тора (трубки)
d_torus = np.sqrt((r_cyl - R_torus)**2 + z**2)

# РАСПРЕДЕЛЁННЫЙ ГАЗ:
# Равномерный слабый газ по всему пространству, с пространственными модуляциями
# для визуализации потока. Газ ослаблен внутри тора (чтобы не мешать структуре).
gas_amplitude = 0.08
gas_modulation = (
    1.0
    + 0.3 * np.sin(2 * np.pi * x / Lx * 3)
    + 0.3 * np.cos(2 * np.pi * z / Lz * 2)
    + 0.2 * np.sin(2 * np.pi * y / Ly * 2) * np.cos(2 * np.pi * z / Lz * 3)
)

# Маска: газ есть везде кроме тела тора (плавный переход)
# При d_torus < 1.5 газ подавлен, при d_torus > 3.0 — полная амплитуда
gas_mask = np.clip((d_torus - 1.5) / 1.5, 0.0, 1.0)

psi_gas = gas_amplitude * gas_modulation * gas_mask

# Суммируем тор + газ
psi['g'] = psi_torus + psi_gas

# ПОЛОИДАЛЬНАЯ ФАЗА (НАСОС):
# Накладываем вихревую фазу в полоидальной плоскости ТОЛЬКО на тор
# theta_pol = arctan2(z, r_cyl - R_torus) — угол вокруг малого сечения тора
# Это создаёт поток: газ втягивается сверху и выбрасывается снизу (или наоборот)
theta_pol = np.arctan2(z, r_cyl - R_torus)
m_pump = 1  # Полоидальный заряд вихря

# Фаза действует сильно на тор и слабо на газ
# (плавная маска: полная фаза в теле тора, ноль далеко от тора)
phase_mask = np.exp(-0.5 * d_torus**2 / 3.0**2)
phase_field = m_pump * theta_pol * phase_mask

psi['g'] *= np.exp(1j * phase_field)

logger.info(f"  Газ добавлен. Амплитуда газа: {gas_amplitude}")
logger.info(f"  Полоидальная фаза: m={m_pump}")

# Сохраняем начальное состояние (Фаза 2)
psi.change_scales(1)
dens_xy = np.abs(psi['g'][:, :, Nz//2])**2
dens_xz = np.abs(psi['g'][:, Ny//2, :])**2
phase_xz = np.angle(psi['g'][:, Ny//2, :])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].imshow(dens_xy.T, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2],
                cmap='inferno', origin='lower')
axes[0].set_title("Initial: Density XY (Top View)")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

im1 = axes[1].imshow(dens_xz.T, extent=[-Lx/2, Lx/2, -Lz/2, Lz/2],
                      cmap='inferno', origin='lower', vmax=0.3)
axes[1].set_title("Initial: Density XZ (Side View)\nGas visible around torus")
axes[1].set_xlabel("x"); axes[1].set_ylabel("z")

axes[2].imshow(phase_xz.T, extent=[-Lx/2, Lx/2, -Lz/2, Lz/2],
                cmap='hsv', origin='lower')
axes[2].set_title("Initial: Phase XZ\nPoloidal winding = pump")
axes[2].set_xlabel("x"); axes[2].set_ylabel("z")

plt.tight_layout()
plt.savefig(f"{output_dir}/initial_state.png")
plt.close()
logger.info("  Начальное состояние сохранено.")

# ==============================================
# ФАЗА 3: РЕАЛЬНАЯ ЭВОЛЮЦИЯ (ДИНАМИКА)
# ==============================================
logger.info("=" * 50)
logger.info("ФАЗА 3: Реальная эволюция — наблюдаем перекачку газа")
logger.info("=" * 50)

# GPE в реальном времени:
# i*dt(psi) = -0.5*Lap(psi) + V*psi + g*|psi|^2*psi
# => dt(psi) = 0.5*i*Lap(psi) - i*V*psi - i*g*|psi|^2*psi
problem_real = d3.IVP([psi], namespace=locals())
problem_real.add_equation(
    "dt(psi) - 0.5*1j*div(grad(psi)) = -1j*V_trap*psi - 1j*g*psi*conj(psi)*psi"
)

solver_real = problem_real.build_solver(d3.RK222)
solver_real.stop_sim_time = 5.0   # Долгая симуляция для наблюдения потока
dt_real = 1e-3                    # Мелкий шаг для стабильности

frame_idx = 0

while solver_real.proceed:
    solver_real.step(dt_real)

    if solver_real.iteration % 100 == 0:
        t = solver_real.sim_time
        logger.info(f"  Real time: t={t:.3f}")

        psi.change_scales(1)
        dens = np.abs(psi['g'])**2

        # Срезы
        dens_xy = dens[:, :, Nz//2]   # Вид сверху (плоскость тора)
        dens_xz = dens[:, Ny//2, :]   # Вид сбоку (разрез через центр)

        # Для количественной оценки потока: средняя плотность выше/ниже тора
        z_grid = z.flatten()
        dens_above = np.mean(dens[:, :, z_grid.flatten() > 2.0])
        dens_below = np.mean(dens[:, :, z_grid.flatten() < -2.0])

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # --- ГРАФИК 1: ВИД СВЕРХУ (XY) ---
        ax1 = axes[0]
        im1 = ax1.imshow(dens_xy.T, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2],
                          cmap='inferno', origin='lower')
        ax1.set_title(f"Top View (XY)  t={t:.2f}")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        plt.colorbar(im1, ax=ax1, fraction=0.046)

        # --- ГРАФИК 2: ВИД СБОКУ (XZ) ---
        ax2 = axes[1]
        # vmax ниже пика тора, чтобы видеть газ
        vmax_side = max(0.05, np.percentile(dens_xz, 95) * 0.5)
        im2 = ax2.imshow(dens_xz.T, extent=[-Lx/2, Lx/2, -Lz/2, Lz/2],
                          cmap='inferno', origin='lower',
                          vmin=0.0, vmax=vmax_side)
        ax2.set_title(f"Side View (XZ)  t={t:.2f}\n"
                       f"ρ_above={dens_above:.4f}  ρ_below={dens_below:.4f}")
        ax2.set_xlabel("x")
        ax2.set_ylabel("z")
        plt.colorbar(im2, ax=ax2, fraction=0.046)

        # Стрелка направления потока (полоидальный)
        ax2.annotate("", xy=(0, -4), xytext=(0, 4),
                      arrowprops=dict(arrowstyle="->", color='cyan',
                                     lw=2, alpha=0.7))
        ax2.text(1, 0, "Pump\nFlow", color='cyan', fontsize=9,
                  ha='left', va='center', alpha=0.8)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/frame_{frame_idx:04d}.png")
        plt.close()
        frame_idx += 1

logger.info("=" * 50)
logger.info("ГОТОВО! Кадры сохранены в: " + output_dir)
logger.info("=" * 50)
