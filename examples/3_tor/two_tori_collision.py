"""
two_tori_collision.py
═══════════════════════════════════════════════════════════════════════
СТОЛКНОВЕНИЕ ДВУХ ВИХРЕВЫХ КОЛЕЦ (ТОРОВ)
3D несжимаемые уравнения Навье–Стокса

ФИЗИКА:
  • Два тороидальных вихря летят НАВСТРЕЧУ друг другу вдоль оси z:
      Тор А: центр z = +Lz/4, циркуляция Γ > 0  → летит в сторону −z
      Тор Б: центр z = −Lz/4, циркуляция Γ < 0  → летит в сторону +z
  • При столкновении кольца растягиваются радиально (закон сохранения
    завихрённости), образуя «блин» из вихревых трубок
  • Радиальное расширение → вторичные эффекты Кельвина–Гельмгольца
  • Наблюдаемые явления:
      1. Кольца сближаются и замедляются (взаимная индукция)
      2. Каждое кольцо расширяется радиально в плоскости столкновения
      3. Завихрённость переходит в хаотическую структуру (при Re>1000)

МЕТОД НАЧАЛЬНОГО УСЛОВИЯ — спектральный Био-Савар:
  Из вихревости ω восстанавливаем скорость через уравнение Пуассона:
    ∇²u = −∇×ω  →  û_k = (ik × ω̂_k) / |k|²
  Два тора: ω_net = ω_A + ω_B  (суперпозиция перед нелинейным шагом)

ВЫВОД (в реальном времени):
  snapshots_collision/*.h5     — поля (h5)
  frames_collision/*.png       — превью (PNG, пишутся параллельно)
  vtk_output_collision/*.pvd   — для ParaView

ЗАПУСК:
  python3 two_tori_collision.py
  mpiexec -n 4 python3 two_tori_collision.py
═══════════════════════════════════════════════════════════════════════
"""
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
SNAP_DIR   = os.path.join(SCRIPT_DIR, "snapshots_collision")
FRAMES_DIR = os.path.join(SCRIPT_DIR, "frames_collision")
VTK_DIR    = os.path.join(SCRIPT_DIR, "vtk_output_collision")
os.makedirs(FRAMES_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════
# §1. ПАРАМЕТРЫ ЗАДАЧИ
# ═══════════════════════════════════════════════════════

# ── Геометрия ───────────────────────────────────────────
Lx = Ly = Lz = 2 * np.pi          # периодический куб
Nx = Ny = Nz = 64                 # 64³ → быстро; 96/128 → красивее
dealias = 3 / 2

# ── Вязкость ────────────────────────────────────────────
Re  = 3000.0   # Число Рейнольдса; при Re>2000 виден богатый распад
nu  = 1.0 / Re

# ── Параметры торов ─────────────────────────────────────
R_ring = 1.0      # большой радиус кольца
a_core = 0.4      # малый радиус ядра (гауссово)
Gamma  = 5.0      # |Γ|: амплитуда циркуляции
# z-позиции колец: A вверху, B внизу
z0_A   = +np.pi * 0.45   # тор A (верхний, летит вниз)
z0_B   = -np.pi * 0.45   # тор B (нижний, летит вверх)

# Знак Γ задаёт направление самодвижения (Кельвин):
#   Γ > 0  → кольцо движется в сторону +z
#   Γ < 0  → кольцо движется в сторону −z
# Тор A сверху → должен лететь ВНИЗ (-z) → Γ < 0
# Тор B снизу → должен лететь ВВЕРХ (+z) → Γ > 0
Gamma_A = -Gamma   # летит ↓ (−z)  — навстречу B
Gamma_B = +Gamma   # летит ↑ (+z)  — навстречу A

# Аналитическая оценка скорости встречного движения (формула Кельвина)
_U_kelvin = abs(Gamma / (4 * np.pi * R_ring)) * (np.log(8 * R_ring / a_core) - 0.25)
_dist     = abs(z0_A - z0_B)   # начальное расстояние между кольцами
_T_coll   = _dist / (2 * _U_kelvin)  # оценка времени до столкновения

# ── Интегратор ──────────────────────────────────────────
# Симулируем до момента «после столкновения» — берём 2.5 × T_coll
stop_sim_time = min(3.5 * _T_coll, 8.0)
max_timestep  = 5e-3
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

x, y, z = dist.local_grids(xbasis, ybasis, zbasis)

# ═══════════════════════════════════════════════════════
# §3. УРАВНЕНИЯ НАВЬЕ–СТОКСА (несжимаемые)
#   ∂u/∂t + (u·∇)u = −∇p + ν∇²u
#   ∇·u = 0
# ═══════════════════════════════════════════════════════
problem = d3.IVP([u, p, tau_p], namespace=locals())
problem.add_equation("dt(u) + grad(p) - nu*lap(u) = -u@grad(u)")
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("integ(p) = 0")

solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = stop_sim_time

# ═══════════════════════════════════════════════════════
# §4. НАЧАЛЬНОЕ УСЛОВИЕ: суперпозиция двух торов
#     Метод: спектральный Био-Савар  û = (ik × ω̂) / |k|²
# ═══════════════════════════════════════════════════════

Nx_g = x.shape[0]
Ny_g = y.shape[1]
Nz_g = z.shape[2]

r_cyl  = np.sqrt(x**2 + y**2)
r_safe = np.maximum(r_cyl, 1e-12)


def torus_vorticity(Gamma_val, z0, x_grid, y_grid, z_grid,
                    R=R_ring, a=a_core):
    """
    Возвращает (omega_x, omega_y) — декартовы компоненты тороидальной
    вихревости одного кольца с циркуляцией Gamma_val, центром (0,0,z0).
    omega_z = 0 для кольца в плоскости XY.
    """
    r_c  = np.sqrt(x_grid**2 + y_grid**2)
    r_s  = np.maximum(r_c, 1e-12)
    phi  = (Gamma_val / (np.pi * a**2)) * np.exp(
        -((r_c - R)**2 + (z_grid - z0)**2) / a**2
    )
    ox = -phi * (y_grid / r_s)
    oy =  phi * (x_grid / r_s)
    return ox, oy


# ── Суперпозиция вихревостей ─────────────────────────────────────
ox_A, oy_A = torus_vorticity(Gamma_A, z0_A, x, y, z)
ox_B, oy_B = torus_vorticity(Gamma_B, z0_B, x, y, z)

omega_x_g = ox_A + ox_B
omega_y_g = oy_A + oy_B
# omega_z_g = 0

# ── Спектральный Био-Савар ────────────────────────────────────────
ox_hat = np.fft.fftn(omega_x_g)
oy_hat = np.fft.fftn(omega_y_g)

kx_arr  = np.fft.fftfreq(Nx_g) * (2 * np.pi * Nx_g / Lx)
ky_arr  = np.fft.fftfreq(Ny_g) * (2 * np.pi * Ny_g / Ly)
kz_arr  = np.fft.fftfreq(Nz_g) * (2 * np.pi * Nz_g / Lz)
KX, KY, KZ = np.meshgrid(kx_arr, ky_arr, kz_arr, indexing='ij')
K2 = KX**2 + KY**2 + KZ**2
K2[0, 0, 0] = 1.0

ux_hat = -1j * KZ * oy_hat / K2
uy_hat =  1j * KZ * ox_hat / K2
uz_hat =  1j * (KX * oy_hat - KY * ox_hat) / K2

ux_hat[0, 0, 0] = uy_hat[0, 0, 0] = uz_hat[0, 0, 0] = 0.0

u['g'][0] = np.real(np.fft.ifftn(ux_hat))
u['g'][1] = np.real(np.fft.ifftn(uy_hat))
u['g'][2] = np.real(np.fft.ifftn(uz_hat))

_u_max0 = float(np.max(np.sqrt(u['g'][0]**2 + u['g'][1]**2 + u['g'][2]**2)))
logger.info(f"Начальный max|u| = {_u_max0:.3f}  (ожидается ≈ {_U_kelvin:.3f})")
logger.info(f"Оценка времени столкновения T_coll ≈ {_T_coll:.2f}")
logger.info(f"Симуляция до t = {stop_sim_time:.2f}")

# ═══════════════════════════════════════════════════════
# §5. ДИАГНОСТИЧЕСКИЕ ВЫРАЖЕНИЯ
# ═══════════════════════════════════════════════════════
omega_field = d3.Curl(u)
speed       = np.sqrt(u @ u)
wmag        = np.sqrt(omega_field @ omega_field)   # |ω| — виден тор!

# ═══════════════════════════════════════════════════════
# §6. ФУНКЦИЯ СОХРАНЕНИЯ КАДРОВ
# ═══════════════════════════════════════════════════════
def _save_frame(frame_idx, t,
                wmag_xz, speed_xz,
                wmag_xy_mid, speed_xy_mid,
                x1d, z1d, y1d,
                T_coll):
    """Сохраняет один PNG-кадр (4 панели)."""
    BG = '#0a0a10'
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), facecolor=BG)
    phase = "до" if t < T_coll else ("≈ СТОЛКНОВЕНИЕ" if t < 1.3 * T_coll else "после")
    fig.suptitle(
        f"Столкновение двух торов   |   t = {t:.3f}  ({phase})  |   Re = {Re:.0f}\n"
        f"Тор A: z₀=+{z0_A:.2f}, Γ=+{Gamma_A}     "
        f"Тор B: z₀={z0_B:.2f}, Γ={Gamma_B}",
        fontsize=12, fontweight='bold', color='white', y=0.98
    )

    clr_accent = '#00d4ff'
    for ax in axes.flat:
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_edgecolor('#2a2a3a')
        ax.tick_params(colors='#777799', labelsize=8)
        ax.xaxis.label.set_color('#aaaacc')
        ax.yaxis.label.set_color('#aaaacc')

    ext_xz = [x1d.min(), x1d.max(), z1d.min(), z1d.max()]
    ext_xy = [x1d.min(), x1d.max(), y1d.min(), y1d.max()]
    kw_xz  = dict(origin='lower', aspect='equal', extent=ext_xz, interpolation='bilinear')
    kw_xy  = dict(origin='lower', aspect='equal', extent=ext_xy, interpolation='bilinear')

    vmax_w = np.percentile(np.abs(wmag_xz), 99.5) + 1e-8

    # ── [0,0] |ω| XZ-срез ─────────────────────────
    im0 = axes[0, 0].imshow(wmag_xz.T, cmap='inferno',
                             vmin=0, vmax=vmax_w, **kw_xz)
    axes[0, 0].set_title("|ω|  завихрённость  (XZ, y=0)", color='white', fontsize=10)
    axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("z")
    # Горизонтальные линии — исходные позиции торов
    axes[0, 0].axhline(z0_A, color=clr_accent, lw=0.6, ls='--', alpha=0.5)
    axes[0, 0].axhline(z0_B, color='#ff6b6b', lw=0.6, ls='--', alpha=0.5)
    cb = plt.colorbar(im0, ax=axes[0, 0]); plt.setp(cb.ax.get_yticklabels(), color='#888888')

    # ── [0,1] |u| XZ-срез ─────────────────────────
    im1 = axes[0, 1].imshow(speed_xz.T, cmap='magma',
                             vmin=0, **kw_xz)
    axes[0, 1].set_title("|u|  скорость  (XZ, y=0)", color='white', fontsize=10)
    axes[0, 1].set_xlabel("x"); axes[0, 1].set_ylabel("z")
    axes[0, 1].axhline(z0_A, color=clr_accent, lw=0.6, ls='--', alpha=0.5)
    axes[0, 1].axhline(z0_B, color='#ff6b6b', lw=0.6, ls='--', alpha=0.5)
    cb = plt.colorbar(im1, ax=axes[0, 1]); plt.setp(cb.ax.get_yticklabels(), color='#888888')

    # ── [1,0] |ω| XY-срез по центру (z=0) ────────
    im2 = axes[1, 0].imshow(wmag_xy_mid.T, cmap='inferno',
                             vmin=0, vmax=vmax_w, **kw_xy)
    axes[1, 0].set_title("|ω|  завихрённость  (XY, z=0  — плоскость столкновения)",
                          color='white', fontsize=10)
    axes[1, 0].set_xlabel("x"); axes[1, 0].set_ylabel("y")
    cb = plt.colorbar(im2, ax=axes[1, 0]); plt.setp(cb.ax.get_yticklabels(), color='#888888')

    # ── [1,1] |u| XY-срез по центру ──────────────
    im3 = axes[1, 1].imshow(speed_xy_mid.T, cmap='magma',
                             vmin=0, **kw_xy)
    axes[1, 1].set_title("|u|  скорость  (XY, z=0  — плоскость столкновения)",
                          color='white', fontsize=10)
    axes[1, 1].set_xlabel("x"); axes[1, 1].set_ylabel("y")
    cb = plt.colorbar(im3, ax=axes[1, 1]); plt.setp(cb.ax.get_yticklabels(), color='#888888')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(FRAMES_DIR, f"frame_{frame_idx:04d}.png")
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════
# §7. ЭКСПОРТ VTK + PVD
# ═══════════════════════════════════════════════════════
def export_vtk(snap_dir=SNAP_DIR, vtk_dir=VTK_DIR, lx=Lx, ly=Ly, lz=Lz):
    """
    Конвертирует HDF5-снапшоты в .vti + collision.pvd для ParaView.

    Рекомендуемые фильтры в ParaView:
      Contour → vorticity_mag  (изоповерхность — видны торы и их деформация)
      StreamTracer → velocity  (линии тока)
    """
    import glob, h5py
    try:
        import pyvista as pv
    except ImportError:
        logger.warning("pyvista не установлена — VTK-экспорт пропущен.")
        logger.warning("Установите: pip install pyvista")
        return

    pvd_path = os.path.join(vtk_dir, "collision.pvd")
    os.makedirs(vtk_dir, exist_ok=True)
    h5_files = sorted(glob.glob(os.path.join(snap_dir, "*.h5")))
    if not h5_files:
        logger.warning(f"Нет .h5 файлов в {snap_dir}"); return

    pvd_entries   = []
    frame_counter = 0
    logger.info(f"VTK-экспорт → {vtk_dir}/")

    for file_path in h5_files:
        with h5py.File(file_path, 'r') as f:
            times            = f['scales/sim_time'][:]
            _, Nx_, Ny_, Nz_ = f['tasks/speed'].shape
            dx = lx / Nx_; dy = ly / Ny_; dz = lz / Nz_

            for i in range(len(times)):
                t = float(times[i])
                grid = pv.ImageData()
                grid.dimensions = (Nx_, Ny_, Nz_)
                grid.spacing    = (dx, dy, dz)
                grid.origin     = (-lx/2, -ly/2, -lz/2)
                grid.point_data["speed"]         = f['tasks/speed'][i].flatten(order="F")
                grid.point_data["vorticity_mag"] = f['tasks/vorticity_mag'][i].flatten(order="F")
                grid.point_data["pressure"]      = f['tasks/pressure'][i].flatten(order="F")
                uv = f['tasks/velocity'][i]
                grid.point_data["velocity"] = np.stack([
                    uv[0].flatten(order="F"),
                    uv[1].flatten(order="F"),
                    uv[2].flatten(order="F"),
                ], axis=1)
                vti_name = f"collision_{frame_counter:04d}.vti"
                grid.save(os.path.join(vtk_dir, vti_name))
                pvd_entries.append((t, vti_name))
                frame_counter += 1

    with open(pvd_path, "w", encoding="utf-8") as pvd:
        pvd.write('<?xml version="1.0"?>\n')
        pvd.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        pvd.write('  <Collection>\n')
        for t_e, vti_name in pvd_entries:
            pvd.write(f'    <DataSet timestep="{t_e:.6f}" group="" part="0" file="{vti_name}"/>\n')
        pvd.write('  </Collection>\n')
        pvd.write('</VTKFile>\n')

    logger.info(f"✅  VTK: {frame_counter} кадров  →  {pvd_path}")


# ═══════════════════════════════════════════════════════
# §8. ГЛАВНЫЙ ЦИКЛ ИНТЕГРИРОВАНИЯ
# ═══════════════════════════════════════════════════════
if __name__ == '__main__':

    # ── HDF5-снапшоты ──────────────────────────────────
    snapshots = solver.evaluator.add_file_handler(
        SNAP_DIR, sim_dt=0.1, max_writes=500
    )
    snapshots.add_task(speed,  name="speed")
    snapshots.add_task(wmag,   name="vorticity_mag")
    snapshots.add_task(p,      name="pressure")
    snapshots.add_task(u,      name="velocity")

    # ── CFL: адаптивный шаг ────────────────────────────
    CFL = d3.CFL(
        solver,
        initial_dt  = max_timestep,
        cadence     = 10,
        safety      = 0.3,
        threshold   = 0.1,
        max_change  = 1.5,
        min_change  = 0.5,
        max_dt      = max_timestep,
    )
    CFL.add_velocity(u)

    # ── Мониторинг ─────────────────────────────────────
    flow = d3.GlobalFlowProperty(solver, cadence=10)
    flow.add_property(speed, name='speed')
    flow.add_property(wmag,  name='omega')

    # ── Сетки для срезов ───────────────────────────────
    x1d = x[:, 0, 0]
    y1d = y[0, :, 0]
    z1d = z[0, 0, :]
    iy0   = int(Ny_g // 2)   # y ≈ 0
    iz_mid = int(Nz_g // 2)  # z ≈ 0 — плоскость столкновения

    logger.info("═══════════════════════════════════════════════════════")
    logger.info(" СТОЛКНОВЕНИЕ ДВУХ ВИХРЕВЫХ КОЛЕЦ — Dedalus v3")
    logger.info(f" R={R_ring}, a={a_core}, |Γ|={Gamma}, Re={Re:.0f}")
    logger.info(f" Тор A: z₀=+{z0_A:.2f}, Γ={Gamma_A}  → летит ↓ (-z)")
    logger.info(f" Тор B: z₀={z0_B:.2f}, Γ=+{Gamma_B}  → летит ↑ (+z)")
    logger.info(f" U_кольца ≈ {_U_kelvin:.3f}  →  T_столкновения ≈ {_T_coll:.2f}")
    logger.info(f" Время симуляции: {stop_sim_time:.2f}")
    logger.info(f" PNG-кадры: {FRAMES_DIR}/")
    logger.info("═══════════════════════════════════════════════════════")

    frame_idx = 0
    executor  = ThreadPoolExecutor(max_workers=2)

    try:
        while solver.proceed:
            dt = CFL.compute_timestep()
            solver.step(dt)

            if (solver.iteration - 1) % 20 == 0:
                t = solver.sim_time
                logger.info(
                    f"it={solver.iteration:6d} | t={t:7.3f} | "
                    f"max|u|={flow.max('speed'):6.3f} | "
                    f"max|ω|={flow.max('omega'):7.2f}"
                )

                # ── Вычисляем поля ──────────────────────────────
                w_ev = wmag.evaluate();  w_ev.change_scales(1)
                s_ev = speed.evaluate(); s_ev.change_scales(1)

                w_g = np.array(w_ev['g'])
                s_g = np.array(s_ev['g'])

                # XZ-срез через y=0
                wmag_xz  = w_g[:, iy0, :]
                speed_xz = s_g[:, iy0, :]

                # XY-срез через z=0 (плоскость столкновения)
                wmag_xy_mid  = w_g[:, :, iz_mid]
                speed_xy_mid = s_g[:, :, iz_mid]

                executor.submit(
                    _save_frame,
                    frame_idx, t,
                    wmag_xz.copy(),      speed_xz.copy(),
                    wmag_xy_mid.copy(),  speed_xy_mid.copy(),
                    x1d, z1d, y1d,
                    _T_coll
                )
                frame_idx += 1

    except Exception:
        logger.exception("Ошибка в главном цикле.")
        raise
    finally:
        executor.shutdown(wait=True)
        solver.log_stats()
        logger.info(f"✅  PNG-кадры: {frame_idx} кадров → {FRAMES_DIR}/")

    # ── Конвертация в VTK/PVD ──────────────────────────
    export_vtk()
