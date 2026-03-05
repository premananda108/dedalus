"""
rotating_tori_interaction.py
═══════════════════════════════════════════════════════════════════════
ДВА ВРАЩАЮЩИХСЯ ТОРА: РАСКРУТКА → СВОБОДНОЕ ДВИЖЕНИЕ → ВЗАИМОДЕЙСТВИЕ
3D несжимаемые уравнения Навье–Стокса + временная сила удержания

ФИЗИКА:
  ФАЗА 1 — РАСКРУТКА (t < t_spinup):
    Релаксационная сила F = γ·(u_target − u) раскручивает торы
    до целевой скорости. Торы набирают вихревость и импульс.

  ФАЗА 2 — ЗАТУХАНИЕ СИЛЫ (t_spinup < t < t_spinup + t_fade):
    Сила плавно выключается (косинусный переход).

  ФАЗА 3 — СВОБОДНЫЙ ПОЛЁТ (t > t_spinup + t_fade):
    Торы движутся только по Навье–Стоксу. Их вращение создаёт
    потоки газа. Потоки двух торов пересекаются → взаимодействие.

  СТАБИЛЬНОСТЬ ПОСЛЕ ОТКЛЮЧЕНИЯ:
    • Высокий Re → медленная вязкая диссипация
    • Тороидальный вихрь — устойчивая структура (как дымовое кольцо)
    • Полоидальный вихрь с нулевой нетто-циркуляцией → нет дрейфа

ВЫВОД:
  frames_rotating_tori/*.png           — PNG-кадры
  snapshots_rotating_tori/*.h5         — HDF5-снапшоты
  vtk_output_rotating_tori/*.pvd       — для ParaView

ЗАПУСК:
  python3 rotating_tori_interaction.py
  mpiexec -n 4 python3 rotating_tori_interaction.py
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
SNAP_DIR   = os.path.join(SCRIPT_DIR, "snapshots_rotating_tori")
FRAMES_DIR = os.path.join(SCRIPT_DIR, "frames_rotating_tori")
VTK_DIR    = os.path.join(SCRIPT_DIR, "vtk_output_rotating_tori")
os.makedirs(FRAMES_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════
# §1. ПАРАМЕТРЫ ЗАДАЧИ
# ═══════════════════════════════════════════════════════

# ── Геометрия ───────────────────────────────────────────
Lx = Ly = Lz = 4 * np.pi          # увеличенный домен
Nx = Ny = Nz = 64                 # 64³; для красоты 96
dealias = 3 / 2

# ── Вязкость ────────────────────────────────────────────
Re  = 10000.0    # ВЫСОКИЙ Re → торы живут долго после отключения силы
nu  = 1.0 / Re

# ── Параметры торов ─────────────────────────────────────
R_ring = 1.2       # большой радиус
a_core = 0.45      # малый радиус ядра

# Расположение: два тора рядом по оси X, центрированы по Z
d_sep   = 3.0 * R_ring    # расстояние между центрами ≈ 3.6
x0_A    = -d_sep / 2
x0_B    = +d_sep / 2

# ── Скорости вращения ──────────────────────────────────
Omega_tor = 10.0   # тороидальное вращение (главный мотор)
Omega_pol = 4.0    # полоидальное вращение (насос)
sign_A = +1
sign_B = -1        # одинаковое направление; -1 для встречного

# ── Фазы симуляции ─────────────────────────────────────
tau_relax = 0.3    # жёсткость удержания
t_spinup  = 0.5    # время раскрутки (достаточно для установления вихря)
t_fade    = 0.25    # время плавного выключения силы
# t > t_spinup + t_fade = 2.0 → свободный полёт

# ── Интегратор ──────────────────────────────────────────
stop_sim_time = 3.0     # долго: видим раскрутку + свободное движение
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

# Поля для силы удержания
gamma_field = dist.Field(name='gamma_field', bases=bases)
u_target    = dist.VectorField(coords, name='u_target', bases=bases)
amp         = dist.Field(name='amp')         # амплитуда силы (1→0)
amp['g']    = 1.0

x, y, z = dist.local_grids(xbasis, ybasis, zbasis)

# ═══════════════════════════════════════════════════════
# §3. ЦЕЛЕВОЕ ПОЛЕ СКОРОСТИ И МАСКА
# ═══════════════════════════════════════════════════════

Nx_g = x.shape[0]
Ny_g = y.shape[1]
Nz_g = z.shape[2]


def compute_torus_velocity(Omega_t, Omega_p, sign, x0,
                           x_g, y_g, z_g, R=R_ring, a=a_core):
    """
    Целевая скорость одного тора: тороидальное + полоидальное вращение.
    Тор центрирован на (x0, 0, 0), ось ê_z.
    """
    x_loc = x_g - x0
    y_loc = y_g
    r_loc  = np.sqrt(x_loc**2 + y_loc**2)
    r_safe = np.maximum(r_loc, 1e-12)

    # s² — расстояние до оси трубки
    s2 = (r_loc - R)**2 + z_g**2
    envelope = np.exp(-s2 / a**2)

    # ── Тороидальная скорость ──────────────────────────
    u_phi = sign * Omega_t * envelope
    ux_tor = -u_phi * (y_loc / r_safe)
    uy_tor =  u_phi * (x_loc / r_safe)

    # ── Полоидальная скорость (нулевая нетто-циркуляция) ─
    s = np.sqrt(s2 + 1e-24)
    dr = r_loc - R
    vpol_mag = sign * Omega_p * (2.0 * s / a) * np.exp(-s2 / a**2)

    s_safe = np.maximum(s, 1e-12)
    ux_pol = -vpol_mag * z_g * (x_loc / r_safe) / s_safe
    uy_pol = -vpol_mag * z_g * (y_loc / r_safe) / s_safe
    uz_pol =  vpol_mag * dr / s_safe

    return (ux_tor + ux_pol, uy_tor + uy_pol, uz_pol)


def compute_torus_mask(x0, x_g, y_g, z_g, R=R_ring, a=a_core):
    """Гауссова маска в ядре тора."""
    x_loc = x_g - x0
    r_loc = np.sqrt(x_loc**2 + y_g**2)
    s2 = (r_loc - R)**2 + z_g**2
    return np.exp(-s2 / a**2)


# ── Маска γ(x) ────────────────────────────────────────
mask_A = compute_torus_mask(x0_A, x, y, z)
mask_B = compute_torus_mask(x0_B, x, y, z)
gamma_field['g'] = (mask_A + mask_B) / tau_relax

# ── Целевая скорость ───────────────────────────────────
ux_A, uy_A, uz_A = compute_torus_velocity(
    Omega_tor, Omega_pol, sign_A, x0_A, x, y, z)
ux_B, uy_B, uz_B = compute_torus_velocity(
    Omega_tor, Omega_pol, sign_B, x0_B, x, y, z)

u_target['g'][0] = ux_A + ux_B
u_target['g'][1] = uy_A + uy_B
u_target['g'][2] = uz_A + uz_B

# ── Начальное условие = целевой профиль ────────────────
u['g'][0] = u_target['g'][0].copy()
u['g'][1] = u_target['g'][1].copy()
u['g'][2] = u_target['g'][2].copy()

_u_max0 = float(np.max(np.sqrt(u['g'][0]**2 + u['g'][1]**2 + u['g'][2]**2)))
logger.info(f"Начальный max|u| = {_u_max0:.3f}")

# ═══════════════════════════════════════════════════════
# §4. УРАВНЕНИЯ НАВЬЕ–СТОКСА + ОТКЛЮЧАЕМАЯ СИЛА
#
#   ∂u/∂t + (u·∇)u = −∇p + ν∇²u + amp · γ(x) · (u_target − u)
#                                     ↑
#                               amp: 1→0 (выключается)
# ═══════════════════════════════════════════════════════
problem = d3.IVP([u, p, tau_p], namespace=locals())
problem.add_equation(
    "dt(u) + grad(p) - nu*lap(u) = "
    "-u@grad(u) + amp * gamma_field * (u_target - u)"
)
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

# ═══════════════════════════════════════════════════════
# §6. ВИЗУАЛИЗАЦИЯ КАДРОВ
# ═══════════════════════════════════════════════════════
def _save_frame(frame_idx, t, current_amp,
                wmag_xz, speed_xz, wmag_xy, speed_xy,
                x1d, z1d, y1d):
    """Сохраняет PNG-кадр с 4 панелями."""
    BG = '#0a0a12'
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), facecolor=BG)

    # Определяем фазу
    t_free = t_spinup + t_fade
    if t < t_spinup:
        phase_str = "РАСКРУТКА"
        phase_clr = '#00ff88'
    elif t < t_free:
        phase_str = f"ЗАТУХАНИЕ СИЛЫ ({current_amp*100:.0f}%)"
        phase_clr = '#ffaa00'
    else:
        phase_str = "СВОБОДНОЕ ДВИЖЕНИЕ"
        phase_clr = '#ff4444'

    fig.suptitle(
        f"Вращающиеся торы   |   t = {t:.3f}   |   [{phase_str}]\n"
        f"Re = {Re:.0f}   |   Ω_tor = {Omega_tor}   Ω_pol = {Omega_pol}   "
        f"|   d = {d_sep:.1f}",
        fontsize=11, fontweight='bold', color=phase_clr, y=0.98
    )

    for ax in axes.flat:
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_edgecolor('#2a2a3a')
        ax.tick_params(colors='#777799', labelsize=8)
        ax.xaxis.label.set_color('#aaaacc')
        ax.yaxis.label.set_color('#aaaacc')

    ext_xz = [x1d.min(), x1d.max(), z1d.min(), z1d.max()]
    ext_xy = [x1d.min(), x1d.max(), y1d.min(), y1d.max()]
    kw_xz = dict(origin='lower', aspect='equal', extent=ext_xz,
                 interpolation='bilinear')
    kw_xy = dict(origin='lower', aspect='equal', extent=ext_xy,
                 interpolation='bilinear')

    vmax_w = np.percentile(np.abs(wmag_xz), 99.5) + 1e-8
    vmax_s = np.percentile(np.abs(speed_xz), 99.5) + 1e-8

    # [0,0] |ω| XZ
    im0 = axes[0, 0].imshow(wmag_xz.T, cmap='inferno',
                             vmin=0, vmax=vmax_w, **kw_xz)
    axes[0, 0].set_title("|ω|  завихрённость  (XZ, y=0)",
                          color='white', fontsize=10)
    axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("z")
    axes[0, 0].axvline(x0_A, color='#00d4ff', lw=0.6, ls='--', alpha=0.4)
    axes[0, 0].axvline(x0_B, color='#ff6b6b', lw=0.6, ls='--', alpha=0.4)
    cb = plt.colorbar(im0, ax=axes[0, 0])
    plt.setp(cb.ax.get_yticklabels(), color='#888888')

    # [0,1] |u| XZ
    im1 = axes[0, 1].imshow(speed_xz.T, cmap='magma',
                             vmin=0, vmax=vmax_s, **kw_xz)
    axes[0, 1].set_title("|u|  скорость  (XZ, y=0)",
                          color='white', fontsize=10)
    axes[0, 1].set_xlabel("x"); axes[0, 1].set_ylabel("z")
    axes[0, 1].axvline(x0_A, color='#00d4ff', lw=0.6, ls='--', alpha=0.4)
    axes[0, 1].axvline(x0_B, color='#ff6b6b', lw=0.6, ls='--', alpha=0.4)
    cb = plt.colorbar(im1, ax=axes[0, 1])
    plt.setp(cb.ax.get_yticklabels(), color='#888888')

    # [1,0] |ω| XY
    vmax_w_xy = np.percentile(np.abs(wmag_xy), 99.5) + 1e-8
    im2 = axes[1, 0].imshow(wmag_xy.T, cmap='inferno',
                             vmin=0, vmax=vmax_w_xy, **kw_xy)
    axes[1, 0].set_title("|ω|  завихрённость  (XY, z=0 — вид сверху)",
                          color='white', fontsize=10)
    axes[1, 0].set_xlabel("x"); axes[1, 0].set_ylabel("y")
    axes[1, 0].axvline(x0_A, color='#00d4ff', lw=0.6, ls='--', alpha=0.4)
    axes[1, 0].axvline(x0_B, color='#ff6b6b', lw=0.6, ls='--', alpha=0.4)
    cb = plt.colorbar(im2, ax=axes[1, 0])
    plt.setp(cb.ax.get_yticklabels(), color='#888888')

    # [1,1] |u| XY
    vmax_s_xy = np.percentile(np.abs(speed_xy), 99.5) + 1e-8
    im3 = axes[1, 1].imshow(speed_xy.T, cmap='magma',
                             vmin=0, vmax=vmax_s_xy, **kw_xy)
    axes[1, 1].set_title("|u|  скорость  (XY, z=0 — вид сверху)",
                          color='white', fontsize=10)
    axes[1, 1].set_xlabel("x"); axes[1, 1].set_ylabel("y")
    axes[1, 1].axvline(x0_A, color='#00d4ff', lw=0.6, ls='--', alpha=0.4)
    axes[1, 1].axvline(x0_B, color='#ff6b6b', lw=0.6, ls='--', alpha=0.4)
    cb = plt.colorbar(im3, ax=axes[1, 1])
    plt.setp(cb.ax.get_yticklabels(), color='#888888')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(FRAMES_DIR, f"frame_{frame_idx:04d}.png")
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════
# §7. ЭКСПОРТ VTK + PVD
# ═══════════════════════════════════════════════════════
def export_vtk(snap_dir=SNAP_DIR, vtk_dir=VTK_DIR,
               lx=Lx, ly=Ly, lz=Lz):
    import glob, h5py
    try:
        import pyvista as pv
    except ImportError:
        logger.warning("pyvista не установлена — VTK-экспорт пропущен.")
        return

    pvd_path = os.path.join(vtk_dir, "rotating_tori.pvd")
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
                t_val = float(times[i])
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
                vti_name = f"rotating_tori_{frame_counter:04d}.vti"
                grid.save(os.path.join(vtk_dir, vti_name))
                pvd_entries.append((t_val, vti_name))
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

    # ── CFL ────────────────────────────────────────────
    CFL = d3.CFL(
        solver, initial_dt=max_timestep, cadence=10,
        safety=0.25, threshold=0.1,
        max_change=1.4, min_change=0.5, max_dt=max_timestep,
    )
    CFL.add_velocity(u)

    # ── Мониторинг ─────────────────────────────────────
    flow = d3.GlobalFlowProperty(solver, cadence=10)
    flow.add_property(speed, name='speed')
    flow.add_property(wmag,  name='omega')

    x1d = x[:, 0, 0]
    y1d = y[0, :, 0]
    z1d = z[0, 0, :]
    iy0    = int(Ny_g // 2)
    iz_mid = int(Nz_g // 2)

    t_free = t_spinup + t_fade

    logger.info("═══════════════════════════════════════════════════════")
    logger.info(" ВРАЩАЮЩИЕСЯ ТОРЫ: РАСКРУТКА → СВОБОДНОЕ ДВИЖЕНИЕ")
    logger.info(f" R={R_ring}, a={a_core}, Re={Re:.0f}")
    logger.info(f" Тор A: x₀={x0_A:.2f}   Тор B: x₀={x0_B:.2f}")
    logger.info(f" Ω_tor={Omega_tor}, Ω_pol={Omega_pol}")
    logger.info(f" Раскрутка: 0 → {t_spinup:.1f}")
    logger.info(f" Затухание силы: {t_spinup:.1f} → {t_free:.1f}")
    logger.info(f" Свободный полёт: {t_free:.1f} → {stop_sim_time:.1f}")
    logger.info(f" PNG-кадры: {FRAMES_DIR}/")
    logger.info("═══════════════════════════════════════════════════════")

    frame_idx = 0
    executor  = ThreadPoolExecutor(max_workers=2)

    try:
        while solver.proceed:
            t = solver.sim_time

            # ── ЛОГИКА ОТКЛЮЧЕНИЯ СИЛЫ ─────────────────
            if t < t_spinup:
                current_amp = 1.0
            elif t < t_free:
                # Плавный косинусный переход 1 → 0
                phase = np.pi * (t - t_spinup) / t_fade
                current_amp = 0.5 * (1.0 + np.cos(phase))
            else:
                current_amp = 0.0

            amp['g'] = current_amp
            # ───────────────────────────────────────────

            dt = CFL.compute_timestep()
            solver.step(dt)

            if (solver.iteration - 1) % 20 == 0:
                max_speed = flow.max('speed')
                max_omega = flow.max('omega')

                # Фаза для лога
                if t < t_spinup:
                    ph = "РАСКРУТКА"
                elif t < t_free:
                    ph = f"ЗАТУХАНИЕ({current_amp:.2f})"
                else:
                    ph = "СВОБОДНЫЙ"

                logger.info(
                    f"it={solver.iteration:5d} | t={t:6.3f} | "
                    f"{ph:20s} | max|u|={max_speed:6.3f} | "
                    f"max|ω|={max_omega:7.2f}"
                )

                # ── Вычисляем поля для кадра ───────────
                w_ev = wmag.evaluate();  w_ev.change_scales(1)
                s_ev = speed.evaluate(); s_ev.change_scales(1)

                w_g = np.array(w_ev['g'])
                s_g = np.array(s_ev['g'])

                executor.submit(
                    _save_frame, frame_idx, t, current_amp,
                    w_g[:, iy0, :].copy(),   s_g[:, iy0, :].copy(),
                    w_g[:, :, iz_mid].copy(), s_g[:, :, iz_mid].copy(),
                    x1d, z1d, y1d
                )
                frame_idx += 1

    except Exception:
        logger.exception("Ошибка в главном цикле.")
        raise
    finally:
        executor.shutdown(wait=True)
        solver.log_stats()
        logger.info(f"✅  PNG-кадры: {frame_idx} → {FRAMES_DIR}/")

    # ── VTK/PVD ────────────────────────────────────────
    export_vtk()
