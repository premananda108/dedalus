"""
rotating_tori_interaction.py
═══════════════════════════════════════════════════════════════════════
ДВА СТАБИЛЬНЫХ ВРАЩАЮЩИХСЯ ТОРА, ВЗАИМОДЕЙСТВУЮЩИХ ЧЕРЕЗ ГАЗОВЫЕ ПОТОКИ
3D несжимаемые уравнения Навье–Стокса

ФИЗИКА:
  • Каждый тор — вращающийся «маховик»:
      – Тороидальное вращение (u_φ вокруг оси z) — основной мотор,
        создаёт центробежный «ветер» в экваториальной плоскости
      – Полоидальное вращение (внутри трубки тора) — создаёт
        перпендикулярные потоки, «перекачивая» газ вдоль оси z

  • СТАБИЛЬНОСТЬ: полоидальная вихревость имеет нулевую нетто-
    циркуляцию (∫ω·dA = 0), поэтому тор НЕ самодвижется (нет
    дрейфа Кельвина). Он остаётся на месте и вращается.

  • МАССИВНОСТЬ: высокая скорость вращения → большая кинетическая
    энергия → эффективная инерция/масса тора.

  • ВЗАИМОДЕЙСТВИЕ: газовые потоки от одного тора достигают
    другого → непрямое динамическое взаимодействие через среду.

РАСПОЛОЖЕНИЕ:
  Тор A: центр (−d/2, 0, 0), ось вращения ê_z
  Тор B: центр (+d/2, 0, 0), ось вращения ê_z
  Оба вращаются в одну сторону (можно поменять знак).

ВЫВОД:
  frames_rotating_tori/*.png       — PNG-кадры
  snapshots_rotating_tori/*.h5     — HDF5-снапшоты
  vtk_output_rotating_tori/*.pvd   — для ParaView

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
Lx = Ly = Lz = 4 * np.pi          # увеличенный домен для потоков
Nx = Ny = Nz = 64                 # 64³; 96→красивее
dealias = 3 / 2

# ── Вязкость ────────────────────────────────────────────
Re  = 3000.0    # высокий Re → вихри живут долго
nu  = 1.0 / Re

# ── Параметры торов ─────────────────────────────────────
R_ring = 1.2       # большой радиус кольца (ось тора)
a_core = 0.45      # малый радиус ядра

# Расстояние между центрами торов
d_sep   = 3.0 * R_ring    # ≈ 3.6 — торы рядом, потоки пересекаются
x0_A    = -d_sep / 2      # центр тора A
x0_B    = +d_sep / 2      # центр тора B

# ── Скорости вращения ──────────────────────────────────
# Тороидальное вращение (вокруг оси z, вдоль большого кольца)
# Это главный «мотор» — создаёт центробежные потоки
Omega_tor = 8.0    # амплитуда азимутальной скорости (высокая!)

# Полоидальное вращение (внутри трубки тора)
# Создаёт дополнительный «насосный» эффект
Omega_pol = 5.0    # амплитуда полоидальной циркуляции

# Знаки: оба вращаются в одном направлении → согласованные потоки
# Поменяйте sign_B на -1 для противоположного вращения
sign_A = +1
sign_B = +1

# ── Интегратор ──────────────────────────────────────────
stop_sim_time = 3.0     # долгая симуляция — наблюдаем взаимодействие
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
# §4. НАЧАЛЬНОЕ УСЛОВИЕ: два стабильных вращающихся тора
# ═══════════════════════════════════════════════════════

Nx_g = x.shape[0]
Ny_g = y.shape[1]
Nz_g = z.shape[2]


def torus_toroidal_velocity(Omega, sign, x0, x_g, y_g, z_g,
                            R=R_ring, a=a_core):
    """
    Тороидальная скорость: вращение газа вокруг оси z (ê_φ)
    в ядре тора, центр которого сдвинут по x на x0.

    u_φ(r,z) = sign · Omega · exp(−((r_loc − R)² + z_g²) / a²)

    Декартово разложение:  u_x = −u_φ · y_loc/r_loc
                           u_y = +u_φ · x_loc/r_loc
    """
    x_loc = x_g - x0     # координаты относительно центра тора
    y_loc = y_g           # тор центрирован по y=0
    r_loc  = np.sqrt(x_loc**2 + y_loc**2)
    r_safe = np.maximum(r_loc, 1e-12)

    # Расстояние до оси трубки тора
    envelope = np.exp(-((r_loc - R)**2 + z_g**2) / a**2)

    u_phi = sign * Omega * envelope
    ux = -u_phi * (y_loc / r_safe)
    uy =  u_phi * (x_loc / r_safe)
    uz = np.zeros_like(ux)
    return ux, uy, uz


def torus_poloidal_vorticity(Omega, sign, x0, x_g, y_g, z_g,
                             R=R_ring, a=a_core):
    """
    Полоидальная вихревость: вращение газа ВНУТРИ трубки тора.

    Профиль ω_pol(s) = sign · Omega · (1 − 2s²/a²) · exp(−s²/a²)
    где s = √((r_loc − R)² + z²) — расстояние до оси трубки.

    Этот профиль даёт ∫₀^∞ ω_pol(s)·s·ds = 0,
    т.е. нулевую нетто-циркуляцию → НЕТ самодвижения Кельвина!

    Вихревость направлена тороидально: ω = ω_pol · ê_φ
    → ω_x = −ω_pol · y_loc/r_loc
      ω_y = +ω_pol · x_loc/r_loc
      ω_z = 0
    """
    x_loc = x_g - x0
    y_loc = y_g
    r_loc  = np.sqrt(x_loc**2 + y_loc**2)
    r_safe = np.maximum(r_loc, 1e-12)

    s2 = (r_loc - R)**2 + z_g**2   # s² — расстояние до оси трубки
    profile = sign * Omega * (1.0 - 2.0 * s2 / a**2) * np.exp(-s2 / a**2)

    ox = -profile * (y_loc / r_safe)
    oy =  profile * (x_loc / r_safe)
    oz = np.zeros_like(ox)
    return ox, oy, oz


# ── Тороидальная скорость (прямое задание, div-free) ────────────
ux_A, uy_A, uz_A = torus_toroidal_velocity(Omega_tor, sign_A, x0_A, x, y, z)
ux_B, uy_B, uz_B = torus_toroidal_velocity(Omega_tor, sign_B, x0_B, x, y, z)

u['g'][0] = ux_A + ux_B
u['g'][1] = uy_A + uy_B
u['g'][2] = uz_A + uz_B

# ── Полоидальная скорость: из вихревости через Био-Савар ────────
ox_A, oy_A, oz_A = torus_poloidal_vorticity(Omega_pol, sign_A, x0_A, x, y, z)
ox_B, oy_B, oz_B = torus_poloidal_vorticity(Omega_pol, sign_B, x0_B, x, y, z)

omega_x_total = ox_A + ox_B
omega_y_total = oy_A + oy_B
# omega_z_total = 0

# ── Спектральный Био-Савар: ω → u_poloidal ────────────────────
ox_hat = np.fft.fftn(omega_x_total)
oy_hat = np.fft.fftn(omega_y_total)

kx_arr = np.fft.fftfreq(Nx_g) * (2 * np.pi * Nx_g / Lx)
ky_arr = np.fft.fftfreq(Ny_g) * (2 * np.pi * Ny_g / Ly)
kz_arr = np.fft.fftfreq(Nz_g) * (2 * np.pi * Nz_g / Lz)
KX, KY, KZ = np.meshgrid(kx_arr, ky_arr, kz_arr, indexing='ij')
K2 = KX**2 + KY**2 + KZ**2
K2[0, 0, 0] = 1.0

# û = (ik × ω̂) / |k|²,  при ω_z = 0:
ux_hat = -1j * KZ * oy_hat / K2
uy_hat =  1j * KZ * ox_hat / K2
uz_hat =  1j * (KX * oy_hat - KY * ox_hat) / K2

ux_hat[0, 0, 0] = uy_hat[0, 0, 0] = uz_hat[0, 0, 0] = 0.0

# Суперпозиция: тороидальная + полоидальная скорости
u['g'][0] += np.real(np.fft.ifftn(ux_hat))
u['g'][1] += np.real(np.fft.ifftn(uy_hat))
u['g'][2] += np.real(np.fft.ifftn(uz_hat))

# ── Диагностика начального поля ────────────────────────────────
_u_max0 = float(np.max(np.sqrt(u['g'][0]**2 + u['g'][1]**2 + u['g'][2]**2)))
_KE0    = float(0.5 * np.mean(u['g'][0]**2 + u['g'][1]**2 + u['g'][2]**2))
logger.info(f"Начальный max|u| = {_u_max0:.3f}")
logger.info(f"Начальная кинетическая энергия <½u²> = {_KE0:.3f}")
logger.info(f"  Тороидальное вращение Ω_tor = {Omega_tor}")
logger.info(f"  Полоидальное вращение Ω_pol = {Omega_pol}")
logger.info(f"  Расстояние между торами d = {d_sep:.2f}")

# ═══════════════════════════════════════════════════════
# §5. ДИАГНОСТИЧЕСКИЕ ВЫРАЖЕНИЯ
# ═══════════════════════════════════════════════════════
omega_field = d3.Curl(u)
speed       = np.sqrt(u @ u)
wmag        = np.sqrt(omega_field @ omega_field)

# ═══════════════════════════════════════════════════════
# §6. ФУНКЦИЯ СОХРАНЕНИЯ КАДРОВ
# ═══════════════════════════════════════════════════════
def _save_frame(frame_idx, t,
                wmag_xz, speed_xz,
                wmag_xy, speed_xy,
                x1d, z1d, y1d):
    """Сохраняет один PNG-кадр (4 панели)."""
    BG = '#0a0a12'
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), facecolor=BG)
    fig.suptitle(
        f"Вращающиеся торы — газовое взаимодействие   |   "
        f"t = {t:.3f}   |   Re = {Re:.0f}\n"
        f"Ω_tor = {Omega_tor}   Ω_pol = {Omega_pol}   "
        f"d = {d_sep:.1f}   R = {R_ring}",
        fontsize=11, fontweight='bold', color='white', y=0.98
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

    # ── [0,0] |ω| XZ-срез (боковой вид) ─────────────
    im0 = axes[0, 0].imshow(wmag_xz.T, cmap='inferno',
                             vmin=0, vmax=vmax_w, **kw_xz)
    axes[0, 0].set_title("|ω|  завихрённость  (XZ, y=0)",
                          color='white', fontsize=10)
    axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("z")
    # Вертикальные линии — позиции центров торов
    axes[0, 0].axvline(x0_A, color='#00d4ff', lw=0.6, ls='--', alpha=0.5)
    axes[0, 0].axvline(x0_B, color='#ff6b6b', lw=0.6, ls='--', alpha=0.5)
    cb = plt.colorbar(im0, ax=axes[0, 0])
    plt.setp(cb.ax.get_yticklabels(), color='#888888')

    # ── [0,1] |u| XZ-срез ─────────────────────────
    im1 = axes[0, 1].imshow(speed_xz.T, cmap='magma',
                             vmin=0, vmax=vmax_s, **kw_xz)
    axes[0, 1].set_title("|u|  скорость  (XZ, y=0)",
                          color='white', fontsize=10)
    axes[0, 1].set_xlabel("x"); axes[0, 1].set_ylabel("z")
    axes[0, 1].axvline(x0_A, color='#00d4ff', lw=0.6, ls='--', alpha=0.5)
    axes[0, 1].axvline(x0_B, color='#ff6b6b', lw=0.6, ls='--', alpha=0.5)
    cb = plt.colorbar(im1, ax=axes[0, 1])
    plt.setp(cb.ax.get_yticklabels(), color='#888888')

    # ── [1,0] |ω| XY-срез (вид сверху, z=0) ──────
    vmax_w_xy = np.percentile(np.abs(wmag_xy), 99.5) + 1e-8
    im2 = axes[1, 0].imshow(wmag_xy.T, cmap='inferno',
                             vmin=0, vmax=vmax_w_xy, **kw_xy)
    axes[1, 0].set_title("|ω|  завихрённость  (XY, z=0 — вид сверху)",
                          color='white', fontsize=10)
    axes[1, 0].set_xlabel("x"); axes[1, 0].set_ylabel("y")
    axes[1, 0].axvline(x0_A, color='#00d4ff', lw=0.6, ls='--', alpha=0.5)
    axes[1, 0].axvline(x0_B, color='#ff6b6b', lw=0.6, ls='--', alpha=0.5)
    cb = plt.colorbar(im2, ax=axes[1, 0])
    plt.setp(cb.ax.get_yticklabels(), color='#888888')

    # ── [1,1] |u| XY-срез ────────────────────────
    vmax_s_xy = np.percentile(np.abs(speed_xy), 99.5) + 1e-8
    im3 = axes[1, 1].imshow(speed_xy.T, cmap='magma',
                             vmin=0, vmax=vmax_s_xy, **kw_xy)
    axes[1, 1].set_title("|u|  скорость  (XY, z=0 — вид сверху)",
                          color='white', fontsize=10)
    axes[1, 1].set_xlabel("x"); axes[1, 1].set_ylabel("y")
    axes[1, 1].axvline(x0_A, color='#00d4ff', lw=0.6, ls='--', alpha=0.5)
    axes[1, 1].axvline(x0_B, color='#ff6b6b', lw=0.6, ls='--', alpha=0.5)
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
    """
    Конвертирует HDF5-снапшоты в .vti + rotating_tori.pvd для ParaView.
    """
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
                vti_name = f"rotating_tori_{frame_counter:04d}.vti"
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
        safety      = 0.25,       # чуть консервативнее для высоких Ω
        threshold   = 0.1,
        max_change  = 1.4,
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
    iy0    = int(Ny_g // 2)    # y ≈ 0
    iz_mid = int(Nz_g // 2)    # z ≈ 0

    logger.info("═══════════════════════════════════════════════════════")
    logger.info(" ДВА СТАБИЛЬНЫХ ВРАЩАЮЩИХСЯ ТОРА — Dedalus v3")
    logger.info(f" R={R_ring}, a={a_core}, Re={Re:.0f}")
    logger.info(f" Тор A: x₀={x0_A:.2f},  знак={sign_A:+d}")
    logger.info(f" Тор B: x₀={x0_B:.2f},  знак={sign_B:+d}")
    logger.info(f" Ω_tor={Omega_tor}, Ω_pol={Omega_pol}")
    logger.info(f" Расстояние d = {d_sep:.2f}")
    logger.info(f" Время симуляции: {stop_sim_time:.1f}")
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
                max_speed = flow.max('speed')
                max_omega = flow.max('omega')

                logger.info(
                    f"it={solver.iteration:6d} | t={t:7.3f} | "
                    f"max|u|={max_speed:6.3f} | "
                    f"max|ω|={max_omega:7.2f}"
                )

                # ── Вычисляем поля ──────────────────────────────
                w_ev = wmag.evaluate();  w_ev.change_scales(1)
                s_ev = speed.evaluate(); s_ev.change_scales(1)

                w_g = np.array(w_ev['g'])
                s_g = np.array(s_ev['g'])

                # XZ-срез через y=0
                wmag_xz  = w_g[:, iy0, :]
                speed_xz = s_g[:, iy0, :]

                # XY-срез через z=0 (вид сверху — видны оба тора)
                wmag_xy  = w_g[:, :, iz_mid]
                speed_xy = s_g[:, :, iz_mid]

                executor.submit(
                    _save_frame,
                    frame_idx, t,
                    wmag_xz.copy(),  speed_xz.copy(),
                    wmag_xy.copy(),  speed_xy.copy(),
                    x1d, z1d, y1d
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
