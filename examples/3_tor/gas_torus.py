"""
gas_torus.py
═══════════════════════════════════════════════════════════════════════
КОЛЬЦЕВОЙ ВИХРЬ (ТОР) В ГАЗОВОЙ СРЕДЕ
3D несжимаемые уравнения Навье–Стокса

ФИЗИКА:
  • Тороидальная вихревость ω = ω_φ·ê_φ сконцентрирована в кольцевом ядре
  • Кольцо самодвижется вдоль оси z по закону Кельвина (smoke ring):
      U ≈ (Γ/4πR)·[ln(8R/a) − 1/4]
  • При высоком Re кольцо сохраняет форму и проходит сквозь домен
  • В периодическом боксе кольцо встречает своё зеркальное изображение →
    эффект взаимного прохождения (leapfrogging)

МЕТОД НАЧАЛЬНОГО УСЛОВИЯ — спектральный Био-Савар:
  Из вихревости ω восстанавливаем скорость через уравнение Пуассона:
    ∇²u = −∇×ω  →  û_k = (ik × ω̂_k) / |k|²
  Тороидальная вихревость (гаусс на кольце):
    ω_φ = (Γ/πa²)·exp[−((r−R)² + (z−z₀)²)/a²]
  Декартовы компоненты: ω_x = −ω_φ·sinθ,  ω_y = ω_φ·cosθ,  ω_z = 0

ВЫВОД (в реальном времени):
  snapshots_gas_torus/*.h5        — поля (h5)
  frames_gas_torus/*.png          — превью (PNG, пишутся параллельно)
  vtk_output_gas_torus/gas_torus.pvd  — для ParaView

ЗАПУСК:
  python3 gas_torus.py
  mpiexec -n 4 python3 gas_torus.py
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
SNAP_DIR   = os.path.join(SCRIPT_DIR, "snapshots_gas_torus")
FRAMES_DIR = os.path.join(SCRIPT_DIR, "frames_gas_torus")
VTK_DIR    = os.path.join(SCRIPT_DIR, "vtk_output_gas_torus")
os.makedirs(FRAMES_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════
# §1. ПАРАМЕТРЫ ЗАДАЧИ
# ═══════════════════════════════════════════════════════

# ── Геометрия ───────────────────────────────────────────
Lx = Ly = Lz = 2 * np.pi          # периодический куб
Nx = Ny = Nz = 64                 # 64³ мод (128→красивее, медленнее)
dealias = 3 / 2

# ── Вязкость ────────────────────────────────────────────
Re = 4000.0    # Число Рейнольдса (> 2000 — кольцо живёт без диссипации)
nu = 1.0 / Re

# ── Параметры тора (вихревого кольца) ──────────────────
R_ring = 1.2     # большой радиус кольца (ось тора)
a_core = 0.5    # малый радиус ядра вихря  (a < R  — тонкое кольцо)
Gamma  = 6.0     # циркуляция Γ — задаёт скорость самодвижения кольца
z0_ring = -np.pi * 0.4   # начальная z-позиция кольца

# Аналитическая оценка скорости самодвижения (формула Кельвина):
_U_ring = (Gamma / (4 * np.pi * R_ring)) * (np.log(8 * R_ring / a_core) - 0.25)

# ── Интегратор ──────────────────────────────────────────
stop_sim_time = 2.0     # t_cross = Lz/U ≈ 6.28/1.35 ≈ 4.65 → увидим ≥1 цикл
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
# §4. НАЧАЛЬНОЕ УСЛОВИЕ: тороидальный вихрь
#     Метод: спектральный Био-Савар  û = (ik × ω̂) / |k|²
# ═══════════════════════════════════════════════════════

# Размеры деалиасированной сетки (реальное число точек в физпространстве)
Nx_g = x.shape[0]   # = round(Nx * dealias) = 96 для Nx=64
Ny_g = y.shape[1]
Nz_g = z.shape[2]

r_cyl  = np.sqrt(x**2 + y**2)
r_safe = np.maximum(r_cyl, 1e-12)

# Тороидальная вихревость: гауссово распределение на кольце
# ω_φ(r, z) = (Γ/πa²)·exp[−((r−R)² + (z−z₀)²)/a²]
omega_phi = (Gamma / (np.pi * a_core**2)) * np.exp(
    -((r_cyl - R_ring)**2 + (z - z0_ring)**2) / a_core**2
)

# Переход в декартовы координаты: ω = ω_φ·ê_φ = ω_φ·(−y/r, x/r, 0)
omega_x_g = -omega_phi * (y / r_safe)
omega_y_g =  omega_phi * (x / r_safe)
# omega_z_g = 0 (кольцо лежит в плоскости XY)

# ── Спектральный Био-Савар ────────────────────────────────────────
# FFT вихревости
ox_hat = np.fft.fftn(omega_x_g)
oy_hat = np.fft.fftn(omega_y_g)

# Волновые числа деалиасированной сетки
kx_arr    = np.fft.fftfreq(Nx_g) * (2 * np.pi * Nx_g / Lx)
ky_arr    = np.fft.fftfreq(Ny_g) * (2 * np.pi * Ny_g / Ly)
kz_arr_ic = np.fft.fftfreq(Nz_g) * (2 * np.pi * Nz_g / Lz)
KX, KY, KZ = np.meshgrid(kx_arr, ky_arr, kz_arr_ic, indexing='ij')
K2 = KX**2 + KY**2 + KZ**2
K2[0, 0, 0] = 1.0   # нулевая мода = нет среднего потока

# û = (ik × ω̂) / |k|²,  при ω_z = 0:
#   û_x = i(k_y·0 − k_z·ω_y) / k² = −i·k_z·ω_y / k²
#   û_y = i(k_z·ω_x − k_x·0) / k² =  i·k_z·ω_x / k²
#   û_z = i(k_x·ω_y − k_y·ω_x) / k²
ux_hat = -1j * KZ * oy_hat / K2
uy_hat =  1j * KZ * ox_hat / K2
uz_hat =  1j * (KX * oy_hat - KY * ox_hat) / K2

# Нулевая мода — нет среднего сноса
ux_hat[0, 0, 0] = uy_hat[0, 0, 0] = uz_hat[0, 0, 0] = 0.0

u['g'][0] = np.real(np.fft.ifftn(ux_hat))
u['g'][1] = np.real(np.fft.ifftn(uy_hat))
u['g'][2] = np.real(np.fft.ifftn(uz_hat))

_u_max0 = float(np.max(np.sqrt(u['g'][0]**2 + u['g'][1]**2 + u['g'][2]**2)))
logger.info(f"Начальный max|u| = {_u_max0:.3f}  (ожидается ≈ {_U_ring:.3f})")

# ═══════════════════════════════════════════════════════
# §5. ДИАГНОСТИЧЕСКИЕ ВЫРАЖЕНИЯ
# ═══════════════════════════════════════════════════════
omega_field = d3.Curl(u)
speed       = np.sqrt(u @ u)
wmag        = np.sqrt(omega_field @ omega_field)   # |ω| — виден тор!


# ═══════════════════════════════════════════════════════
# §6. ПАРАЛЛЕЛЬНАЯ ЗАПИСЬ PNG-КАДРОВ (в фоновом потоке)
# ═══════════════════════════════════════════════════════
def _save_frame(frame_idx, t,
                wmag_xz, speed_xz,
                wmag_xy, speed_xy,
                x1d, z1d, y1d):
    """Сохраняет один кадр. Вызывается из фонового потока."""
    BG = '#0d0d0d'
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), facecolor=BG)
    fig.suptitle(
        f"Кольцевой вихрь (тор) в газе   |   t = {t:.3f}   |   Re = {Re:.0f}",
        fontsize=13, fontweight='bold', color='white'
    )

    for ax in axes.flat:
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
        ax.tick_params(colors='#888888', labelsize=8)
        ax.xaxis.label.set_color('#aaaaaa')
        ax.yaxis.label.set_color('#aaaaaa')

    ext_xz = [x1d.min(), x1d.max(), z1d.min(), z1d.max()]
    ext_xy = [x1d.min(), x1d.max(), y1d.min(), y1d.max()]
    kw_xz = dict(origin='lower', aspect='equal', extent=ext_xz, interpolation='bilinear')
    kw_xy = dict(origin='lower', aspect='equal', extent=ext_xy, interpolation='bilinear')

    # ── Строка 1: XZ-срезы (продольный вид) ─────────────
    im0 = axes[0, 0].imshow(wmag_xz.T,  cmap='inferno', vmin=0, **kw_xz)
    axes[0, 0].set_title("|ω|  завихрённость (XZ, y=0)",   color='white', fontsize=10)
    axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("z")
    cb = plt.colorbar(im0, ax=axes[0, 0])
    plt.setp(cb.ax.get_yticklabels(), color='#888888')

    im1 = axes[0, 1].imshow(speed_xz.T, cmap='magma',   vmin=0, **kw_xz)
    axes[0, 1].set_title("|u|  скорость (XZ, y=0)",        color='white', fontsize=10)
    axes[0, 1].set_xlabel("x"); axes[0, 1].set_ylabel("z")
    cb = plt.colorbar(im1, ax=axes[0, 1])
    plt.setp(cb.ax.get_yticklabels(), color='#888888')

    # ── Строка 2: XY-срезы (поперечный вид — сечение кольца) ──
    im2 = axes[1, 0].imshow(wmag_xy.T,  cmap='inferno', vmin=0, **kw_xy)
    axes[1, 0].set_title("|ω|  завихрённость (XY, z=z_ring)", color='white', fontsize=10)
    axes[1, 0].set_xlabel("x"); axes[1, 0].set_ylabel("y")
    cb = plt.colorbar(im2, ax=axes[1, 0])
    plt.setp(cb.ax.get_yticklabels(), color='#888888')

    im3 = axes[1, 1].imshow(speed_xy.T, cmap='magma',   vmin=0, **kw_xy)
    axes[1, 1].set_title("|u|  скорость (XY, z=z_ring)",       color='white', fontsize=10)
    axes[1, 1].set_xlabel("x"); axes[1, 1].set_ylabel("y")
    cb = plt.colorbar(im3, ax=axes[1, 1])
    plt.setp(cb.ax.get_yticklabels(), color='#888888')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FRAMES_DIR, f"frame_{frame_idx:04d}.png")
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════
# §7. ЭКСПОРТ VTK + PVD (запускается после симуляции)
# ═══════════════════════════════════════════════════════
def export_vtk(snap_dir=SNAP_DIR, vtk_dir=VTK_DIR, lx=Lx, ly=Ly, lz=Lz):
    """
    Конвертирует HDF5-снапшоты в .vti кадры + gas_torus.pvd для ParaView.

    Поля в каждом кадре:
      speed         — |u|  (скорость)
      vorticity_mag — |ω|  (завихрённость — виден тор!)
      pressure      — p    (давление)
      velocity      — u    (3D вектор, для стримлайнов и линий тока)

    Открывать в ParaView:
      File → Open → vtk_output_gas_torus/gas_torus.pvd
    Рекомендуемые фильтры:
      Contour → vorticity_mag   (изоповерхность тора)
      StreamTracer → velocity   (линии тока внутри кольца)
    """
    import glob, h5py
    try:
        import pyvista as pv
    except ImportError:
        logger.warning("pyvista не установлена — VTK-экспорт пропущен.")
        logger.warning("Установите: .../dedalus_env/bin/pip install pyvista")
        return

    pvd_path = os.path.join(vtk_dir, "gas_torus.pvd")
    os.makedirs(vtk_dir, exist_ok=True)

    h5_files = sorted(glob.glob(os.path.join(snap_dir, "*.h5")))
    if not h5_files:
        logger.warning(f"Нет .h5 файлов в {snap_dir} — VTK-экспорт пропущен.")
        return

    logger.info(f"VTK-экспорт → {vtk_dir}/")
    pvd_entries   = []
    frame_counter = 0

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

                uv = f['tasks/velocity'][i]   # (3, Nx, Ny, Nz)
                grid.point_data["velocity"] = np.stack([
                    uv[0].flatten(order="F"),
                    uv[1].flatten(order="F"),
                    uv[2].flatten(order="F"),
                ], axis=1)

                vti_name = f"gas_torus_{frame_counter:04d}.vti"
                grid.save(os.path.join(vtk_dir, vti_name))
                pvd_entries.append((t, vti_name))
                frame_counter += 1

    # ── PVD-коллекция ─────────────────────────────────────
    with open(pvd_path, "w", encoding="utf-8") as pvd:
        pvd.write('<?xml version="1.0"?>\n')
        pvd.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        pvd.write('  <Collection>\n')
        for t, vti_name in pvd_entries:
            pvd.write(f'    <DataSet timestep="{t:.6f}" group="" part="0" file="{vti_name}"/>\n')
        pvd.write('  </Collection>\n')
        pvd.write('</VTKFile>\n')

    logger.info(f"✅  VTK: {frame_counter} кадров")
    logger.info(f"🎬  ParaView: {pvd_path}")
    logger.info(f"   → Фильтр: Contour → vorticity_mag   (изоповерхность тора)")
    logger.info(f"   → Фильтр: StreamTracer → velocity   (линии тока)")


# ═══════════════════════════════════════════════════════
# §8. ГЛАВНЫЙ ЦИКЛ ИНТЕГРИРОВАНИЯ
# ═══════════════════════════════════════════════════════
if __name__ == '__main__':

    # ── HDF5-снапшоты ──────────────────────────────────
    snapshots = solver.evaluator.add_file_handler(SNAP_DIR, sim_dt=0.1, max_writes=500)
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

    # ── Сетки для срезов (1D) ──────────────────────────
    x1d = x[:, 0, 0]
    y1d = y[0, :, 0]
    z1d = z[0, 0, :]
    iy0 = int(Ny_g // 2)   # y ≈ 0 (срез через ось тора)

    logger.info("═══════════════════════════════════════════════════════")
    logger.info(" КОЛЬЦЕВОЙ ВИХРЬ (ТОР) В ГАЗОВОЙ СРЕДЕ — Dedalus v3")
    logger.info(f" R = {R_ring},  a = {a_core},  Γ = {Gamma},  Re = {Re:.0f}")
    logger.info(f" U_ring ≈ {_U_ring:.3f}  →  T_crossing ≈ {Lz/_U_ring:.2f}")
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

                # ── Вычисляем поля в основном потоке, затем КОПИРУЕМ ──
                w_ev = wmag.evaluate();  w_ev.change_scales(1)
                s_ev = speed.evaluate(); s_ev.change_scales(1)

                w_g = np.array(w_ev['g'])
                s_g = np.array(s_ev['g'])

                # Адаптивный z-срез: следим за положением кольца
                # (z-индекс с максимальным |ω| усреднённым по x,y)
                iz_ring = int(np.argmax(w_g.max(axis=(0, 1))))

                wmag_xz  = w_g[:, iy0, :]
                speed_xz = s_g[:, iy0, :]
                wmag_xy  = w_g[:, :, iz_ring]
                speed_xy = s_g[:, :, iz_ring]

                # ── Отправляем запись PNG в фоновый поток ──
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
        logger.info(f"✅  PNG-кадры: {frame_idx} → {FRAMES_DIR}/")

    # ── Конвертация в VTK/PVD ──────────────────────────
    export_vtk()
