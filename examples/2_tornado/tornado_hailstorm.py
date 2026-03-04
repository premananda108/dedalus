"""
tornado_hailstorm.py
═══════════════════════════════════════════════════════════════════════
СИМУЛЯЦИЯ СУПЕРЯЧЕЙКИ И ТОРНАДО: Микрофизика облаков и образование града

НОВАЯ ФИЗИКА (Основано на реальной метеорологии):
  1. T_bg(z): Температура падает с высотой (от +20C внизу до -20C наверху).
  2. Три фазы воды: q_v (пар), q_c (вода/облако), q_i (лед/град).
  3. Фазовые переходы:
     - Конденсация (пар -> вода) при T < T_dew
     - Замерзание (вода -> лед) при T < T_freeze
     - Таяние (лед -> вода) при T > T_freeze
  4. Падение града: Лед (q_i) имеет собственную скорость падения V_fall вниз.
  5. Скрытая теплота: Переходы выделяют/поглощают тепло (L_v, L_f), питая шторм.

ГЕОМЕТРИЯ ДОМЕНА:
  z ∈[-Lz/2, Lz/2].
  "Земля" находится на краях (z = ±Lz/2), "Тропопауза" (высота 10 км) — в центре (z = 0).

ВЫВОД (в реальном времени, параллельно с симуляцией):
  snapshots_hailstorm/*.h5    — полные данные (h5)
  frames_hailstorm/*.png      — превью-кадры (PNG, пишутся в фоне)
  vtk_output_hailstorm/*.vti  — 3D сетки для ParaView
  vtk_output_hailstorm/tornado_hailstorm.pvd — открывать в ParaView
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

# ── Папки вывода ────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
SNAP_DIR     = os.path.join(SCRIPT_DIR, "snapshots_hailstorm")
FRAMES_DIR   = os.path.join(SCRIPT_DIR, "frames_hailstorm")
VTK_DIR      = os.path.join(SCRIPT_DIR, "vtk_output_hailstorm")
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(VTK_DIR,    exist_ok=True)

# ═══════════════════════════════════════════════════════
# §1. ПАРАМЕТРЫ ЗАДАЧИ
# ═══════════════════════════════════════════════════════
Lx = Ly = Lz = 2 * np.pi
Nx = Ny = Nz = 64
dealias = 3 / 2

Re    = 5000.0
Pr    = 0.71
nu    = 1.0 / Re
kappa = nu / Pr

# Параметры вихря
Gamma = 5.0
r_c   = 0.5
w0    = 1.0

# ── МИКРОФИЗИКА И ТЕРМОДИНАМИКА ────────────────────────
C_p      = 1.0    # Влияние падения давления на температуру
T_dew    = 5.0    # Точка росы (+5 C)
T_freeze = 0.0    # Точка замерзания (0 C)

L_v      = 2.0    # Скрытая теплота конденсации (пар -> вода)
L_f      = 0.5    # Скрытая теплота кристаллизации (вода -> лед)

R_cond   = 5.0    # Скорость конденсации
R_freeze = 5.0    # Скорость замерзания
R_melt   = 5.0    # Скорость таяния града

V_fall   = 1.5    # Терминальная скорость падения града (вниз!)

stop_sim_time = 2.0
max_timestep  = 2e-3
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

x, y, z = dist.local_grids(xbasis, ybasis, zbasis)

u     = dist.VectorField(coords, name='u', bases=bases)
b     = dist.Field(name='b', bases=bases)       # Температурная аномалия
p     = dist.Field(name='p', bases=bases)       # Давление
tau_p = dist.Field(name='tau_p')

# Три агрегатных состояния воды
qv    = dist.Field(name='qv', bases=bases)      # Водяной пар
qc    = dist.Field(name='qc', bases=bases)      # Облачная вода (воронка)
qi    = dist.Field(name='qi', bases=bases)      # Лед / Град

ez = dist.VectorField(coords, name='ez')
ez['g'][2] = 1.0

# ── ФОНОВАЯ ТЕМПЕРАТУРА (Стратификация) ────────────────
T_bg = dist.Field(name='T_bg', bases=bases)
T_bg['g'] = -20.0 * np.cos(2.0 * np.pi * z / Lz)

# ═══════════════════════════════════════════════════════
# §3. УРАВНЕНИЯ С МИКРОФИЗИКОЙ
# ═══════════════════════════════════════════════════════
T_act  = T_bg + b + C_p * p
Cond   = R_cond   * qv * 0.5 * (1.0 + d3.tanh(5.0 * (T_dew   - T_act)))
Freeze = R_freeze  * qc * 0.5 * (1.0 + d3.tanh(5.0 * (T_freeze - T_act)))
Melt   = R_melt    * qi * 0.5 * (1.0 + d3.tanh(5.0 * (T_act   - T_freeze)))

problem = d3.IVP([u, b, p, tau_p, qv, qc, qi], namespace=locals())
problem.add_equation("dt(u) + grad(p) - nu*lap(u) = -u@grad(u) + b*ez")
problem.add_equation("dt(b) - kappa*lap(b) = -u@grad(b) + L_v*Cond + L_f*Freeze - L_f*Melt")
problem.add_equation("dt(qv) - kappa*lap(qv) = -u@grad(qv) - Cond")
problem.add_equation("dt(qc) - kappa*lap(qc) = -u@grad(qc) + Cond - Freeze + Melt")
problem.add_equation(f"dt(qi) - kappa*lap(qi) = -(u - {V_fall}*ez)@grad(qi) + Freeze - Melt")
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("integ(p) = 0")

solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = stop_sim_time

# ═══════════════════════════════════════════════════════
# §4. НАЧАЛЬНЫЕ УСЛОВИЯ
# ═══════════════════════════════════════════════════════
rho_sq   = x**2 + y**2
rho      = np.sqrt(rho_sq)
rho_safe = np.where(rho < 1e-12, 1e-12, rho)

v_theta = (Gamma / (2.0 * np.pi)) / rho_safe * (1.0 - np.exp(-rho_sq / r_c**2))
v_z     = w0 * np.exp(-rho_sq / r_c**2) * (-np.sin(2.0 * np.pi * z / Lz))
dg_dz   = (2.0 * np.pi / Lz) * (-np.cos(2.0 * np.pi * z / Lz))
v_rho   = -(w0 * r_c**2 / (2.0 * rho_safe)) * (1.0 - np.exp(-rho_sq / r_c**2)) * dg_dz

u['g'][0] = -v_theta * (y / rho_safe) + v_rho * (x / rho_safe)
u['g'][1] =  v_theta * (x / rho_safe) + v_rho * (y / rho_safe)
u['g'][2] =  v_z

b['g']  = 2.0 * np.exp(-rho_sq / 0.5**2) * np.exp(-(z + Lz/2)**2 / 0.5)
qv['g'] = 3.0 * np.exp(-(z + Lz/2)**2 / 0.5) + 3.0 * np.exp(-(z - Lz/2)**2 / 0.5)

speed = np.sqrt(u @ u)


# ═══════════════════════════════════════════════════════
# §5. ПАРАЛЛЕЛЬНАЯ ЗАПИСЬ КАДРОВ PNG (в фоновом потоке)
# ═══════════════════════════════════════════════════════
def _save_frame(frame_idx, t,
                qc_xz, qi_xz, T_xz, speed_xz,
                qc_xy, qi_xy,
                x1d, z1d, y1d):
    """Сохраняет один PNG-кадр. Запускается в отдельном потоке."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"Tornado Hailstorm  |  t = {t:.3f}", fontsize=13, fontweight='bold')

    kw_xz = dict(origin='lower', aspect='auto',
                 extent=[x1d.min(), x1d.max(), z1d.min(), z1d.max()])
    kw_xy = dict(origin='lower', aspect='equal',
                 extent=[x1d.min(), x1d.max(), y1d.min(), y1d.max()])

    # Строка 1: XZ-срезы (y=0)
    im0 = axes[0,0].imshow(speed_xz.T, cmap='hot',    vmin=0, **kw_xz)
    axes[0,0].set_title("|u| — скорость (XZ, y=0)");  plt.colorbar(im0, ax=axes[0,0])

    im1 = axes[0,1].imshow(T_xz.T,     cmap='RdBu_r', **kw_xz)
    axes[0,1].set_title("T — температура (XZ)");      plt.colorbar(im1, ax=axes[0,1])

    im2 = axes[0,2].imshow(qc_xz.T,    cmap='Blues',  vmin=0, **kw_xz)
    axes[0,2].set_title("qc — облако/воронка (XZ)");  plt.colorbar(im2, ax=axes[0,2])

    # Строка 2: XY-срезы (z=z_mid) + XZ-срез града
    im3 = axes[1,0].imshow(qi_xz.T,    cmap='cool',   vmin=0, **kw_xz)
    axes[1,0].set_title("qi — град/лёд (XZ)");        plt.colorbar(im3, ax=axes[1,0])

    im4 = axes[1,1].imshow(qc_xy.T,    cmap='Blues',  vmin=0, **kw_xy)
    axes[1,1].set_title("qc — облако (XY, z=0)");     plt.colorbar(im4, ax=axes[1,1])

    im5 = axes[1,2].imshow(qi_xy.T,    cmap='cool',   vmin=0, **kw_xy)
    axes[1,2].set_title("qi — град (XY, z=0)");       plt.colorbar(im5, ax=axes[1,2])

    for ax in axes.flat:
        ax.set_xlabel("x"); ax.set_ylabel("y/z")

    plt.tight_layout()
    out = os.path.join(FRAMES_DIR, f"frame_{frame_idx:04d}.png")
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════
# §6. ЭКСПОРТ VTK + PVD (запускается после симуляции)
# ═══════════════════════════════════════════════════════
def export_vtk(snap_dir=SNAP_DIR, vtk_dir=VTK_DIR,
               lx=Lx, ly=Ly, lz=Lz):
    """
    Конвертирует HDF5-снапшоты в VTK ImageData (.vti) и собирает
    коллекцию tornado_hailstorm.pvd для открытия в ParaView.

    Поля: speed, pressure, Temperature, Vapor, Cloud_Water, Hail_Ice
    """
    import glob, h5py
    try:
        import pyvista as pv
    except ImportError:
        logger.warning("pyvista не установлена — VTK-экспорт пропущен.")
        logger.warning("Установите: .../dedalus_env/bin/pip install pyvista")
        return

    pvd_path = os.path.join(vtk_dir, "tornado_hailstorm.pvd")
    os.makedirs(vtk_dir, exist_ok=True)

    h5_files = sorted(glob.glob(os.path.join(snap_dir, "*.h5")))
    if not h5_files:
        logger.warning(f"Нет .h5 файлов в {snap_dir} — VTK-экспорт пропущен.")
        return

    logger.info(f"Экспорт VTK → {vtk_dir}/")
    pvd_entries   = []
    frame_counter = 0

    for file_path in h5_files:
        with h5py.File(file_path, 'r') as f:
            times = f['scales/sim_time'][:]
            _, Nx_, Ny_, Nz_ = f['tasks/speed'].shape
            dx = lx / Nx_; dy = ly / Ny_; dz = lz / Nz_

            for i in range(len(times)):
                t = float(times[i])

                grid = pv.ImageData()
                grid.dimensions = (Nx_, Ny_, Nz_)
                grid.spacing    = (dx, dy, dz)
                grid.origin     = (-lx/2, -ly/2, -lz/2)

                for key in ('speed', 'pressure', 'Temperature',
                            'Vapor', 'Cloud_Water', 'Hail_Ice'):
                    if key in f['tasks']:
                        grid.point_data[key] = f[f'tasks/{key}'][i].flatten(order="F")

                vti_name = f"hailstorm_frame_{frame_counter:04d}.vti"
                grid.save(os.path.join(vtk_dir, vti_name))
                pvd_entries.append((t, vti_name))
                frame_counter += 1

    # ── PVD-коллекция ────────────────────────────────────────────────
    with open(pvd_path, "w", encoding="utf-8") as pvd:
        pvd.write('<?xml version="1.0"?>\n')
        pvd.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        pvd.write('  <Collection>\n')
        for t, vti_name in pvd_entries:
            pvd.write(f'    <DataSet timestep="{t:.6f}" group="" part="0" file="{vti_name}"/>\n')
        pvd.write('  </Collection>\n')
        pvd.write('</VTKFile>\n')

    logger.info(f"✅  VTK-экспорт завершён: {frame_counter} кадров")
    logger.info(f"🎬  Открыть в ParaView: {pvd_path}")


# ═══════════════════════════════════════════════════════
# §7. ГЛАВНЫЙ ЦИКЛ
# ═══════════════════════════════════════════════════════
if __name__ == '__main__':
    # ── HDF5-снапшоты (FileHandler) ────────────────────────────────
    snapshots = solver.evaluator.add_file_handler(SNAP_DIR, sim_dt=0.05, max_writes=400)
    snapshots.add_task(speed,  name="speed")
    snapshots.add_task(p,      name="pressure")
    snapshots.add_task(T_act,  name="Temperature")
    snapshots.add_task(qv,     name="Vapor")
    snapshots.add_task(qc,     name="Cloud_Water")
    snapshots.add_task(qi,     name="Hail_Ice")

    CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10,
                 safety=0.2, max_dt=max_timestep)
    CFL.add_velocity(u)

    flow = d3.GlobalFlowProperty(solver, cadence=10)
    flow.add_property(speed, name='speed')
    flow.add_property(qc,    name='cloud_max')
    flow.add_property(qi,    name='hail_max')

    logger.info("══════════════════════════════════════════════════")
    logger.info(" Запуск симуляции: Торнадо + Фабрика Града")
    logger.info(f" Кадры PNG пишутся в реальном времени → {FRAMES_DIR}/")
    logger.info(" Ожидайте: Пар -> Облако -> Град -> Падение града")
    logger.info("══════════════════════════════════════════════════")

    frame_idx  = 0
    # Пул из 2 фоновых потоков для записи PNG
    executor = ThreadPoolExecutor(max_workers=2)

    # Сетки для срезов (вычисляем один раз)
    x1d = x[:, 0, 0]
    y1d = y[0, :, 0]
    z1d = z[0, 0, :]
    iy0 = int(Ny // 2)   # y=0
    iz0 = int(Nz // 2)   # z=0 (середина, "высшая точка")

    try:
        while solver.proceed:
            dt = CFL.compute_timestep()
            solver.step(dt)

            if (solver.iteration - 1) % 20 == 0:
                t = solver.sim_time
                logger.info(
                    f"it={solver.iteration:5d} | t={t:4.2f} | "
                    f"max|u|={flow.max('speed'):5.2f} | "
                    f"Max Cloud={flow.max('cloud_max'):5.2f} | "
                    f"Max Hail={flow.max('hail_max'):5.2f}"
                )

                # ── Снимаем срезы во ГЛАВНОМ потоке (копии!) ───────
                qc_g     = qc.evaluate(); qc_g.change_scales(1)
                qi_g     = qi.evaluate(); qi_g.change_scales(1)
                T_g      = T_act.evaluate(); T_g.change_scales(1)
                sp_g     = speed.evaluate(); sp_g.change_scales(1)

                qc_xz  = np.array(qc_g['g'][:, iy0, :])
                qi_xz  = np.array(qi_g['g'][:, iy0, :])
                T_xz   = np.array(T_g['g'][:, iy0, :])
                sp_xz  = np.array(sp_g['g'][:, iy0, :])
                qc_xy  = np.array(qc_g['g'][:, :, iz0])
                qi_xy  = np.array(qi_g['g'][:, :, iz0])

                # ── Отправляем запись PNG в фоновый поток ──────────
                executor.submit(
                    _save_frame,
                    frame_idx, t,
                    qc_xz, qi_xz, T_xz, sp_xz,
                    qc_xy, qi_xy,
                    x1d, z1d, y1d
                )
                frame_idx += 1

    except Exception:
        logger.exception("Ошибка симуляции.")
        raise
    finally:
        # Ждём завершения всех задач PNG
        executor.shutdown(wait=True)
        solver.log_stats()
        logger.info(f"✅  PNG-кадры: {frame_idx} шт. → {FRAMES_DIR}/")

    # ── После симуляции — конвертация в VTK/PVD ────────────────────
    export_vtk()