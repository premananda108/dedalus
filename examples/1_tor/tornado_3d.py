"""
tornado_3d.py
═══════════════════════════════════════════════════════════════════════
СИМУЛЯЦИЯ СМЕРЧА: 3D несжимаемые уравнения Буссинеска–Навье–Стокса

ФИЗИКА:
  ∂u/∂t + (u·∇)u = −∇p + ν∇²u + b·ẑ         (импульс + плавучесть)
  ∂b/∂t + u·∇b   = κ∇²b + Q(r,z)             (тепло / плавучесть)
  ∇·u = 0                                       (несжимаемость)

НАЧАЛЬНОЕ УСЛОВИЕ — вихрь Лэмба–Осеена + апдрафт Бюргерса:
  v_θ(r)  = (Γ/2πr)[1 − exp(−r²/r_c²)]   ← закрутка (потенц. вихрь)
  v_z(r,z)= w₀ exp(−r²/r_c²)·sin(2πz/Lz) ← апдрафт в ядре
  v_r(r,z)= получен точно из ∇·u = 0       ← радиальный приток

ТЕПЛОВОЙ ИСТОЧНИК:
  Q(r,z) = Q₀ exp(−r²/r_Q²)·max[0, sin(πz/Lz + π/2)]
  Имитирует прогрев поверхностного слоя воздуха в ядре смерча.

ПАРАМЕТРЫ БЕЗРАЗМЕРНОСТИ:
  [L] = 1 км,  [U] = 10 м/с,  [T] = 100 с
  Re = UL/ν → ν = 1/Re,  Pr = ν/κ → κ = ν/Pr

ЗАПУСК:
  python3 tornado_3d.py                # последовательно
  mpiexec -n 4 python3 tornado_3d.py  # параллельно (MPI)

ВЫВОД:
  snapshots_tornado/*.h5  — поля: |u|, |ω|, b, p  (каждые dt=0.05)

ОПЦИОНАЛЬНОЕ РАСШИРЕНИЕ — f-плоскость Кориолиса:
  Раскомментируйте секцию "КОРИОЛИС" ниже, чтобы добавить:
  + f_cor·(ẑ×u)   (отклонение ветра из-за вращения Земли)
  Реализован через тензорное поле S_cor:  S·u = ẑ×u = (−uy, ux, 0)

═══════════════════════════════════════════════════════════════════════
"""
import os
# Disable internal threading for numpy/numexpr to avoid performance degradation in Dedalus
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"

import numpy as np
import dedalus.public as d3
import logging

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════
# §1. ПАРАМЕТРЫ ЗАДАЧИ
# ═══════════════════════════════════════════════════════

# ── Геометрия (периодический куб) ──────────────────────
Lx = Ly = Lz = 2 * np.pi          # длина ребра куба [безразм. ~6.28]
Nx = Ny = Nz = 64                 # число мод  (64→быстро, 128→красиво)
dealias     = 3 / 2                # антиалиасинг 3/2-правилом

# ── Безразмерные числа ─────────────────────────────────
Re    = 1e4    # число Рейнольдса: выше → слабее вязкость → дольше смерч
Pr    = 0.71   # число Прандтля для воздуха
nu    = 1.0 / Re          # кинематическая вязкость  ν = 1/Re
kappa = nu / Pr            # температуропроводность   κ = ν/Pr

# ── Вихрь Лэмба–Осеена (начальная закрутка) ────────────
Gamma = 4.0   # циркуляция  Γ  (сила завихрённости)
r_c   = 0.5   # радиус ядра вихря  r_c
w0    = 1.2   # пиковая вертикальная скорость апдрафта

# ── Начальная плавучесть (тёплый столб) ─────────────────
B0    = 2.5   # амплитуда аномалии плавучести
r_b   = 0.7   # радиус тёплого ядра

# ── Тепловой источник Q (непрерывный нагрев в ядре) ─────
Q0    = 0.8   # мощность источника (больше → сильнее апдрафт)
r_Q   = 0.45  # горизонтальный размер источника

# ── f-плоскость Кориолиса (для активации — раскомм. ниже)
# f_cor = 1.0   # параметр Кориолиса: NH > 0, SH < 0

# ── Интегратор по времени ──────────────────────────────
timestepper   = d3.RK222   # 2-й порядок IMEX-RK (неявная диффузия)
stop_sim_time = 1.5       # конечное время симуляции
max_timestep  = 5e-3       # верхняя граница шага dt (CFL ограничивает)

dtype = np.float64

# ═══════════════════════════════════════════════════════
# §2. ПРОСТРАНСТВЕННЫЕ БАЗИСЫ И ДИСТРИБЬЮТОР
# ═══════════════════════════════════════════════════════
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist   = d3.Distributor(coords, dtype=dtype)

# Все три направления — Фурье (периодические граничные условия)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)
bases  = (xbasis, ybasis, zbasis)

# Координатные сетки (локальные для данного MPI-процесса)
x, y, z = dist.local_grids(xbasis, ybasis, zbasis)

# ═══════════════════════════════════════════════════════
# §3. ПОЛЯ ЗАДАЧИ
# ═══════════════════════════════════════════════════════
u     = dist.VectorField(coords, name='u', bases=bases)   # скорость 3D
b     = dist.Field(name='b', bases=bases)                  # плавучесть
p     = dist.Field(name='p', bases=bases)                  # давление
tau_p = dist.Field(name='tau_p')                           # калибровка p

# ── Тепловой источник Q(r,z) ────────────────────────────
# Q сосредоточен в ядре (r < r_Q) и только в нижней полусфере (z > 0)
rho_sq = x**2 + y**2                     # r² = x² + y²
Q      = dist.Field(name='Q', bases=bases)
Q['g'] = (Q0 * np.exp(-rho_sq / r_Q**2)
            * np.maximum(0.0, np.sin(np.pi * z / Lz + np.pi / 2)))

# ── Единичный вектор ẑ — для члена плавучести b·ẑ ──────
ez = dist.VectorField(coords, name='ez')
ez['g'][2] = 1.0   # только z-компонента ненулевая

# ═══════════════════════════════════════════════════════
# §4. ПОСТАНОВКА ЗАДАЧИ (IVP)
# ═══════════════════════════════════════════════════════
problem = d3.IVP([u, b, p, tau_p], namespace=locals())

# ── Уравнение импульса (Навье–Стокс–Буссинеск) ─────────
# ∂u/∂t − ν∇²u + ∇p = −(u·∇)u + b·ẑ
problem.add_equation("dt(u) + grad(p) - nu*lap(u) = -u@grad(u) + b*ez")

# ── Уравнение плавучести (перенос тепла + нагрев) ───────
# ∂b/∂t − κ∇²b = −u·∇b + Q(r,z)
problem.add_equation("dt(b) - kappa*lap(b) = -u@grad(b) + Q")

# ── Несжимаемость ∇·u = 0  (с калибровкой давления) ───
problem.add_equation("div(u) + tau_p = 0")   # убирает нулевую моду
problem.add_equation("integ(p) = 0")          # фиксирует среднее p = 0

solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# ═══════════════════════════════════════════════════════
# §5. НАЧАЛЬНОЕ УСЛОВИЕ
#     Аналитически дивергентный ноль: ∇·u_0 = 0
# ═══════════════════════════════════════════════════════
rho      = np.sqrt(rho_sq)          # полярный радиус r
eps      = 1e-12
rho_safe = np.where(rho < eps, eps, rho)   # защита от деления на 0

# ── 5а. Азимутальная скорость: профиль Лэмба–Осеена ────
# Физика: твёрдотельное вращение (∝r) при r < r_c,
#         потенциальный вихрь (∝1/r) при r > r_c
#
#   v_θ(r) = (Γ/2π) · [1 − exp(−r²/r_c²)] / r
#
v_theta = (Gamma / (2.0 * np.pi)) / rho_safe * (1.0 - np.exp(-rho_sq / r_c**2))

# ── 5б. Вертикальная скорость: апдрафт в ядре ──────────
# Апдрафт максимален в центре (r=0) и периодичен по z (sin)
# Физика: тёплый воздух поднимается в ядре смерча
#
#   v_z(r,z) = w₀ · exp(−r²/r_c²) · sin(2πz/Lz)
#
v_z = w0 * np.exp(-rho_sq / r_c**2) * np.sin(2.0 * np.pi * z / Lz)

# ── 5в. Радиальная скорость: точно из ∇·u = 0 ──────────
# Условие ∇·u = 0 в цилиндрических координатах:
#   (1/r)∂(r·v_r)/∂r + ∂v_z/∂z = 0
#
# Интегрируя: v_r = −(w₀·r_c²) / (2r) · cos(2πz/Lz)·(2π/Lz)·[1−exp(−r²/r_c²)]
#
# Физика: радиальный приток у поверхности (воздух засасывается в ядро)
#
dg_dz  = (2.0 * np.pi / Lz) * np.cos(2.0 * np.pi * z / Lz)
v_rho  = -(w0 * r_c**2 / (2.0 * rho_safe)) * (1.0 - np.exp(-rho_sq / r_c**2)) * dg_dz

# ── Перевод цилиндр. → декарт. ─────────────────────────
# u_x = −v_θ·sinθ + v_r·cosθ,  u_y = v_θ·cosθ + v_r·sinθ
cos_th = x / rho_safe
sin_th = y / rho_safe

u['g'][0] = -v_theta * sin_th + v_rho * cos_th   # ux
u['g'][1] =  v_theta * cos_th + v_rho * sin_th   # uy
u['g'][2] =  v_z                                   # uz

# ── 5г. Начальная плавучесть: тёплый цилиндр ───────────
# Гауссово распределение тепла в ядре — имитирует прогретый воздух
b['g'] = B0 * np.exp(-rho_sq / r_b**2)

# ═══════════════════════════════════════════════════════
# §6. ДИАГНОСТИКА (вычислимые выражения — безопасно при импорте)
# ═══════════════════════════════════════════════════════
omega = d3.Curl(u)                    # завихрённость ω = ∇×u
speed = np.sqrt(u @ u)               # |u| — модуль скорости
wmag  = np.sqrt(omega @ omega)        # |ω| — модуль завихрённости

# ═══════════════════════════════════════════════════════
# §7. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ (запускать отдельно)
# ═══════════════════════════════════════════════════════
def visualize(snap_dir="snapshots_tornado", frame=0, nz_slice=None):
    """
    Строит 4 панели для заданного снапшота:
      (a) |u| в плоскости z=const (горизонтальный срез)
      (b) |ω| в том же срезе
      (c) b   в том же срезе
      (d) p   в вертикальном срезе y=0 (XZ-плоскость)

    Использование:
      from tornado_3d import visualize
      visualize(frame=10)
    """
    import h5py
    import matplotlib.pyplot as plt
    import glob, os

    files = sorted(glob.glob(os.path.join(snap_dir, "*.h5")))
    if not files:
        print(f"Нет .h5 файлов в {snap_dir}/")
        return

    with h5py.File(files[0], 'r') as f:
        times  = f['scales/sim_time'][:]
        speed  = f['tasks/speed'][frame, :, :, :]      # (Nx,Ny,Nz)
        wmag   = f['tasks/vorticity_mag'][frame, :, :, :]
        buoy   = f['tasks/buoyancy'][frame, :, :, :]
        pres   = f['tasks/pressure'][frame, :, :, :]

    Nz = speed.shape[2]
    iz  = Nz // 2 if nz_slice is None else nz_slice   # горизонт. срез
    iy  = speed.shape[1] // 2                           # вертик. срез

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Смерч  |  t = {times[frame]:.2f}", fontsize=14, fontweight='bold')

    kw = dict(origin='lower', cmap='hot', aspect='equal')
    im0 = axes[0,0].imshow(speed[:,:,iz].T, **kw)
    axes[0,0].set_title("|u| — скорость (z-срез)")
    plt.colorbar(im0, ax=axes[0,0], label='|u|')

    im1 = axes[0,1].imshow(wmag[:,:,iz].T, **{**kw, 'cmap':'plasma'})
    axes[0,1].set_title("|ω| — завихрённость (z-срез)")
    plt.colorbar(im1, ax=axes[0,1], label='|ω|')

    im2 = axes[1,0].imshow(buoy[:,:,iz].T, **{**kw, 'cmap':'RdYlBu_r'})
    axes[1,0].set_title("b — плавучесть / тепло (z-срез)")
    plt.colorbar(im2, ax=axes[1,0], label='b')

    im3 = axes[1,1].imshow(pres[:,iy,:].T, **{**kw, 'cmap':'coolwarm'})
    axes[1,1].set_title("p — давление (y=0 вертик. срез XZ)")
    plt.colorbar(im3, ax=axes[1,1], label='p')

    plt.tight_layout()
    out = f"tornado_frame_{frame:04d}.png"
    plt.savefig(out, dpi=150)
    print(f"Сохранено: {out}")
    plt.show()


# ═══════════════════════════════════════════════════════
# §8. ЭКСПОРТ VTK + PVD (вызывается автоматически после симуляции)
# ═══════════════════════════════════════════════════════
def export_vtk(snap_dir="snapshots_tornado", lx=None, ly=None, lz=None):
    """
    Конвертирует HDF5-снапшоты смерча в формат VTK для ParaView.
    Создаёт один файл-коллекцию tornado_simulation.pvd с временны́ми метками.

    Поля в каждом кадре:
      speed         — |u|   (модуль скорости)
      vorticity_mag — |ω|   (завихрённость — главное поле смерча)
      buoyancy      — b     (теплова́я плавучесть)
      pressure      — p     (давление)
      velocity      — u     (3D вектор, для стримлайнов)

    Открывать в ParaView:
      File → Open → vtk_output_tornado/tornado_simulation.pvd
    """
    import glob, h5py
    # Используем глобальные размеры домена, если не переданы явно
    _lx = lx if lx is not None else Lx
    _ly = ly if ly is not None else Ly
    _lz = lz if lz is not None else Lz

    try:
        import pyvista as pv
    except ImportError:
        logger.warning("pyvista не установлена — VTK-экспорт пропущен.")
        logger.warning("Установите: .../dedalus_env/bin/pip install pyvista")
        return

    vtk_dir  = snap_dir.replace("snapshots_", "vtk_output_")
    pvd_path = os.path.join(vtk_dir, "tornado_simulation.pvd")
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
            times           = f['scales/sim_time'][:]
            _, Nx_, Ny_, Nz_ = f['tasks/speed'].shape
            dx = _lx / Nx_
            dy = _ly / Ny_
            dz = _lz / Nz_

            for i in range(len(times)):
                t = float(times[i])

                grid = pv.ImageData()
                grid.dimensions = (Nx_, Ny_, Nz_)
                grid.spacing    = (dx, dy, dz)
                grid.origin     = (-_lx / 2, -_ly / 2, -_lz / 2)

                grid.point_data["speed"]         = f['tasks/speed'][i].flatten(order="F")
                grid.point_data["vorticity_mag"] = f['tasks/vorticity_mag'][i].flatten(order="F")
                grid.point_data["buoyancy"]      = f['tasks/buoyancy'][i].flatten(order="F")
                grid.point_data["pressure"]      = f['tasks/pressure'][i].flatten(order="F")

                # Вектор скорости: (3, Nx, Ny, Nz) → N×3
                uv = f['tasks/velocity'][i]
                grid.point_data["velocity"] = np.stack([
                    uv[0].flatten(order="F"),
                    uv[1].flatten(order="F"),
                    uv[2].flatten(order="F"),
                ], axis=1)

                vti_name = f"tornado_frame_{frame_counter:04d}.vti"
                grid.save(os.path.join(vtk_dir, vti_name))
                pvd_entries.append((t, vti_name))
                frame_counter += 1

    # ── .pvd коллекция ────────────────────────────────────────────
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
# §9. ГЛАВНЫЙ ЦИКЛ ИНТЕГРИРОВАНИЯ
# (FileHandler, CFL и flow — только при запуске, не при импорте)
# ═══════════════════════════════════════════════════════
if __name__ == '__main__':
    # ── Запись снапшотов каждые 0.05 единиц времени ────
    snapshots = solver.evaluator.add_file_handler(
        "snapshots_tornado", sim_dt=0.05, max_writes=400
    )
    snapshots.add_task(speed,  name="speed")
    snapshots.add_task(wmag,   name="vorticity_mag")
    snapshots.add_task(b,      name="buoyancy")
    snapshots.add_task(p,      name="pressure")
    snapshots.add_task(u,      name="velocity")   # полный 3D вектор

    # ── CFL: адаптивный шаг ────────────────────────────
    # dt = safety · Δx / max|u|   (обеспечивает устойчивость)
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

    # ── Мониторинг в лог ───────────────────────────────
    flow = d3.GlobalFlowProperty(solver, cadence=10)
    flow.add_property(speed, name='speed')
    flow.add_property(wmag,  name='omega')
    flow.add_property(b,     name='buoyancy')

    try:
        logger.info("══════════════════════════════════════")
        logger.info(" Запуск симуляции смерча (Dedalus v3) ")
        logger.info(f" Re={Re:.0e}, Pr={Pr}, Γ={Gamma}, r_c={r_c}")
        logger.info(f" Nx=Ny=Nz={Nx},  T_max={stop_sim_time}")
        logger.info("══════════════════════════════════════")
        while solver.proceed:
            dt = CFL.compute_timestep()
            solver.step(dt)
            if (solver.iteration - 1) % 20 == 0:
                logger.info(
                    f"it={solver.iteration:6d} | "
                    f"t={solver.sim_time:8.4f} | "
                    f"dt={dt:8.2e} | "
                    f"max|u|={flow.max('speed'):7.3f} | "
                    f"max|ω|={flow.max('omega'):7.3f} | "
                    f"max b={flow.max('buoyancy'):7.3f}"
                )
    except Exception:
        logger.exception("Ошибка в главном цикле — завершение.")
        raise
    finally:
        solver.log_stats()

    export_vtk()


# ═══════════════════════════════════════════════════════
# §10. ОПЦИОНАЛЬНЫЙ БЛОК: f-ПЛОСКОСТЬ КОРИОЛИСА
# ═══════════════════════════════════════════════════════
#
# Для добавления Coriolis-силы f·(ẑ×u) в уравнение движения
# раскомментируйте этот блок и добавьте S_cor в namespace.
#
# Метод: тензорное поле-ротатор  S·u = (−uy, ux, 0) = ẑ×u
#
# f_cor = 1.0   # параметр Кориолиса  (>0 — сев. полушарие)
#
# S_cor = dist.TensorField((coords, coords), name='S_cor')
# S_cor['g'][0, 1] = -1.0   # S_xy = -1  → (ẑ×u)_x = -uy
# S_cor['g'][1, 0] = +1.0   # S_yx = +1  → (ẑ×u)_y = +ux
# # Компонента z — ноль (автоматически)
#
# Затем в уравнении импульса замените строку на:
# problem.add_equation(
#     "dt(u) + grad(p) - nu*lap(u) + f_cor*(S_cor@u) = -u@grad(u) + b*ez"
# )
# И добавьте в namespace: f_cor=f_cor, S_cor=S_cor
#
# Физический эффект: закрутка смерча усиливается / ослабляется
# в зависимости от знака f_cor (полушарие Земли).
# ═══════════════════════════════════════════════════════