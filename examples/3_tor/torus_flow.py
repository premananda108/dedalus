"""
Два соосных тора — дырками друг к другу.
Каждый тор прокачивает среду через своё отверстие навстречу второму.
Объёмная пенализация задаёт целевое поле скоростей внутри торов.
Визуализация: 2D срез xz, серия PNG.
"""

import os
import numpy as np
import dedalus.public as d3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frames")
SNAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
os.makedirs(OUT, exist_ok=True)
os.makedirs(SNAP_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════════
# §1. ПАРАМЕТРЫ
# ════════════════════════════════════════════════════════════════════

# Расчётная область (x, y, z) — периодическая по всем осям
Lx = Ly = 6.0
Lz = 12.0
Nx = Ny = 64
Nz = 128
dealias = 3/2

# Вязкость (Re = 1/nu)
nu = 5e-3

# ── Геометрия торов ──────────────────────────────────────────────────
# Оба тора соосны с осью Z (ось симметрии — Z)
# Тор A: центр в z = +z0, прокачивает поток В СТОРОНУ z = 0  (uz < 0)
# Тор B: центр в z = -z0, прокачивает поток В СТОРОНУ z = 0  (uz > 0)
z0      = 1.5      # расстояние центров от z=0
R_ring  = 1.0      # большой радиус тора (радиус кольца)
a_core  = 0.40     # толщина трубки тора (гауссова полуширина)

# ── Скорость накачки ─────────────────────────────────────────────────
# Поток идёт вдоль оси Z через дырку тора
# Тороидальное вращение добавляет «закрутку» — делает поток красивее
U_jet   = 9.0      # скорость струи через дырку (вдоль z)
U_swirl = 2.0      # тороидальная закрутка внутри трубки тора

# ── Пенализация ──────────────────────────────────────────────────────
tau_p   = 0.05     # время релаксации к целевому полю (меньше = жёстче тор)

# ── Время симуляции ──────────────────────────────────────────────────
t_stop      = 3.0
max_dt      = 5e-3
frame_every = 15   # сохранять кадр каждые N итераций

# ════════════════════════════════════════════════════════════════════
# §2. БАЗИС И ПОЛЯ
# ════════════════════════════════════════════════════════════════════
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist   = d3.Distributor(coords, dtype=np.float64)

xb = d3.RealFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=dealias)
yb = d3.RealFourier(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2), dealias=dealias)
zb = d3.RealFourier(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)

u      = dist.VectorField(coords, name='u',      bases=(xb, yb, zb))
p      = dist.Field(       name='p',             bases=(xb, yb, zb))
tau_p0 = dist.Field(       name='tau_p0')

# Поля пенализации (обновляются каждый шаг)
gamma  = dist.Field(name='gamma',  bases=(xb, yb, zb))   # γ(x) = mask/τ
utgt   = dist.VectorField(coords, name='utgt', bases=(xb, yb, zb))  # u_target

x, y, z = dist.local_grids(xb, yb, zb)

# ════════════════════════════════════════════════════════════════════
# §3. ПОЛЕ СКОРОСТЕЙ ТОРА
# ════════════════════════════════════════════════════════════════════

def torus_fields(cx, cy, cz, jet_sign):
    """
    Вычисляет маску и целевое поле скоростей для одного тора.

    Тор расположен в плоскости z = cz, его ось — вдоль Z.
    jet_sign = +1 → поток вниз (uz = -U_jet)
    jet_sign = -1 → поток вверх (uz = +U_jet)

    Идея целевого поля:
    ─ Внутри трубки тора (гауссова огибающая по расстоянию до оси кольца):
        · uz = -jet_sign * U_jet   — аксиальный поток через дырку
        · u_phi = U_swirl          — тороидальная закрутка (вихрь трубки)
    ─ Центр тора (дырка) остаётся открытым — поток свободно проходит.
    """
    dx = x - cx
    dy = y - cy
    dz = z - cz

    # Расстояние от оси цилиндра (радиальное в плоскости xy)
    r_cyl  = np.sqrt(dx**2 + dy**2)
    r_safe = np.maximum(r_cyl, 1e-10)

    # Расстояние от оси трубки тора (тороидальный радиус)
    s2 = (r_cyl - R_ring)**2 + dz**2

    # Гауссова огибающая — ненулевая только внутри трубки
    mask = np.exp(-s2 / a_core**2)

    # ── Аксиальный поток через дырку ──────────────────────────────
    # Поток направлен ВДОЛЬ ОСИ Z, но сосредоточен в дырке тора.
    # Дырка — это область r_cyl < R_ring.
    # Мы задаём uz как функцию r_cyl: максимум при r=0, спад к R_ring.
    hole_profile = np.exp(-(r_cyl / (R_ring * 0.7))**2)
    # Высота струи локализована около плоскости тора
    axial_envelope = np.exp(-(dz / a_core)**2)

    uz_jet = -jet_sign * U_jet * hole_profile * axial_envelope

    # ── Тороидальная закрутка внутри трубки ──────────────────────
    # Направление — касательное к кольцу в плоскости xy
    # e_phi = (-dy/r, dx/r, 0)
    u_phi = U_swirl * mask
    ux_swirl = -u_phi * (dy / r_safe)
    uy_swirl =  u_phi * (dx / r_safe)

    # ── Полоидальный поток внутри трубки (красивее взаимодействие) ─
    # Добавляем небольшой рециркуляционный поток внутри трубки тора
    # направленный от внешнего края к оси и обратно
    s_safe = np.maximum(np.sqrt(s2), 1e-10)
    dr = r_cyl - R_ring   # >0 — внешняя сторона трубки, <0 — внутренняя
    pol_strength = 0.5 * U_jet
    ux_pol = pol_strength * mask * (-dz * dx / r_safe) / s_safe
    uy_pol = pol_strength * mask * (-dz * dy / r_safe) / s_safe
    uz_pol = pol_strength * mask * dr / s_safe

    ux = ux_swirl + ux_pol
    uy = uy_swirl + uy_pol
    uz = uz_jet   + uz_pol

    return mask, ux, uy, uz


def set_penalty_fields():
    gamma.change_scales(1)
    utgt.change_scales(1)

    mA, uxA, uyA, uzA = torus_fields(0.0, 0.0, +z0, jet_sign=+1)  # A качает вниз
    mB, uxB, uyB, uzB = torus_fields(0.0, 0.0, -z0, jet_sign=-1)  # B качает вверх

    # γ(x) = (mA + mB) / tau_p
    gamma['g'] = (mA + mB) / tau_p

    # Целевое поле — сумма полей двух торов
    # Там где маски перекрываются, поля суммируются
    utgt['g'][0] = uxA + uxB
    utgt['g'][1] = uyA + uyB
    utgt['g'][2] = uzA + uzB


# ════════════════════════════════════════════════════════════════════
# §4. УРАВНЕНИЯ НАВЬЕ-СТОКСА + ОБЪЁМНАЯ ПЕНАЛИЗАЦИЯ
# ════════════════════════════════════════════════════════════════════
#
#  ∂u/∂t + u·∇u + ∇p − ν∇²u = γ(x)·(u_target − u)
#  ∇·u = 0
#
# Правая часть γ·(u_target − u) — «мягкое» принуждение:
# внутри торов поле скоростей тянется к u_target за время tau_p.
# Вне торов γ ≈ 0, жидкость движется свободно.

set_penalty_fields()

# Начальное условие: сразу задаём u = u_target
u.change_scales(1)
u['g'][0] = utgt['g'][0].copy()
u['g'][1] = utgt['g'][1].copy()
u['g'][2] = utgt['g'][2].copy()

problem = d3.IVP([u, p, tau_p0], namespace=locals())
problem.add_equation(
    "dt(u) + grad(p) - nu*lap(u) = -u@grad(u) + gamma*(utgt - u)"
)
problem.add_equation("div(u) + tau_p0 = 0")
problem.add_equation("integ(p) = 0")

solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = t_stop

# ════════════════════════════════════════════════════════════════════
# §5. ДИАГНОСТИКА
# ════════════════════════════════════════════════════════════════════
speed  = np.sqrt(u @ u)
omega  = d3.Curl(u)
wmag   = np.sqrt(omega @ omega)

# Индекс для среза y = 0
iy0 = Ny // 2

x1d = x[:, 0, 0]
z1d = z[0, 0, :]
extent_xz = [x1d.min(), x1d.max(), z1d.min(), z1d.max()]

# Маска торов для отрисовки контуров
_, _, _, _ = torus_fields(0.0, 0.0, +z0, +1)  # warm up
mA_xz = np.exp(-((np.sqrt(x1d[:, None]**2) - R_ring)**2 + (z1d[None, :] - z0)**2) / a_core**2)
mB_xz = np.exp(-((np.sqrt(x1d[:, None]**2) - R_ring)**2 + (z1d[None, :] + z0)**2) / a_core**2)

# ════════════════════════════════════════════════════════════════════
# §6. ФУНКЦИЯ СОХРАНЕНИЯ КАДРА
# ════════════════════════════════════════════════════════════════════

def save_frame(idx, t, uz_xz, speed_xz, wmag_xz):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor='#06060f')
    fig.suptitle(f"t = {t:.3f}   |   два тора · соосно · струи навстречу",
                 color='#aaccff', fontsize=11, fontfamily='monospace', y=0.97)

    kw = dict(origin='lower', aspect='equal', extent=extent_xz,
              interpolation='bilinear')

    # ── Левый: uz (аксиальная скорость) ──────────────────────────
    vmax_uz = max(U_jet * 0.8, np.abs(uz_xz).max() * 0.9 + 1e-6)
    im0 = axes[0].imshow(uz_xz.T, cmap='RdBu_r', vmin=-vmax_uz, vmax=vmax_uz, **kw)
    axes[0].set_title("uz  (аксиальная скорость)", color='#aaccff', fontsize=9)
    plt.colorbar(im0, ax=axes[0], fraction=0.035, pad=0.02).ax.yaxis.set_tick_params(color='#556')

    # ── Средний: |u| (скорость потока) ────────────────────────────
    vmax_s = max(U_jet, speed_xz.max() * 0.9 + 1e-6)
    im1 = axes[1].imshow(speed_xz.T, cmap='inferno', vmin=0, vmax=vmax_s, **kw)
    axes[1].set_title("|u|  (модуль скорости)", color='#aaccff', fontsize=9)
    plt.colorbar(im1, ax=axes[1], fraction=0.035, pad=0.02)

    # Контуры торов
    for ax in axes[:2]:
        ax.contour(x1d, z1d, mA_xz.T, levels=[0.3], colors='#00ffcc', linewidths=0.8, alpha=0.7)
        ax.contour(x1d, z1d, mB_xz.T, levels=[0.3], colors='#ff6688', linewidths=0.8, alpha=0.7)
        ax.axhline(0, color='#ffffff18', lw=0.5)

    # ── Правый: завихренность |ω| ──────────────────────────────────
    vmax_w = max(5.0, wmag_xz.max() * 0.9 + 1e-6)
    im2 = axes[2].imshow(wmag_xz.T, cmap='plasma', vmin=0, vmax=vmax_w, **kw)
    axes[2].set_title("|ω|  (завихренность)", color='#aaccff', fontsize=9)
    plt.colorbar(im2, ax=axes[2], fraction=0.035, pad=0.02)
    for ax in axes[2:]:
        ax.contour(x1d, z1d, mA_xz.T, levels=[0.3], colors='#00ffcc', linewidths=0.8, alpha=0.5)
        ax.contour(x1d, z1d, mB_xz.T, levels=[0.3], colors='#ff6688', linewidths=0.8, alpha=0.5)

    for ax in axes:
        ax.set_facecolor('#06060f')
        ax.tick_params(colors='#556677', labelsize=7)
        ax.set_xlabel("x", color='#556677', fontsize=8)
        ax.set_ylabel("z", color='#556677', fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#223344')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUT, f"frame_{idx:04d}.png")
    fig.savefig(path, dpi=110, facecolor='#06060f', bbox_inches='tight')
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# §7. ГЛАВНЫЙ ЦИКЛ
# ════════════════════════════════════════════════════════════════════
if __name__ == '__main__':

    CFL = d3.CFL(solver, initial_dt=max_dt, cadence=10,
                 safety=0.3, max_change=1.3, min_change=0.5,
                 max_dt=max_dt, threshold=0.05)
    CFL.add_velocity(u)

    # ── Настройка сохранения HDF5 ──────────────────────────────────────
    snapshots = solver.evaluator.add_file_handler(SNAP_DIR, sim_dt=0.1, max_writes=50)
    snapshots.add_task(u,     name='velocity')
    snapshots.add_task(p,     name='pressure')
    snapshots.add_task(wmag,  name='vorticity_mag')
    snapshots.add_task(gamma, name='tori_mask')  # чтобы видеть положение торов
    # ───────────────────────────────────────────────────────────────────

    frame_idx = 0
    print(f"Симуляция запущена. Кадры → {OUT}/")
    print(f"{'iter':>6}  {'t':>7}  {'dt':>8}  {'|uz|max':>10}")
    print("─" * 40)

    try:
        while solver.proceed:
            dt = CFL.compute_timestep()
            solver.step(dt)

            if solver.iteration % frame_every == 0:
                t = solver.sim_time

                # Вычисляем поля для среза
                u.change_scales(1)

                s_ev = speed.evaluate(); s_ev.change_scales(1)
                w_ev = wmag.evaluate();  w_ev.change_scales(1)

                uz_xz    = np.array(u['g'][2])[:, iy0, :]
                speed_xz = np.array(s_ev['g'])[:, iy0, :]
                wmag_xz  = np.array(w_ev['g'])[:, iy0, :]

                uz_max = np.abs(uz_xz).max()
                print(f"{solver.iteration:>6}  {t:>7.3f}  {dt:>8.2e}  {uz_max:>10.4f}")

                save_frame(frame_idx, t, uz_xz, speed_xz, wmag_xz)
                frame_idx += 1

    except Exception as e:
        logger.exception("Ошибка в цикле")
        raise
    finally:
        solver.log_stats()

    print(f"\nГотово! Сохранено {frame_idx} кадров в {OUT}/")
    print(f"HDF5 снапшоты сохранены в {SNAP_DIR}/")
    print("Для сборки видео:")
    print("  ffmpeg -r 15 -i frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p torus.mp4")