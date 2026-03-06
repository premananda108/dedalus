"""
rotating_tori_interaction.py
═══════════════════════════════════════════════════════════════════════
ДВА СТАБИЛЬНЫХ ПОДВИЖНЫХ ТОРА, ВЗАИМОДЕЙСТВУЮЩИХ ЧЕРЕЗ ГАЗОВЫЕ ПОТОКИ
3D несжимаемые уравнения Навье–Стокса + погружённые тела

ФИЗИКА:
  • Каждый тор — «погружённое тело» (immersed boundary).
    Релаксационная сила F = γ·(u_target − u) поддерживает вращение
    ПОСТОЯННО. Торы никогда не распадаются.

  • Торы ПОДВИЖНЫ: их центры перемещаются под действием газовых
    потоков, создаваемых другим тором. Каждый шаг по времени:
      1) Вычисляем среднюю скорость потока в области тора
      2) Обновляем скорость центра: v += (F_flow / m_eff) · dt
      3) Перемещаем центр: pos += v · dt
      4) Пересчитываем маску и целевую скорость на новой позиции

  • ВЗАИМОДЕЙСТВИЕ через эффект Бернулли:
      Со-вращение → ускорение потока → падение давления → ПРИТЯЖЕНИЕ
      Контр-вращение → торможение потока → рост давления → ОТТАЛКИВАНИЕ

  • ЭФФЕКТИВНАЯ МАССА m_eff определяет инерцию тора.
    Большая масса → медленная реакция → «тяжёлый» тор.

РАСПОЛОЖЕНИЕ:
  Тор A: верхний (z > 0)
  Тор B: нижний (z < 0)
  Оба на оси x=0, y=0

ЗАПУСК:
  python3 rotating_tori_interaction.py
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
# §1. ПАРАМЕТРЫ
# ═══════════════════════════════════════════════════════

Lx = Ly = Lz = 4 * np.pi
Nx = Ny = Nz = 64
dealias = 3 / 2

Re  = 10000.0
nu  = 1.0 / Re

# ── Параметры торов ─────────────────────────────────────
R_ring = 1.2       # большой радиус
a_core = 0.55      # малый радиус ядра

# Начальные позиции: друг над другом по Z
d_sep = 3.0 * R_ring
pos_A = np.array([0.0, 0.0, +d_sep / 2])   # верхний тор
pos_B = np.array([0.0, 0.0, -d_sep / 2])   # нижний тор
vel_A = np.array([0.0, 0.0, 0.0])           # начальная скорость = 0
vel_B = np.array([0.0, 0.0, 0.0])

# ── Скорости вращения ──────────────────────────────────
Omega_tor = 10.0   # тороидальное вращение
Omega_pol = 3.0    # полоидальное (внутри трубки)
sign_A = +1
sign_B = -1        # +1/+1 → со-вращение → притяжение
                   # +1/-1 → контр-вращение → отталкивание

# ── Сила удержания и масса ─────────────────────────────
tau_relax = 0.15   # время релаксации (маленькое → жёсткое удержание)
m_eff     = 5.0    # эффективная масса тора (больше → инертнее)

# ── Интегратор ──────────────────────────────────────────
stop_sim_time = 3.0
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

gamma_field = dist.Field(name='gamma_field', bases=bases)
u_target    = dist.VectorField(coords, name='u_target', bases=bases)

x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
Nx_g, Ny_g, Nz_g = x.shape[0], y.shape[1], z.shape[2]

# ═══════════════════════════════════════════════════════
# §3. ФУНКЦИИ РАСЧЁТА МАСКИ И ЦЕЛЕВОЙ СКОРОСТИ
# ═══════════════════════════════════════════════════════

def torus_mask(cx, cy, cz, x_g, y_g, z_g, R=R_ring, a=a_core):
    """Гауссова маска ядра тора с центром (cx, cy, cz)."""
    dx = x_g - cx
    dy = y_g - cy
    dz = z_g - cz
    r_cyl = np.sqrt(dx**2 + dy**2)
    s2 = (r_cyl - R)**2 + dz**2
    return np.exp(-s2 / a**2)


def torus_velocity(Omega_t, Omega_p, sign, cx, cy, cz,
                   x_g, y_g, z_g, R=R_ring, a=a_core):
    """Целевая скорость тора: тороидальное + полоидальное вращение."""
    dx = x_g - cx
    dy = y_g - cy
    dz = z_g - cz
    r_cyl  = np.sqrt(dx**2 + dy**2)
    r_safe = np.maximum(r_cyl, 1e-12)

    s2 = (r_cyl - R)**2 + dz**2
    envelope = np.exp(-s2 / a**2)

    # Тороидальное: u_φ · ê_φ
    u_phi = sign * Omega_t * envelope
    ux = -u_phi * (dy / r_safe)
    uy =  u_phi * (dx / r_safe)

    # Полоидальное: вращение внутри трубки (нулевая нетто-циркуляция)
    s = np.sqrt(s2 + 1e-24)
    dr = r_cyl - R
    vpol = sign * Omega_p * (2.0 * s / a) * np.exp(-s2 / a**2)
    s_safe = np.maximum(s, 1e-12)
    ux += -vpol * dz * (dx / r_safe) / s_safe
    uy += -vpol * dz * (dy / r_safe) / s_safe
    uz  =  vpol * dr / s_safe

    return ux, uy, uz


def update_fields(pA, pB):
    """Пересчитать gamma_field и u_target для текущих позиций торов."""
    gamma_field.change_scales(1)
    u_target.change_scales(1)
    mA = torus_mask(pA[0], pA[1], pA[2], x, y, z)
    mB = torus_mask(pB[0], pB[1], pB[2], x, y, z)
    gamma_field['g'] = (mA + mB) / tau_relax

    uxA, uyA, uzA = torus_velocity(
        Omega_tor, Omega_pol, sign_A, pA[0], pA[1], pA[2], x, y, z)
    uxB, uyB, uzB = torus_velocity(
        Omega_tor, Omega_pol, sign_B, pB[0], pB[1], pB[2], x, y, z)

    u_target['g'][0] = uxA + uxB
    u_target['g'][1] = uyA + uyB
    u_target['g'][2] = uzA + uzB


def compute_drift_force(u_field, pos, sign_t):
    """
    Вычисляет силу на тор из газовых потоков.

    Метод: усредняем скорость по объёму ядра тора (взвешенную маской),
    вычитаем собственную скорость вращения → получаем «внешний поток»
    (от другого тора и от взаимодействия). Этот поток создаёт силу.
    """
    cx, cy, cz = pos
    mask = torus_mask(cx, cy, cz, x, y, z)
    mask_sum = float(np.sum(mask)) + 1e-30

    # Приводим u к стандартному масштабу (64³), чтобы совпало с маской x,y,z
    u_field.change_scales(1)

    # Средняя скорость потока в области тора
    u_mean = np.array([
        float(np.sum(u_field['g'][0] * mask)) / mask_sum,
        float(np.sum(u_field['g'][1] * mask)) / mask_sum,
        float(np.sum(u_field['g'][2] * mask)) / mask_sum,
    ])

    # Вычитаем собственную скорость вращения (она не должна двигать тор)
    ux_self, uy_self, uz_self = torus_velocity(
        Omega_tor, Omega_pol, sign_t, cx, cy, cz, x, y, z)
    u_self_mean = np.array([
        float(np.sum(ux_self * mask)) / mask_sum,
        float(np.sum(uy_self * mask)) / mask_sum,
        float(np.sum(uz_self * mask)) / mask_sum,
    ])

    # Внешний поток = полный − собственный
    u_ext = u_mean - u_self_mean

    # Сила ∝ внешний поток (drag-like)
    drag_coeff = mask_sum * 0.1   # коэффициент связи с потоком
    force = drag_coeff * u_ext

    return force


# ── Инициализация полей ────────────────────────────────
update_fields(pos_A, pos_B)
u['g'][0] = u_target['g'][0].copy()
u['g'][1] = u_target['g'][1].copy()
u['g'][2] = u_target['g'][2].copy()

_u_max0 = float(np.max(np.sqrt(u['g'][0]**2 + u['g'][1]**2 + u['g'][2]**2)))
logger.info(f"Начальный max|u| = {_u_max0:.3f}")

# ═══════════════════════════════════════════════════════
# §4. УРАВНЕНИЯ (сила ВСЕГДА включена)
# ═══════════════════════════════════════════════════════
problem = d3.IVP([u, p, tau_p], namespace=locals())
problem.add_equation(
    "dt(u) + grad(p) - nu*lap(u) = "
    "-u@grad(u) + gamma_field*(u_target - u)"
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
# §6. ВИЗУАЛИЗАЦИЯ
# ═══════════════════════════════════════════════════════
def _save_frame(frame_idx, t,
                wmag_xz, speed_xz, w_g, s_g,
                x1d, z1d, y1d,
                pA, pB, track_A, track_B):
    """PNG-кадр: 4 панели + траектории торов."""
    BG = '#08081a'
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), facecolor=BG)

    fig.suptitle(
        f"Стабильные торы — газовое взаимодействие   |   t = {t:.3f}\n"
        f"Re={Re:.0f}   Ω_tor={Omega_tor}   m_eff={m_eff}   "
        f"z_A={pA[2]:.2f}   z_B={pB[2]:.2f}   "
        f"Δz={abs(pA[2]-pB[2]):.2f}",
        fontsize=11, fontweight='bold', color='#66ccff', y=0.98
    )

    for ax in axes.flat:
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_edgecolor('#2a2a3a')
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

    # ── [0,0] |ω| XZ + траектории ─────────────
    im0 = axes[0, 0].imshow(wmag_xz.T, cmap='inferno',
                             vmin=0, vmax=vmax_w, **kw_xz)
    axes[0, 0].set_title("|ω|  завихрённость  (XZ, y=0)",
                          color='white', fontsize=10)
    axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("z")
    # Траектории
    tA = np.array(track_A)
    tB = np.array(track_B)
    if len(tA) > 1:
        axes[0, 0].plot(tA[:, 0], tA[:, 2], '-', color='#00d4ff',
                        lw=1.0, alpha=0.7, label='Тор A')
        axes[0, 0].plot(tB[:, 0], tB[:, 2], '-', color='#ff6b6b',
                        lw=1.0, alpha=0.7, label='Тор B')
    axes[0, 0].plot(pA[0], pA[2], 'o', color='#00d4ff', ms=5)
    axes[0, 0].plot(pB[0], pB[2], 'o', color='#ff6b6b', ms=5)
    cb = plt.colorbar(im0, ax=axes[0, 0])
    plt.setp(cb.ax.get_yticklabels(), color='#888888')

    # ── [0,1] |u| XZ ─────────────────────────
    im1 = axes[0, 1].imshow(speed_xz.T, cmap='magma',
                             vmin=0, vmax=vmax_s, **kw_xz)
    axes[0, 1].set_title("|u|  скорость  (XZ, y=0)",
                          color='white', fontsize=10)
    axes[0, 1].set_xlabel("x"); axes[0, 1].set_ylabel("z")
    axes[0, 1].axhline(pA[2], color='#00d4ff', lw=0.6, ls='--', alpha=0.4)
    axes[0, 1].axhline(pB[2], color='#ff6b6b', lw=0.6, ls='--', alpha=0.4)
    cb = plt.colorbar(im1, ax=axes[0, 1])
    plt.setp(cb.ax.get_yticklabels(), color='#888888')

    # ── [1,0] |ω| XY (срез через тор A) ──────
    iz_A = int(np.argmin(np.abs(z1d - pA[2])))
    iz_A = max(0, min(iz_A, w_g.shape[2] - 1))
    wmag_xy_A = w_g[:, :, iz_A]
    vmax_w_xy = np.percentile(np.abs(wmag_xy_A), 99.5) + 1e-8
    im2 = axes[1, 0].imshow(wmag_xy_A.T, cmap='inferno',
                             vmin=0, vmax=vmax_w_xy, **kw_xy)
    axes[1, 0].set_title(f"|ω|  (XY, z={pA[2]:.1f} — тор A)",
                          color='#00d4ff', fontsize=10)
    axes[1, 0].set_xlabel("x"); axes[1, 0].set_ylabel("y")
    cb = plt.colorbar(im2, ax=axes[1, 0])
    plt.setp(cb.ax.get_yticklabels(), color='#888888')

    # ── [1,1] |ω| XY (срез через тор B) ──────
    iz_B = int(np.argmin(np.abs(z1d - pB[2])))
    iz_B = max(0, min(iz_B, w_g.shape[2] - 1))
    wmag_xy_B = w_g[:, :, iz_B]
    vmax_w_xyB = np.percentile(np.abs(wmag_xy_B), 99.5) + 1e-8
    im3 = axes[1, 1].imshow(wmag_xy_B.T, cmap='inferno',
                             vmin=0, vmax=vmax_w_xyB, **kw_xy)
    axes[1, 1].set_title(f"|ω|  (XY, z={pB[2]:.1f} — тор B)",
                          color='#ff6b6b', fontsize=10)
    axes[1, 1].set_xlabel("x"); axes[1, 1].set_ylabel("y")
    cb = plt.colorbar(im3, ax=axes[1, 1])
    plt.setp(cb.ax.get_yticklabels(), color='#888888')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(FRAMES_DIR, f"frame_{frame_idx:04d}.png")
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════
# §8. ГЛАВНЫЙ ЦИКЛ
# ═══════════════════════════════════════════════════════
if __name__ == '__main__':

    # HDF5-снапшоты
    snapshots = solver.evaluator.add_file_handler(
        SNAP_DIR, sim_dt=0.2, max_writes=500
    )
    snapshots.add_task(speed, name="speed")
    snapshots.add_task(wmag,  name="vorticity_mag")
    snapshots.add_task(p,     name="pressure")
    snapshots.add_task(u,     name="velocity")

    # CFL
    CFL = d3.CFL(
        solver, initial_dt=max_timestep, cadence=10,
        safety=0.25, threshold=0.1,
        max_change=1.4, min_change=0.5, max_dt=max_timestep,
    )
    CFL.add_velocity(u)

    # Мониторинг
    flow = d3.GlobalFlowProperty(solver, cadence=10)
    flow.add_property(speed, name='speed')
    flow.add_property(wmag,  name='omega')

    x1d = x[:, 0, 0]
    y1d = y[0, :, 0]
    z1d = z[0, 0, :]
    iy0 = int(Ny_g // 2)

    # Траектории центров торов (для визуализации)
    track_A = [pos_A.copy()]
    track_B = [pos_B.copy()]

    logger.info("═══════════════════════════════════════════════════════")
    logger.info(" СТАБИЛЬНЫЕ ПОДВИЖНЫЕ ТОРЫ — Dedalus v3")
    logger.info(f" R={R_ring}, a={a_core}, Re={Re:.0f}")
    logger.info(f" Ω_tor={Omega_tor}, Ω_pol={Omega_pol}")
    logger.info(f" sign_A={sign_A:+d}, sign_B={sign_B:+d}")
    logger.info(f" m_eff={m_eff}, τ_relax={tau_relax}")
    logger.info(f" Тор A: z₀={pos_A[2]:.2f}   Тор B: z₀={pos_B[2]:.2f}")
    logger.info(f" Δz₀={abs(pos_A[2]-pos_B[2]):.2f}")
    logger.info(f" Время: {stop_sim_time:.1f}")
    logger.info("═══════════════════════════════════════════════════════")

    frame_idx = 0
    executor = ThreadPoolExecutor(max_workers=2)

    try:
        while solver.proceed:
            t = solver.sim_time

            # ── 1. Вычисляем силу на каждый тор ───────
            F_A = compute_drift_force(u, pos_A, sign_A)
            F_B = compute_drift_force(u, pos_B, sign_B)

            # ── 2. Обновляем скорости (F = ma) ────────
            dt_sim = CFL.compute_timestep()
            vel_A += (F_A / m_eff) * dt_sim
            vel_B += (F_B / m_eff) * dt_sim

            # Лёгкое демпфирование (стабилизация)
            vel_A *= 0.999
            vel_B *= 0.999

            # ── 3. Перемещаем центры ──────────────────
            pos_A += vel_A * dt_sim
            pos_B += vel_B * dt_sim

            # Ограничение: торы не выходят за домен
            half = Lz / 2 - R_ring
            pos_A[2] = np.clip(pos_A[2], -half, half)
            pos_B[2] = np.clip(pos_B[2], -half, half)
            pos_A[0] = np.clip(pos_A[0], -Lx/2 + R_ring, Lx/2 - R_ring)
            pos_B[0] = np.clip(pos_B[0], -Lx/2 + R_ring, Lx/2 - R_ring)

            # ── 4. Пересчитываем поля на новых позициях ─
            update_fields(pos_A, pos_B)

            # ── 5. Шаг Навье-Стокса ───────────────────
            solver.step(dt_sim)

            # Запись траекторий
            if (solver.iteration - 1) % 5 == 0:
                track_A.append(pos_A.copy())
                track_B.append(pos_B.copy())

            # ── Логирование и визуализация ────────────
            if (solver.iteration - 1) % 20 == 0:
                max_speed = flow.max('speed')
                max_omega = flow.max('omega')
                dz = abs(pos_A[2] - pos_B[2])

                logger.info(
                    f"it={solver.iteration:5d} | t={t:6.3f} | "
                    f"max|u|={max_speed:6.3f} | max|ω|={max_omega:7.2f} | "
                    f"Δz={dz:.3f} | "
                    f"zA={pos_A[2]:+.3f} zB={pos_B[2]:+.3f}"
                )

                w_ev = wmag.evaluate(); w_ev.change_scales(1)
                s_ev = speed.evaluate(); s_ev.change_scales(1)
                w_g = np.array(w_ev['g'])
                s_g = np.array(s_ev['g'])

                executor.submit(
                    _save_frame, frame_idx, t,
                    w_g[:, iy0, :].copy(), s_g[:, iy0, :].copy(),
                    w_g.copy(), s_g.copy(),
                    x1d, z1d, y1d,
                    pos_A.copy(), pos_B.copy(),
                    list(track_A), list(track_B)
                )
                frame_idx += 1

    except Exception:
        logger.exception("Ошибка.")
        raise
    finally:
        executor.shutdown(wait=True)
        solver.log_stats()
        logger.info(f"✅  PNG: {frame_idx} → {FRAMES_DIR}/")
        logger.info(f"   Финальные позиции: A={pos_A}, B={pos_B}")
        logger.info(f"   Δz финальное = {abs(pos_A[2]-pos_B[2]):.4f}")
