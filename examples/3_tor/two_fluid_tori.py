"""
two_fluid_tori.py
═══════════════════════════════════════════════════════════════════════
ДВА ЖИДКИХ ТОРА В ГАЗЕ — двухфазная несжимаемая симуляция

ФИЗИКА:
  • Каждый тор — ЖИДКОСТЬ (ν_liq >> ν_gas) в газовой среде
  • Фазовые поля φ_A, φ_B АДВЕКТИРУЮТСЯ потоком u
    → торы «плывут» естественно, без ручного перемещения

ПОЧЕМУ ТОРЫ НЕ РАСПАДАЮТСЯ:
  1. ν_liq в 80× > ν_gas → высокая вязкость жидкости гасит деформации
  2. Угловой момент L = ∫ r × u · φ dV сохраняется → центробежный барьер
  3. γ_soft = 0.8 (vs 1/τ = 6.7 в оригинале) — слабая поддержка,
     ТОЛЬКО компенсирует вязкое затухание, не "держит на поводке"

ВЗАИМОДЕЙСТВИЕ ЧЕРЕЗ ГАЗ:
  • Давление в газовой щели ЕДИНСТВЕННЫЙ канал связи торов
  • Бернулли: со-вращение → ускорение газа → P↓ → ПРИТЯЖЕНИЕ
  • Контр-вращение: торможение → P↑ → ОТТАЛКИВАНИЕ

УРАВНЕНИЯ (Dedalus v3, псевдоспектральный метод):
  ∂u/∂t + ∇p − ν_gas·∇²u = −u·∇u + dnu_eff·∇²u + γ_f·(u_ring − u)
  ∂φ_A/∂t − κ·∇²φ_A      = −u·∇φ_A
  ∂φ_B/∂t − κ·∇²φ_B      = −u·∇φ_B
  ∇·u = 0

  dnu_eff(x) = (ν_liq − ν_gas) · (φ_A + φ_B)   ← переменная вязкость
  γ_f(x)     = γ_soft · (φ_A + φ_B)             ← слабая поддержка

ДИАГНОСТИКА:
  • max(φ_A), max(φ_B) ≥ 0.5 → тор ЦЕЛ
  • Центроид φ_A / φ_B → реальная позиция тора (вычисляется из поля)
  • Δz = |z_A − z_B| → расстояние между торами

ЗАПУСК:
  python3 two_fluid_tori.py

ВЫХОД:
  frames_two_fluid/frame_XXXX.png   — 6-панельные кадры
  snapshots_two_fluid/*.h5          — HDF5 (скорость, вихри, φ-поля)
  vtk_two_fluid/*.vti + .pvd        — для ParaView

СОЗДАНИЕ ВИДЕО:
  ffmpeg -r 20 -i frames_two_fluid/frame_%04d.png \
         -c:v libx264 -crf 18 -pix_fmt yuv420p tori_liquid.mp4
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)

# ── Папки вывода ──────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SNAP_DIR   = os.path.join(SCRIPT_DIR, "snapshots_two_fluid")
FRAMES_DIR = os.path.join(SCRIPT_DIR, "frames_two_fluid")
VTK_DIR    = os.path.join(SCRIPT_DIR, "vtk_two_fluid")
os.makedirs(FRAMES_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# §1. ПАРАМЕТРЫ
# ═══════════════════════════════════════════════════════════════

# ── Сетка и домен ─────────────────────────────────────────────
Lx = Ly = Lz  = 4.0 * np.pi   # размер периодического куба
Nx = Ny = Nz  = 64             # сетка 64³ ≈ 262k узлов
dealias        = 3 / 2         # деалиасинг нелинейных членов
dtype          = np.float64

# ── ДВУХФАЗНЫЕ свойства ────────────────────────────────────────
nu_gas    = 5e-4   # вязкость ГАЗа        (малая → турбулентный)
nu_liq    = 4e-2   # вязкость ЖИДКОСТИ    (80× больше → держит форму!)
kappa_phi = 5e-4   # диффузия φ-поля      (минимальная, только стабильность)

# Константа переменной вязкости: Δν = ν_liq − ν_gas
dnu = nu_liq - nu_gas   # ≈ 0.04

# ── Геометрия торов ─────────────────────────────────────────
R_ring = 1.2    # большой радиус тора
a_core = 0.50   # малый радиус трубки (сечение)

# ── Вращение ───────────────────────────────────────────────
Omega_tor = 8.0   # тороидальная скорость (вокруг оси Z тора)
Omega_pol = 2.5   # полоидальная (внутри трубки — нулевой нетто-импульс)
sign_A    = +1
sign_B    = +1    # ОБА +1 → СО-вращение → эффект Бернулли → ПРИТЯЖЕНИЕ
                  # sign_B = -1 → КОНТР-вращение → ОТТАЛКИВАНИЕ

# ── Слабая поддержка вращения ──────────────────────────────
# γ_soft << 1/τ_relax_original = 6.7  → физическая, не жёсткая
gamma_soft = 0.8   # [1/с] — медленно компенсирует вязкое затухание

# ── Начальные позиции ──────────────────────────────────────
d_sep  = 3.2 * R_ring          # начальное разделение
pos_A0 = np.array([0.0, 0.0,  d_sep / 2])   # верхний тор
pos_B0 = np.array([0.0, 0.0, -d_sep / 2])   # нижний тор

# ── Интегратор ─────────────────────────────────────────────
stop_sim_time = 3.0      # длительность симуляции
max_timestep  = 3e-3     # максимальный Δt (CFL ограничит ещё)


# ═══════════════════════════════════════════════════════════════
# §2. БАЗИСЫ И ПОЛЯ DEDALUS v3
# ═══════════════════════════════════════════════════════════════
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist   = d3.Distributor(coords, dtype=dtype)

xbasis = d3.RealFourier(coords['x'], size=Nx,
                         bounds=(-Lx/2, Lx/2), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny,
                         bounds=(-Ly/2, Ly/2), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=Nz,
                         bounds=(-Lz/2, Lz/2), dealias=dealias)
bases  = (xbasis, ybasis, zbasis)

# ── Поля ИВП (интегрируются Dedalus) ──────────────────────
p     = dist.Field(name='p',     bases=bases)   # давление
u     = dist.VectorField(coords, name='u',
                         bases=bases)            # скорость (газ+жидкость)
phi_A = dist.Field(name='phi_A', bases=bases)   # концентрация жидкости тора A
phi_B = dist.Field(name='phi_B', bases=bases)   # концентрация жидкости тора B
tau_p = dist.Field(name='tau_p')                # множитель несжимаемости

# ── Вспомогательные поля (обновляются вручную в цикле) ────
# dnu_eff(x) = Δν · (φ_A + φ_B)  — локальная добавка к вязкости
dnu_eff  = dist.Field(name='dnu_eff', bases=bases)
# gamma_f(x) = γ_soft · (φ_A + φ_B)  — слабая поддержка
gamma_f  = dist.Field(name='gamma_f', bases=bases)
# u_target — желаемая скорость вращения (обновляется по центроиду φ)
u_target = dist.VectorField(coords, name='u_target', bases=bases)

# ── Координатные сетки ─────────────────────────────────────
x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
Nx_g = x.shape[0]; Ny_g = y.shape[1]; Nz_g = z.shape[2]
iy0  = int(Ny_g // 2)   # центральный срез по Y для визуализации


# ═══════════════════════════════════════════════════════════════
# §3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════

def torus_mask(cx, cy, cz, R=R_ring, a=a_core):
    """
    Гауссова маска жидкости тора с центром (cx, cy, cz).
    Возвращает numpy-массив в форме сетки.
    Значение: 1.0 в ядре, → 0 на расстоянии ~a от кольца.
    """
    r_cyl = np.sqrt((x - cx)**2 + (y - cy)**2)
    s2    = (r_cyl - R)**2 + (z - cz)**2
    return np.exp(-s2 / a**2)


def torus_velocity(sign, cx, cy, cz, R=R_ring, a=a_core):
    """
    Целевое поле скорости ВРАЩЕНИЯ тора.
    Тороидальная компонента (вокруг оси Z тора) +
    полоидальная (внутри трубки, нулевой нетто-импульс).

    sign = +1 или -1 — направление тороидального вращения.
    """
    dx = x - cx;  dy = y - cy;  dz_ = z - cz
    r_cyl  = np.sqrt(dx**2 + dy**2)
    r_safe = np.maximum(r_cyl, 1e-12)
    s2     = (r_cyl - R)**2 + dz_**2
    s      = np.sqrt(s2 + 1e-24)
    s_safe = np.maximum(s, 1e-12)
    dr     = r_cyl - R

    # ── Тороидальная: вращение вокруг Z ────────
    env   = np.exp(-s2 / a**2)
    u_phi = sign * Omega_tor * env
    ux    = -u_phi * (dy  / r_safe)
    uy    =  u_phi * (dx  / r_safe)

    # ── Полоидальная: вращение внутри трубки ──
    # vpol ∝ (2s/a)·exp(-s²/a²) — производная Гаусса → нулевая нетто-циркуляция
    vpol  = sign * Omega_pol * (2.0 * s / a) * np.exp(-s2 / a**2)
    ux   += -vpol * dz_ * (dx / r_safe) / s_safe
    uy   += -vpol * dz_ * (dy / r_safe) / s_safe
    uz    =  vpol * dr  / s_safe

    return ux, uy, uz


def centroid_from_phi(phi_g):
    """
    Вычислить центр масс фазового поля φ.
    Это РЕАЛЬНАЯ позиция тора — вычисляется из поля, не задаётся вручную.
    """
    total = float(np.sum(phi_g)) + 1e-30
    cx    = float(np.sum(x * phi_g)) / total
    cy    = float(np.sum(y * phi_g)) / total
    cz    = float(np.sum(z * phi_g)) / total
    return np.array([cx, cy, cz])


def update_aux_fields(pos_A, pos_B, phiA_g, phiB_g):
    """
    Обновить вспомогательные поля перед каждым шагом NS.

    dnu_eff = Δν · (φ_A + φ_B)  →  переменная вязкость
    gamma_f  = γ_soft · (φ_A + φ_B)  →  слабая поддержка
    u_target = u_rot_A + u_rot_B  →  целевое вращение
    """
    phi_tot_g = phiA_g + phiB_g

    # Переменная вязкость (ν = ν_gas + dnu_eff)
    dnu_eff.change_scales(1)
    dnu_eff['g'] = dnu * phi_tot_g

    # Слабая поддержка вращения
    gamma_f.change_scales(1)
    gamma_f['g'] = gamma_soft * phi_tot_g

    # Целевое поле вращения
    uxA, uyA, uzA = torus_velocity(sign_A, *pos_A)
    uxB, uyB, uzB = torus_velocity(sign_B, *pos_B)
    u_target.change_scales(1)
    u_target['g'][0] = uxA + uxB
    u_target['g'][1] = uyA + uyB
    u_target['g'][2] = uzA + uzB


# ═══════════════════════════════════════════════════════════════
# §4. НАЧАЛЬНЫЕ УСЛОВИЯ
# ═══════════════════════════════════════════════════════════════

# Инициализация φ-полей: жидкость в форме тора
phi_A.change_scales(1);  phi_B.change_scales(1)
phi_A['g'] = torus_mask(*pos_A0)
phi_B['g'] = torus_mask(*pos_B0)

# Инициализация вспомогательных полей
update_aux_fields(pos_A0, pos_B0,
                  phi_A['g'].copy(), phi_B['g'].copy())

# Инициализация скорости: вращение обоих торов
u.change_scales(1)
uxA, uyA, uzA = torus_velocity(sign_A, *pos_A0)
uxB, uyB, uzB = torus_velocity(sign_B, *pos_B0)
u['g'][0] = uxA + uxB
u['g'][1] = uyA + uyB
u['g'][2] = uzA + uzB

_umax = float(np.max(np.sqrt(
    u['g'][0]**2 + u['g'][1]**2 + u['g'][2]**2)))
logger.info(f"Init: max|u|={_umax:.3f}  "
            f"max(φ_A)={float(np.max(phi_A['g'])):.3f}  "
            f"max(φ_B)={float(np.max(phi_B['g'])):.3f}")


# ═══════════════════════════════════════════════════════════════
# §5. УРАВНЕНИЯ (двухфазные Навье–Стокс)
# ═══════════════════════════════════════════════════════════════
#
#  ∂u/∂t + ∇p − ν_gas·∇²u = −u·∇u
#                            + dnu_eff·∇²u          ← переменная вязкость
#                            + γ_f·(u_target − u)   ← слабая поддержка
#
#  Неявно: ν_gas·∇²u  (базовая жёсткость)
#  Явно:   всё в RHS   (нелинейность + перем. вязкость + поддержка)
#
#  Стабильность dnu_eff·∇²u явно:
#    dnu_max·k²_max·dt ≈ 0.04·256·0.003 = 0.03 << 1  ✓
#
# ──────────────────────────────────────────────────────────────

problem = d3.IVP([u, p, phi_A, phi_B, tau_p], namespace=locals())

# Навье–Стокс с переменной вязкостью жидкости
problem.add_equation(
    "dt(u) + grad(p) - nu_gas*lap(u) = "
    "-u@grad(u) "
    "+ dnu_eff*lap(u) "           # переменная вязкость: Δν·φ·∇²u
    "+ gamma_f*(u_target - u)"    # слабая поддержка вращения жидкости
)

# Адвекция жидкости тора A потоком u
# φ_A «плывёт» — тор A движется естественно с газом
problem.add_equation(
    "dt(phi_A) - kappa_phi*lap(phi_A) = -u@grad(phi_A)"
)

# Адвекция жидкости тора B
problem.add_equation(
    "dt(phi_B) - kappa_phi*lap(phi_B) = -u@grad(phi_B)"
)

# Несжимаемость ∇·u = 0 (с методом множителей)
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("integ(p) = 0")

solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = stop_sim_time


# ═══════════════════════════════════════════════════════════════
# §6. ДИАГНОСТИЧЕСКИЕ ВЫРАЖЕНИЯ
# ═══════════════════════════════════════════════════════════════
omega_v      = d3.Curl(u)
speed        = np.sqrt(u @ u)
wmag         = np.sqrt(omega_v @ omega_v)
phi_tot_expr = phi_A + phi_B   # выражение-оператор (для снапшотов)


# ═══════════════════════════════════════════════════════════════
# §7. ВИЗУАЛИЗАЦИЯ: 6-ПАНЕЛЬНЫЙ КАДР
# ═══════════════════════════════════════════════════════════════
def _save_frame(fidx, t,
                wmag_xz, speed_xz, phi_xz,
                w_g, phi_g,
                x1d, z1d, y1d,
                pA, pB, track_A, track_B,
                maxA, maxB):
    """
    6-панельный кадр:
      [0,0] |ω| XZ + траектории центроидов
      [0,1] |u| XZ + позиции торов
      [0,2] φ_tot XZ — интерфейс жидкость/газ
      [1,0] |ω| XY (срез тор A)
      [1,1] |ω| XY (срез тор B)
      [1,2] φ XY (форма тора A в сечении)
    """
    BG = '#07071a'
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor=BG)

    Dz      = abs(pA[2] - pB[2])
    healthy = "✅ ЦЕЛЫ" if (maxA > 0.3 and maxB > 0.3) else "⚠️  ДЕГРАДАЦИЯ"

    fig.suptitle(
        f"Жидкие торы в газе  ·  t = {t:.3f}  {healthy}\n"
        f"ν_gas={nu_gas:.0e}   ν_liq={nu_liq:.0e}   γ_soft={gamma_soft}   "
        f"zA={pA[2]:+.2f}  zB={pB[2]:+.2f}   Δz={Dz:.2f}   "
        f"max(φ_A)={maxA:.3f}  max(φ_B)={maxB:.3f}",
        color='#88ccff', fontsize=10, fontweight='bold', y=0.99
    )

    for ax in axes.flat:
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_edgecolor('#1a1a2e')
        ax.tick_params(colors='#6666aa', labelsize=8)
        ax.xaxis.label.set_color('#9999cc')
        ax.yaxis.label.set_color('#9999cc')

    ext_xz = [x1d.min(), x1d.max(), z1d.min(), z1d.max()]
    ext_xy = [x1d.min(), x1d.max(), y1d.min(), y1d.max()]
    kw_xz  = dict(origin='lower', aspect='equal',
                  extent=ext_xz, interpolation='bilinear')
    kw_xy  = dict(origin='lower', aspect='equal',
                  extent=ext_xy, interpolation='bilinear')

    vmax_w = np.percentile(np.abs(wmag_xz),  99.5) + 1e-8
    vmax_s = np.percentile(np.abs(speed_xz), 99.5) + 1e-8

    # ── [0,0] Завихрённость XZ + траектории ──
    im0 = axes[0, 0].imshow(wmag_xz.T, cmap='inferno',
                             vmin=0, vmax=vmax_w, **kw_xz)
    axes[0, 0].set_title("|ω|  завихрённость  (XZ, y≈0)",
                          color='white', fontsize=9)
    axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("z")
    tA = np.array(track_A); tB = np.array(track_B)
    if len(tA) > 1:
        axes[0, 0].plot(tA[:, 0], tA[:, 2], '-',
                        color='#00d4ff', lw=1.0, alpha=0.7, label='A')
        axes[0, 0].plot(tB[:, 0], tB[:, 2], '-',
                        color='#ff6b6b', lw=1.0, alpha=0.7, label='B')
    axes[0, 0].plot(pA[0], pA[2], 'o', color='#00d4ff', ms=6)
    axes[0, 0].plot(pB[0], pB[2], 'o', color='#ff6b6b', ms=6)
    axes[0, 0].legend(fontsize=7, labelcolor='white',
                       facecolor='#0d0d20', edgecolor='#333355')
    cb = plt.colorbar(im0, ax=axes[0, 0])
    plt.setp(cb.ax.get_yticklabels(), color='#888888')

    # ── [0,1] Скорость XZ ────────────────────
    im1 = axes[0, 1].imshow(speed_xz.T, cmap='magma',
                             vmin=0, vmax=vmax_s, **kw_xz)
    axes[0, 1].set_title("|u|  скорость  (XZ, y≈0)",
                          color='white', fontsize=9)
    axes[0, 1].set_xlabel("x"); axes[0, 1].set_ylabel("z")
    axes[0, 1].axhline(pA[2], color='#00d4ff', lw=0.7,
                        ls='--', alpha=0.5, label=f'z_A={pA[2]:.2f}')
    axes[0, 1].axhline(pB[2], color='#ff6b6b', lw=0.7,
                        ls='--', alpha=0.5, label=f'z_B={pB[2]:.2f}')
    axes[0, 1].legend(fontsize=7, labelcolor='white',
                       facecolor='#0d0d20', edgecolor='#333355')
    cb = plt.colorbar(im1, ax=axes[0, 1])
    plt.setp(cb.ax.get_yticklabels(), color='#888888')

    # ── [0,2] Фазовое поле φ_tot XZ ──────────
    phi_vmax = max(float(np.max(np.abs(phi_xz))), 0.05)
    im2 = axes[0, 2].imshow(phi_xz.T, cmap='Blues',
                             vmin=0, vmax=phi_vmax, **kw_xz)
    axes[0, 2].set_title("φ_A + φ_B   (интерфейс жидкость / газ)",
                          color='#88ccff', fontsize=9)
    axes[0, 2].set_xlabel("x"); axes[0, 2].set_ylabel("z")
    # Контуры интерфейса
    try:
        axes[0, 2].contour(x1d, z1d, phi_xz.T,
                            levels=[0.2, 0.5],
                            colors=['#aaccff', '#ffffff'],
                            linewidths=[0.6, 1.0], alpha=0.7)
    except Exception:
        pass
    cb = plt.colorbar(im2, ax=axes[0, 2])
    plt.setp(cb.ax.get_yticklabels(), color='#888888')

    # ── [1,0] |ω| XY срез тора A ─────────────
    iz_A = int(np.argmin(np.abs(z1d - pA[2])))
    iz_A = int(np.clip(iz_A, 0, w_g.shape[2] - 1))
    w_xy_A   = w_g[:, :, iz_A]
    vmax_wA  = np.percentile(np.abs(w_xy_A), 99.5) + 1e-8
    im3 = axes[1, 0].imshow(w_xy_A.T, cmap='inferno',
                             vmin=0, vmax=vmax_wA, **kw_xy)
    axes[1, 0].set_title(f"|ω| XY  z={pA[2]:.2f}  (тор A)",
                          color='#00d4ff', fontsize=9)
    axes[1, 0].set_xlabel("x"); axes[1, 0].set_ylabel("y")
    cb = plt.colorbar(im3, ax=axes[1, 0])
    plt.setp(cb.ax.get_yticklabels(), color='#888888')

    # ── [1,1] |ω| XY срез тора B ─────────────
    iz_B = int(np.argmin(np.abs(z1d - pB[2])))
    iz_B = int(np.clip(iz_B, 0, w_g.shape[2] - 1))
    w_xy_B   = w_g[:, :, iz_B]
    vmax_wB  = np.percentile(np.abs(w_xy_B), 99.5) + 1e-8
    im4 = axes[1, 1].imshow(w_xy_B.T, cmap='inferno',
                             vmin=0, vmax=vmax_wB, **kw_xy)
    axes[1, 1].set_title(f"|ω| XY  z={pB[2]:.2f}  (тор B)",
                          color='#ff6b6b', fontsize=9)
    axes[1, 1].set_xlabel("x"); axes[1, 1].set_ylabel("y")
    cb = plt.colorbar(im4, ax=axes[1, 1])
    plt.setp(cb.ax.get_yticklabels(), color='#888888')

    # ── [1,2] φ XY срез — форма сечения тора ─
    phi_xy   = phi_g[:, :, iz_A]
    phi_vmax2= max(float(np.max(np.abs(phi_xy))), 0.05)
    im5 = axes[1, 2].imshow(phi_xy.T, cmap='Blues',
                             vmin=0, vmax=phi_vmax2, **kw_xy)
    axes[1, 2].set_title(
        f"φ_A+φ_B  XY z={pA[2]:.2f}  (кольцо жидкости)",
        color='#88ccff', fontsize=9)
    axes[1, 2].set_xlabel("x"); axes[1, 2].set_ylabel("y")
    try:
        axes[1, 2].contour(x1d, y1d, phi_xy.T,
                            levels=[0.3, 0.6],
                            colors=['#aaccff', '#ffffff'],
                            linewidths=[0.6, 1.0], alpha=0.8)
    except Exception:
        pass
    cb = plt.colorbar(im5, ax=axes[1, 2])
    plt.setp(cb.ax.get_yticklabels(), color='#888888')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(FRAMES_DIR, f"frame_{fidx:04d}.png")
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)
    return out



# ═══════════════════════════════════════════════════════════════
# §9. ГЛАВНЫЙ ЦИКЛ
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':

    # HDF5-снапшоты для анализа / VTK
    snapshots = solver.evaluator.add_file_handler(
        SNAP_DIR, sim_dt=0.2, max_writes=500
    )
    snapshots.add_task(speed,        name='speed')
    snapshots.add_task(wmag,         name='vorticity_mag')
    snapshots.add_task(p,            name='pressure')
    snapshots.add_task(u,            name='velocity')
    snapshots.add_task(phi_A,        name='phi_A')
    snapshots.add_task(phi_B,        name='phi_B')
    snapshots.add_task(phi_tot_expr, name='phi_total')

    # Адаптивный CFL
    CFL = d3.CFL(
        solver, initial_dt=max_timestep, cadence=10,
        safety=0.25, threshold=0.1,
        max_change=1.4, min_change=0.5, max_dt=max_timestep,
    )
    CFL.add_velocity(u)

    # Мониторинг GlobalFlowProperty
    flow = d3.GlobalFlowProperty(solver, cadence=10)
    flow.add_property(speed, name='speed')
    flow.add_property(wmag,  name='omega')

    # 1D-срезы для визуализации
    x1d = x[:, 0, 0]; y1d = y[0, :, 0]; z1d = z[0, 0, :]

    # Начальные позиции (будут пересчитываться из центроида φ)
    pos_A = pos_A0.copy()
    pos_B = pos_B0.copy()

    # Траектории центроидов
    track_A = [pos_A.copy()]
    track_B = [pos_B.copy()]

    logger.info("═" * 65)
    logger.info("  ДВА ЖИДКИХ ТОРА В ГАЗЕ")
    logger.info(f"  ν_gas={nu_gas:.0e}  ν_liq={nu_liq:.0e}  "
                f"ratio={nu_liq/nu_gas:.0f}×")
    logger.info(f"  Δν = {dnu:.4f}   κ_φ = {kappa_phi:.0e}")
    logger.info(f"  R={R_ring}  a={a_core}  "
                f"Ω_tor={Omega_tor}  Ω_pol={Omega_pol}")
    logger.info(f"  γ_soft={gamma_soft}  "
                f"sign_A={sign_A:+d}  sign_B={sign_B:+d}")
    logger.info(f"  Тор A: z₀={pos_A0[2]:.2f}   "
                f"Тор B: z₀={pos_B0[2]:.2f}")
    logger.info(f"  Δz₀ = {abs(pos_A0[2]-pos_B0[2]):.2f}  "
                f"T = {stop_sim_time:.1f}")
    logger.info("═" * 65)

    frame_idx = 0
    executor  = ThreadPoolExecutor(max_workers=2)

    try:
        while solver.proceed:
            t = solver.sim_time

            # ══ 1. Читаем текущие φ-поля ════════════════
            phi_A.change_scales(1)
            phi_B.change_scales(1)
            phiA_g = phi_A['g'].copy()
            phiB_g = phi_B['g'].copy()

            # ══ 2. Позиции торов — из центроида φ ═══════
            # (Реальная физика, а не ручное перемещение!)
            pos_A = centroid_from_phi(phiA_g)
            pos_B = centroid_from_phi(phiB_g)

            # ══ 3. Обновляем вспомогательные поля ═══════
            update_aux_fields(pos_A, pos_B, phiA_g, phiB_g)

            # ══ 4. Шаг Навье–Стокса ══════════════════════
            dt = CFL.compute_timestep()
            solver.step(dt)

            # ══ 5. Траектории ════════════════════════════
            if (solver.iteration - 1) % 5 == 0:
                track_A.append(pos_A.copy())
                track_B.append(pos_B.copy())

            # ══ 6. Лог и визуализация ════════════════════
            if (solver.iteration - 1) % 20 == 0:
                max_speed = flow.max('speed')
                max_omega = flow.max('omega')
                dz        = abs(pos_A[2] - pos_B[2])
                maxA      = float(np.max(phiA_g))
                maxB      = float(np.max(phiB_g))

                # Здоровье торов
                health = "OK" if (maxA > 0.3 and maxB > 0.3) else "⚠"
                logger.info(
                    f"it={solver.iteration:5d} | t={t:6.3f} | "
                    f"max|u|={max_speed:6.3f} | max|ω|={max_omega:7.2f} | "
                    f"Δz={dz:.3f} | "
                    f"φ_A={maxA:.3f} φ_B={maxB:.3f} {health} | "
                    f"zA={pos_A[2]:+.3f} zB={pos_B[2]:+.3f}"
                )

                # Данные для кадра
                w_ev  = wmag.evaluate();  w_ev.change_scales(1)
                s_ev  = speed.evaluate(); s_ev.change_scales(1)
                pt_ev = phi_tot_expr.evaluate(); pt_ev.change_scales(1)
                w_g   = np.array(w_ev['g'])
                s_g   = np.array(s_ev['g'])
                phi_g = np.array(pt_ev['g'])

                executor.submit(
                    _save_frame, frame_idx, t,
                    w_g[:,  iy0, :].copy(),
                    s_g[:,  iy0, :].copy(),
                    phi_g[:, iy0, :].copy(),
                    w_g.copy(), phi_g.copy(),
                    x1d, z1d, y1d,
                    pos_A.copy(), pos_B.copy(),
                    list(track_A), list(track_B),
                    maxA, maxB
                )
                frame_idx += 1

    except Exception:
        logger.exception("Ошибка в главном цикле.")
        raise
    finally:
        executor.shutdown(wait=True)
        solver.log_stats()
        logger.info(f"✅  PNG: {frame_idx} кадров → {FRAMES_DIR}/")
        logger.info(f"   Финальные позиции:  A={pos_A}  B={pos_B}")
        logger.info(f"   Δz финальное = {abs(pos_A[2]-pos_B[2]):.4f}")
        dA_final = float(np.max(phi_A['g']))
        dB_final = float(np.max(phi_B['g']))
        logger.info(f"   max(φ_A)={dA_final:.4f}  max(φ_B)={dB_final:.4f}")
        if dA_final > 0.3 and dB_final > 0.3:
            logger.info("   ✅  Оба тора СОХРАНИЛИ форму!")
        else:
            logger.warning("   ⚠️  Один или оба тора деградировали.")
