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
═══════════════════════════════════════════════════════════════════════
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"

import numpy as np
import dedalus.public as d3
import logging

logger = logging.getLogger(__name__)

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
# У земли (z = ±Lz/2) температура +20C. На высоте (z = 0) температура -20C.
T_bg = dist.Field(name='T_bg', bases=bases)
T_bg['g'] = -20.0 * np.cos(2.0 * np.pi * z / Lz)

# ═══════════════════════════════════════════════════════
# §3. УРАВНЕНИЯ С МИКРОФИЗИКОЙ
# ═══════════════════════════════════════════════════════
# Реальная температура = Фоновая(высота) + Аномалия(нагрев) + Эффект давления(торнадо)
T_act = T_bg + b + C_p * p

# Скорости фазовых переходов (используем сглаженную ступеньку tanh для стабильности)
Cond   = R_cond * qv * 0.5 * (1.0 + d3.tanh(5.0 * (T_dew - T_act)))
Freeze = R_freeze * qc * 0.5 * (1.0 + d3.tanh(5.0 * (T_freeze - T_act)))
Melt   = R_melt * qi * 0.5 * (1.0 + d3.tanh(5.0 * (T_act - T_freeze)))

problem = d3.IVP([u, b, p, tau_p, qv, qc, qi], namespace=locals())

# 1. Импульс (Навье-Стокс)
problem.add_equation("dt(u) + grad(p) - nu*lap(u) = -u@grad(u) + b*ez")

# 2. Тепло (Нагрев от конденсации и замерзания, охлаждение от таяния)
problem.add_equation("dt(b) - kappa*lap(b) = -u@grad(b) + L_v*Cond + L_f*Freeze - L_f*Melt")

# 3. Водяной пар (Убывает при конденсации)
problem.add_equation("dt(qv) - kappa*lap(qv) = -u@grad(qv) - Cond")

# 4. Облачная вода (Появляется из пара, исчезает при замерзании, появляется при таянии града)
problem.add_equation("dt(qc) - kappa*lap(qc) = -u@grad(qc) + Cond - Freeze + Melt")

# 5. ГРАД (Летит ВНИЗ со скоростью V_fall относительно потока воздуха!)
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

# Вращение
v_theta = (Gamma / (2.0 * np.pi)) / rho_safe * (1.0 - np.exp(-rho_sq / r_c**2))
# Апдрафт (направлен вверх от "земли" z=-Lz/2 к "небу" z=0)
v_z     = w0 * np.exp(-rho_sq / r_c**2) * (-np.sin(2.0 * np.pi * z / Lz))
dg_dz   = (2.0 * np.pi / Lz) * (-np.cos(2.0 * np.pi * z / Lz))
v_rho   = -(w0 * r_c**2 / (2.0 * rho_safe)) * (1.0 - np.exp(-rho_sq / r_c**2)) * dg_dz

u['g'][0] = -v_theta * (y / rho_safe) + v_rho * (x / rho_safe)
u['g'][1] =  v_theta * (x / rho_safe) + v_rho * (y / rho_safe)
u['g'][2] =  v_z

# Теплый пузырь у земли для старта конвекции
b['g'] = 2.0 * np.exp(-rho_sq / 0.5**2) * np.exp(-(z + Lz/2)**2 / 0.5)

# Озеро водяного пара у земли
qv['g'] = 3.0 * np.exp(-(z + Lz/2)**2 / 0.5) + 3.0 * np.exp(-(z - Lz/2)**2 / 0.5)

# ═══════════════════════════════════════════════════════
# §5. ДИАГНОСТИКА И ЗАПУСК
# ═══════════════════════════════════════════════════════
speed = np.sqrt(u @ u)

if __name__ == '__main__':
    snapshots = solver.evaluator.add_file_handler("snapshots_hailstorm", sim_dt=0.05, max_writes=400)
    snapshots.add_task(speed, name="speed")
    snapshots.add_task(p,     name="pressure")
    snapshots.add_task(T_act, name="Temperature")
    snapshots.add_task(qv,    name="Vapor")
    snapshots.add_task(qc,    name="Cloud_Water") # Воронка торнадо
    snapshots.add_task(qi,    name="Hail_Ice")    # Град!

    CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.2, max_dt=max_timestep)
    CFL.add_velocity(u)

    flow = d3.GlobalFlowProperty(solver, cadence=10)
    flow.add_property(speed, name='speed')
    flow.add_property(qc, name='cloud_max')
    flow.add_property(qi, name='hail_max')

    logger.info("══════════════════════════════════════════════════")
    logger.info(" Запуск симуляции: Торнадо + Фабрика Града")
    logger.info(" Ожидайте: Пар -> Облако -> Град -> Падение града")
    logger.info("══════════════════════════════════════════════════")
    
    try:
        while solver.proceed:
            dt = CFL.compute_timestep()
            solver.step(dt)
            if (solver.iteration - 1) % 20 == 0:
                logger.info(
                    f"it={solver.iteration:5d} | t={solver.sim_time:4.2f} | "
                    f"max|u|={flow.max('speed'):5.2f} | "
                    f"Max Cloud={flow.max('cloud_max'):5.2f} | "
                    f"Max Hail={flow.max('hail_max'):5.2f}"
                )
    except Exception:
        logger.exception("Ошибка симуляции.")
        raise
    finally:
        solver.log_stats()