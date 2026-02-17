import numpy as np
import dedalus.public as d3
import logging
import matplotlib.pyplot as plt
import os

# Настройки логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Папка для сохранения кадров
output_dir = 'examples/1_tor/pump_2views'
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# 1. Параметры пространства и сетки
Lx, Ly, Lz = 24, 24, 16
Nx, Ny, Nz = 48, 48, 24  # Хорошее разрешение для вихрей
R_torus = 6.0            # Радиус кольца
thickness = 1.2          # Толщина "трубы" бублика
g = 50.0                 # Сила взаимодействия

# Базис (Фурье - периодические граничные условия)
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype=np.complex128)
xbasis = d3.ComplexFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=3/2)
ybasis = d3.ComplexFourier(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2), dealias=3/2)
zbasis = d3.ComplexFourier(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=3/2)

# Поля
psi = dist.Field(name='psi', bases=(xbasis, ybasis, zbasis))
V_trap = dist.Field(name='V', bases=(xbasis, ybasis, zbasis))

# Сетки в физическом пространстве
x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
r_cyl = np.sqrt(x**2 + y**2)

# 2. Потенциал ловушки (удерживает только сам тор, газ свободен)
# Потенциал имеет форму "желоба" по кругу R=6
V_trap['g'] = 0.5 * ((r_cyl - R_torus)**2 + z**2)

# 3. Начальное состояние: Тор + Облака газа
psi.change_scales(1)

# А) Сам Тор (плотный)
psi_torus = 1.2 * np.exp(-0.5 * ((r_cyl - R_torus)**2 + z**2) / thickness**2)

# Б) Окружающий газ ("облака")
# Добавляем случайные пятна (шум), чтобы было видно движение газа!
# Без шума газ прозрачен и движение не заметно.
np.random.seed(101)
# Создаем низкочастотный шум (пятна)
noise = np.random.normal(0, 1, size=psi_torus.shape)
# Делаем газ плотностью 0.1 с вариациями
psi_gas = 0.15 + 0.05 * np.sin(2*x) * np.cos(2*z) + 0.03 * noise

psi['g'] = psi_torus + psi_gas

# 4. "Включаем Насос" (Полоидальная фаза)
# Фаза закручивается вокруг СЕЧЕНИЯ тора (в плоскости r-z)
# Это создает поток сквозь центр бублика (вдоль Z)
theta_pump = np.arctan2(z, r_cyl - R_torus)
m_pump = 1  # Заряд вихря (1 = один оборот фазы)

psi['g'] *= np.exp(-1j * m_pump * theta_pump)

logger.info("Pump (Poloidal) phase applied.")

# Уравнение Гросса-Питаевского
problem = d3.IVP([psi], namespace=locals())
problem.add_equation("dt(psi) - 0.5*1j*div(grad(psi)) = -1j*V_trap*psi - 1j*g*psi*conj(psi)*psi")

solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = 4.0

# Цикл симуляции
frame_idx = 0
dt = 2e-3 

logger.info("Starting simulation...")
while solver.proceed:
    solver.step(dt)
    
    if solver.iteration % 25 == 0:
        t = solver.sim_time
        logger.info(f"Time: {t:.2f}")
        
        psi.change_scales(1)
        
        # Данные для графиков
        dens_xy = np.abs(psi['g'][:,:,Nz//2])**2  # Срез посередине Z (вид сверху)
        dens_xz = np.abs(psi['g'][:,Ny//2,:])**2  # Срез посередине Y (вид сбоку)
        
        # Настройка фигуры: 2 подграфика рядом
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # --- График 1: Вид Сверху (XY) ---
        # Показывает форму кольца
        ax1 = axes[0]
        im1 = ax1.imshow(dens_xy.T, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2], 
                         cmap='inferno', origin='lower', vmax=1.2)
        ax1.set_title(f"Top View (XY) t={t:.2f}")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        
        # --- График 2: Вид Сбоку (XZ) ---
        # Показывает "разрез" и поток газа
        ax2 = axes[1]
        # ВАЖНО: vmax=0.4 — мы специально "пересвечиваем" тор (он будет белым),
        # чтобы увидеть слабый газ (сине-оранжевый) вокруг.
        im2 = ax2.imshow(dens_xz.T, extent=[-Lx/2, Lx/2, -Lz/2, Lz/2], 
                         cmap='inferno', origin='lower', vmax=0.4)
        
        ax2.set_title(f"Side View (XZ) - PUMPING\nLow vmax to see gas flow")
        ax2.set_xlabel("x")
        ax2.set_ylabel("z")
        
        # Добавим стрелки потока (примерно), чтобы понимать направление
        # (Просто для оформления на центральной оси)
        if t > 0.1:
            ax2.arrow(0, -2, 0, 4, head_width=1, head_length=1, fc='cyan', ec='cyan', alpha=0.5)
            ax2.text(1, 0, "Gas Flow", color='cyan', fontsize=9)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/frame_{frame_idx:04d}.png")
        plt.close()
        
        frame_idx += 1