import os
# Ограничиваем потоки для скорости
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"

import numpy as np
import dedalus.public as d3
import logging
import matplotlib.pyplot as plt

# Настройка
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

output_dir = 'examples/1_tor/smoke_2views'
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# --- ПАРАМЕТРЫ ---
Lx, Ly, Lz = 20, 20, 16  # Высота 16, чтобы видеть газ сверху
Nx, Ny, Nz = 64, 64, 64  # Хорошее разрешение
R_torus = 5.0
g = 20.0
omega_trap = 2.0

coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype=np.complex128)
xbasis = d3.ComplexFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=3/2)
ybasis = d3.ComplexFourier(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2), dealias=3/2)
zbasis = d3.ComplexFourier(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=3/2)

psi = dist.Field(name='psi', bases=(xbasis, ybasis, zbasis))
V_trap = dist.Field(name='V', bases=(xbasis, ybasis, zbasis))

x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
r_cyl = np.sqrt(x**2 + y**2)

# Ловушка (держит форму тора)
V_trap['g'] = 0.5 * (omega_trap**2) * ((r_cyl - R_torus)**2 + z**2)

# ==========================================
# 1. СОЗДАЕМ СЦЕНУ
# ==========================================
psi.change_scales(1)

# А) Сам ТОР (Плотность ~ 4.0)
torus = 2.0 * np.exp(-0.5 * ((r_cyl - R_torus)**2 + z**2) / 0.8)

# Б) ОБЛАКО ГАЗА ("Дым") над дыркой (Плотность ~ 1.0)
# Сдвинуто по Z вверх (+3.5)
smoke_blob = 1.0 * np.exp(-0.5 * (x**2 + y**2 + (z - 3.5)**2) / 1.5**2)

# В) Слабый фон (чтобы не делить на ноль)
background = 0.05

psi['g'] = torus + smoke_blob + background

# Г) ВКЛЮЧАЕМ НАСОС
# m_pump = -1 тянет сверху вниз (z+ -> z-)
theta_pump = np.arctan2(z, r_cyl - R_torus)
psi['g'] *= np.exp(-1j * (-1) * theta_pump)

logger.info("Setup complete. Blob is above the hole.")

# ==========================================
# 2. ЗАПУСК И ВИЗУАЛИЗАЦИЯ
# ==========================================
problem = d3.IVP([psi], namespace=locals())
problem.add_equation("dt(psi) - 0.5*1j*div(grad(psi)) = -1j*V_trap*psi - 1j*g*psi*conj(psi)*psi")

solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = 4.0
dt = 5e-4 

frame_idx = 0

while solver.proceed:
    solver.step(dt)
    
    if solver.iteration % 100 == 0:
        t = solver.sim_time
        logger.info(f"Time: {t:.2f}")
        
        psi.change_scales(1)
        dens = np.abs(psi['g'])**2
        
        # Срезы
        dens_xy = dens[:,:,Nz//2] # Вид сверху (центр тора)
        dens_xz = dens[:,Ny//2,:] # Вид сбоку (разрез)
        
        # Рисуем
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # --- ГРАФИК 1: ВИД СВЕРХУ (XY) ---
        ax1 = axes[0]
        # Здесь шкала полная, чтобы видеть кольцо целиком
        im1 = ax1.imshow(dens_xy.T, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2], 
                         cmap='inferno', origin='lower', vmax=4.0)
        ax1.set_title(f"Top View (XY) t={t:.2f}")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        
        # --- ГРАФИК 2: ВИД СБОКУ (XZ) - САМОЕ ВАЖНОЕ ---
        ax2 = axes[1]
        
        # ХИТРОСТЬ: vmax=1.2 (хотя тор имеет плотность 4.0).
        # Тор будет ПЕРЕСВЕЧЕН (белые столбы), зато мы увидим газ!
        im2 = ax2.imshow(dens_xz.T, extent=[-Lx/2, Lx/2, -Lz/2, Lz/2], 
                         cmap='inferno', origin='lower', 
                         vmin=0.05, vmax=1.2)
        
        ax2.set_title("Side View (XZ)\nWatch the blob go DOWN")
        ax2.set_xlabel("x")
        ax2.set_ylabel("z")
        
        # Рисуем стрелку направления потока
        ax2.arrow(0, 4, 0, -3, head_width=0.5, color='cyan', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/frame_{frame_idx:04d}.png")
        plt.close()
        frame_idx += 1