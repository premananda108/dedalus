import numpy as np
import dedalus.public as d3
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Отключаем многопоточность для ускорения Dedalus
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

output_dir = 'examples/1_tor/frames_gpe_collision'
os.makedirs(output_dir, exist_ok=True)

# Domain
Lx, Ly, Lz = 24, 24, 24
Nx, Ny, Nz = 48, 48, 48  # Оптимальное разрешение для скорости (Гросс-Питаевский считается дольше гидродинамики)
R_torus = 5.0
g = 100.0  # Увеличили самодействие (repulsion) до 100, как в вашем тесте

coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype=np.complex128)
xbasis = d3.ComplexFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=3/2)
ybasis = d3.ComplexFourier(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2), dealias=3/2)
zbasis = d3.ComplexFourier(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=3/2)

psi = dist.Field(name='psi', bases=(xbasis, ybasis, zbasis))
V_trap = dist.Field(name='V', bases=(xbasis, ybasis, zbasis))

x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
r_cyl = np.sqrt(x**2 + y**2)

# ==========================================
# 1. Ловушка (Внешний потенциал)
# ==========================================
# В отличие от классической жидкости, "квантовый газ" (конденсат) 
# держится в форме тора за счет внешней магнитной или оптической ловушки.
# Этот потенциал затягивает вещество в радиус R_torus и на высоту z=0.
V_trap['g'] = 0.5 * ((r_cyl - R_torus)**2 + z**2)

# ==========================================
# 2. Начальное условие: Два кольца
# ==========================================
# Создаем два кольца: одно выше центра (ловушки), другое ниже.
# Поскольку ловушка стремится стянуть всё к z=0, кольца сами "упадут" друг на друга!
z1 = -5.0
z2 = 5.0

psi.change_scales(1)
theta = np.arctan2(y, x)

# Кольцо 1 закручено по часовой стрелке (m = 1)
psi_1 = np.exp(-0.5 * ((r_cyl - R_torus)**2 + (z - z1)**2)) * np.exp(1j * 1 * theta)

# Кольцо 2 закручено против часовой стрелки (m = -1)
psi_2 = np.exp(-0.5 * ((r_cyl - R_torus)**2 + (z - z2)**2)) * np.exp(1j * -1 * theta)

psi['g'] = psi_1 + psi_2

def normalize_psi(field):
    norm = d3.Integrate(field * d3.conj(field)).evaluate()['g'][0,0,0]
    if np.real(norm) > 1e-10:
        field['g'] /= np.sqrt(np.real(norm))
        field['g'] *= np.sqrt(2.0) # Частиц в 2 раза больше (на 2 кольца)

normalize_psi(psi)

# ==========================================
# 3. Динамика (Уравнение Гросса-Питаевского)
# ==========================================
problem = d3.IVP([psi], namespace=locals())
# Это то же самое квантовое уравнение Шредингера из torus_drag_test.py
problem.add_equation("dt(psi) - 0.5*1j*div(grad(psi)) = -1j*V_trap*psi - 1j*g*psi*conj(psi)*psi")

solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = 10.0

logger.info("Starting GPE Collision...")
frame_idx = 0
dt = 2e-3 

while solver.proceed:
    solver.step(dt)
    
    if solver.iteration % 20 == 0:
        t = solver.sim_time
        logger.info(f"Time: {t:.2f}")
        
        psi.change_scales(1)
        z_slice_idx = Nz // 2
        y_slice_idx = Ny // 2
        
        # Вычисляем ПЛОТНОСТЬ как |psi|^2
        dens_xy = np.abs(psi['g'][:,:,z_slice_idx])**2
        dens_xz = np.abs(psi['g'][:, y_slice_idx, :])**2
        
        plt.figure(figsize=(10, 5))
        
        # 1. Вид сверху (XY)
        plt.subplot(1, 2, 1)
        plt.imshow(dens_xy.T, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2], cmap='inferno', origin='lower')
        plt.title(f"Top View (XY, z=0) t={t:.2f}")
        plt.xlabel("x")
        plt.ylabel("y")
        
        # 2. Вид сбоку (XZ)
        plt.subplot(1, 2, 2)
        plt.imshow(dens_xz.T, extent=[-Lx/2, Lx/2, -Lz/2, Lz/2], cmap='inferno', origin='lower')
        plt.title(f"Side View (XZ, y=0) t={t:.2f}")
        plt.xlabel("x")
        plt.ylabel("z")
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/frame_{frame_idx:04d}.png", dpi=100)
        plt.close()
        frame_idx += 1
