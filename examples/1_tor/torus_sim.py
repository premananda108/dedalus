
import numpy as np
import dedalus.public as d3
import logging
import matplotlib.pyplot as plt

# 1. Настройки симуляции
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Размер коробки и разрешение
Lx, Ly, Lz = 20, 20, 10
Nx, Ny, Nz = 32, 32, 16  # Reduced for speed

# Физические параметры
R_torus = 5.0  # Радиус самого кольца (расстояние от центра до оси трубки)
g = 100.0      # Сила отталкивания атомов (нелинейность)
N_particles = 1.0 # Нормировка

# 2. Создаем пространство (Базис)
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype=np.complex128)
xbasis = d3.ComplexFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=3/2)
ybasis = d3.ComplexFourier(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2), dealias=3/2)
zbasis = d3.ComplexFourier(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=3/2)

# Поля
psi = dist.Field(name='psi', bases=(xbasis, ybasis, zbasis))
V_trap = dist.Field(name='V', bases=(xbasis, ybasis, zbasis))

# 3. Задаем Тороидальный Потенциал (Ловушку)
x, y, z = dist.local_grids(xbasis, ybasis, zbasis)

# Формула потенциала: V = 0.5 * ((r - R)^2 + z^2)
r_cyl = np.sqrt(x**2 + y**2)
V_trap['g'] = 0.5 * ((r_cyl - R_torus)**2 + z**2)

# 4. Начальное состояние (Случайный шум или облако)
# Начнем с облака, "размазанного" где-то рядом с кольцом
psi['g'] = np.exp(-0.5 * ((r_cyl - R_torus)**2 + z**2)) + 0.1 * np.random.randn(*psi['g'].shape)
psi.change_scales(1) # Применить изменения

# 5. Уравнение Гросса-Питаевского в Мнимом Времени (tau = -i * t)
# Обычное GPE: i * d_psi/dt = -0.5*Laplace(psi) + V*psi + g*|psi|^2*psi
# Mнимое время: d_psi/d_tau = 0.5*Laplace(psi) - V*psi - g*|psi|^2*psi
# В Dedalus переносим линейные члены влево, нелинейные вправо.

problem = d3.IVP([psi], namespace=locals())
problem.add_equation("dt(psi) - 0.5*div(grad(psi)) = -V_trap*psi - g * psi * conj(psi) * psi")

# 6. Солвер
solver = problem.build_solver(d3.RK222) # Рунге-Кутта 2-го порядка
solver.stop_sim_time = 2.0 # Reduced simulation time

# Output directory
import os
if not os.path.exists('frames'):
    os.mkdir('frames')

# В мнимом времени амплитуда падает, нужно возвращать число частиц к 1
def normalize_psi(field):
    # Интеграл |psi|^2 dV
    norm = d3.Integrate(field * d3.conj(field)).evaluate()['g'][0,0,0]
    # Prevent division by zero or negative sqrt if things go bad, though unlikely here
    if np.real(norm) > 1e-10:
        field['g'] /= np.sqrt(np.real(norm))
        field['g'] *= np.sqrt(N_particles)

# Цикл расчета
try:
    logger.info('Starting loop')
    dt = 5e-3 # Increased timestep
    while solver.proceed:
        solver.step(dt) 
        
        if solver.iteration % 20 == 0:
            normalize_psi(psi)
            logger.info(f"Iteration: {solver.iteration}, Time: {solver.sim_time:.3f}")
            
            # Save frame
            psi.change_scales(1)
            psi_slice = psi['g'][:, :, Nz//2]
            plt.figure(figsize=(6, 5))
            plt.imshow(np.abs(psi_slice)**2, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
            plt.title(f"Density at t={solver.sim_time:.3f}")
            plt.colorbar()
            plt.savefig(f"frames/torus_{solver.iteration:04d}.png")
            plt.close()

except Exception as e:
    logger.error('Error in loop')
    logger.error(e)
    raise e

# 7. Сохранение / Визуализация
# Здесь можно сохранить данные в HDF5 или просто вывести срез

# Берем срез по Z=0 (плоскость кольца)
psi.change_scales(1)
psi_slice = psi['g'][:, :, Nz//2] # Индекс середины по Z

plt.figure(figsize=(8, 6))
plt.imshow(np.abs(psi_slice)**2, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
plt.title("Плотность газа (срез тора)")
plt.colorbar()
output_filename = "torus_density.png"
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
