
import numpy as np
import dedalus.public as d3
import logging
import matplotlib.pyplot as plt
import os

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not os.path.exists('examples/1_tor/frames_drag_test'):
    os.makedirs('examples/1_tor/frames_drag_test', exist_ok=True)

# Domain
Lx, Ly, Lz = 20, 20, 10
Nx, Ny, Nz = 32, 32, 16 
R_torus = 5.0
g = 100.0
N_particles = 1.0

coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype=np.complex128)
xbasis = d3.ComplexFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=3/2)
ybasis = d3.ComplexFourier(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2), dealias=3/2)
zbasis = d3.ComplexFourier(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=3/2)

psi = dist.Field(name='psi', bases=(xbasis, ybasis, zbasis))
V_trap = dist.Field(name='V', bases=(xbasis, ybasis, zbasis))

x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
r_cyl = np.sqrt(x**2 + y**2)
V_trap['g'] = 0.5 * ((r_cyl - R_torus)**2 + z**2)

# ==========================================
# 0. Initial Relaxed Torus
# ==========================================
psi.change_scales(1)
psi['g'] = np.exp(-0.5 * ((r_cyl - R_torus)**2 + z**2))

def normalize_psi(field):
    norm = d3.Integrate(field * d3.conj(field)).evaluate()['g'][0,0,0]
    if np.real(norm) > 1e-10:
        field['g'] /= np.sqrt(np.real(norm))
        field['g'] *= np.sqrt(N_particles)

problem_im = d3.IVP([psi], namespace=locals())
problem_im.add_equation("dt(psi) - 0.5*div(grad(psi)) = -V_trap*psi - g*psi*conj(psi)*psi")
solver_im = problem_im.build_solver(d3.RK222)
solver_im.stop_sim_time = 0.5 # Just enough to smooth it out

logger.info("Softening Torus...")
while solver_im.proceed:
    solver_im.step(5e-3)
    if solver_im.iteration % 10 == 0:
        normalize_psi(psi)

# ==========================================
# 1. Add Satellite + Spin Up
# ==========================================
logger.info("Adding Satellite and Spin...")
psi.change_scales(1)
# Create satellite far enough: R=8
satellite = 0.3 * np.exp(-0.5 * ((x - 8.0)**2 + y**2 + z**2) / 0.8**2) 
psi['g'] += satellite
normalize_psi(psi)

# Imprint Vortex Phase (m=1 is safer for stability than m=2)
# Ensure scale is 1
psi.change_scales(1)
m_charge = 1
theta = np.arctan2(y, x)
psi['g'] *= np.exp(1j * m_charge * theta)

# Save Frame 0
psi.change_scales(1)
plt.figure(figsize=(6, 5))
plt.imshow(np.abs(psi['g'][:,:,Nz//2])**2, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2], cmap='inferno')
plt.title("Initial: R=5 Torus + R=8 Satellite (m=1)")
plt.colorbar()
# Mark the satellite start
plt.scatter([8], [0], color='white', marker='x') 
plt.savefig("examples/1_tor/frames_drag_test/frame_0000.png")
plt.close()

# ==========================================
# 2. Dynamics
# ==========================================
problem = d3.IVP([psi], namespace=locals())
problem.add_equation("dt(psi) - 0.5*1j*div(grad(psi)) = -1j*V_trap*psi - 1j*g*psi*conj(psi)*psi")

solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = 4.0

frame_idx = 1
dt = 2e-3 # Smaller timestep for safety

logger.info("Starting Drag Test (Real Time)...")
while solver.proceed:
    solver.step(dt)
    
    if solver.iteration % 20 == 0:
        t = solver.sim_time
        logger.info(f"Time: {t:.2f}")
        
        psi.change_scales(1)
        z_slice_idx = Nz // 2
        y_slice_idx = Ny // 2
        
        # Data for Top View (XY)
        dens_xy = np.abs(psi['g'][:,:,z_slice_idx])**2
        
        # Data for Side View (XZ) - Slice at Y=0 (middle)
        # psi['g'] is (Nx, Ny, Nz) -> slice at axis 1 -> (Nx, Nz)
        # For classic torus side view: X horizontal, Z vertical
        # This shows the two sides of the ring (at x=+R and x=-R)
        dens_xz = np.abs(psi['g'][:, y_slice_idx, :])**2
        
        plt.figure(figsize=(10, 5))
        
        # 1. Top View (XY)
        plt.subplot(1, 2, 1)
        plt.imshow(dens_xy.T, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2], cmap='inferno', origin='lower', vmax=dens_xy.max())
        plt.title(f"Top View (XY) t={t:.2f}")
        plt.xlabel("x")
        plt.ylabel("y")
        
        # 2. Side View (XZ)
        plt.subplot(1, 2, 2)
        plt.imshow(dens_xz.T, extent=[-Lx/2, Lx/2, -Lz/2, Lz/2], cmap='inferno', origin='lower', vmax=dens_xy.max())
        plt.title(f"Side View (XZ) t={t:.2f}")
        plt.xlabel("x")
        plt.ylabel("z")
        
        plt.tight_layout()
        plt.savefig(f"examples/1_tor/frames_drag_test/frame_{frame_idx:04d}.png")
        plt.close()
        frame_idx += 1
