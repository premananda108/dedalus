
import numpy as np
import dedalus.public as d3
import logging
import matplotlib.pyplot as plt
import os

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not os.path.exists('examples/1_tor/frames_suck_test'):
    os.makedirs('examples/1_tor/frames_suck_test', exist_ok=True)

# Domain
Lx, Ly, Lz = 20, 20, 10
Nx, Ny, Nz = 32, 32, 16 
R_torus = 5.0
g = 100.0

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
# 1. Create Initial State: Torus + Blob in Center
# ==========================================
# Main Torus
psi['g'] = np.exp(-0.5 * ((r_cyl - R_torus)**2 + z**2))
# Test Blob in the center (The "Victim")
psi['g'] += 0.5 * np.exp(-0.5 * (r_cyl**2 + z**2) / 0.5**2) 

# Normalize manually roughly to visible levels
psi['g'] *= 1.0

# Save initial state
psi.change_scales(1)
plt.figure(figsize=(6, 5))
plt.imshow(np.abs(psi['g'][:,:,Nz//2])**2, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
plt.title("Initial: Torus + Blob in Hole")
plt.colorbar()
plt.savefig("examples/1_tor/frames_suck_test/frame_0000.png")
plt.close()

# ==========================================
# 2. Dynamics
# ==========================================
problem = d3.IVP([psi], namespace=locals())
problem.add_equation("dt(psi) - 0.5*1j*div(grad(psi)) = -1j*V_trap*psi - 1j*g*psi*conj(psi)*psi")

solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = 2.0

frame_idx = 1
dt = 5e-3

logger.info("Starting Suck Test...")
while solver.proceed:
    solver.step(dt)
    
    if solver.iteration % 10 == 0:
        logger.info(f"Time: {solver.sim_time:.2f}")
        
        psi.change_scales(1)
        z_slice_idx = Nz // 2
        
        plt.figure(figsize=(6, 5))
        plt.imshow(np.abs(psi['g'][:,:,z_slice_idx])**2, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2], vmax=0.2)
        plt.title(f"Density t={solver.sim_time:.2f}")
        plt.colorbar()
        plt.savefig(f"examples/1_tor/frames_suck_test/frame_{frame_idx:04d}.png")
        plt.close()
        frame_idx += 1
