
import numpy as np
import dedalus.public as d3
import logging
import matplotlib.pyplot as plt
import os

# 1. Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Output directory
if not os.path.exists('examples/1_tor/frames_vortex'):
    os.makedirs('examples/1_tor/frames_vortex', exist_ok=True)

# Resolution (keep low for speed, increase for quality)
Lx, Ly, Lz = 20, 20, 10
Nx, Ny, Nz = 32, 32, 16 

# Physics
R_torus = 5.0
g = 100.0
N_particles = 1.0

# 2. Basis
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype=np.complex128)
xbasis = d3.ComplexFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=3/2)
ybasis = d3.ComplexFourier(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2), dealias=3/2)
zbasis = d3.ComplexFourier(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=3/2)

# Fields
psi = dist.Field(name='psi', bases=(xbasis, ybasis, zbasis))
V_trap = dist.Field(name='V', bases=(xbasis, ybasis, zbasis))

# 3. Potential & Initial Cloud
x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
r_cyl = np.sqrt(x**2 + y**2)
V_trap['g'] = 0.5 * ((r_cyl - R_torus)**2 + z**2)

# Initial guess: noisy cloud around ring
psi['g'] = np.exp(-0.5 * ((r_cyl - R_torus)**2 + z**2)) + 0.1 * np.random.randn(*psi['g'].shape)

# helper for normalization
def normalize_psi(field):
    norm = d3.Integrate(field * d3.conj(field)).evaluate()['g'][0,0,0]
    if np.real(norm) > 1e-10:
        field['g'] /= np.sqrt(np.real(norm))
        field['g'] *= np.sqrt(N_particles)

# ==========================================
# PHASE 1: Imaginary Time (Find Ground State)
# ==========================================
logger.info("--- PHASE 1: Finding Ground State (Imaginary Time) ---")
problem_im = d3.IVP([psi], namespace=locals())
problem_im.add_equation("dt(psi) - 0.5*div(grad(psi)) = -V_trap*psi - g*psi*conj(psi)*psi")

solver_im = problem_im.build_solver(d3.RK222)
solver_im.stop_sim_time = 1.0 # Shortened for speed

dt_im = 5e-3
while solver_im.proceed:
    solver_im.step(dt_im)
    if solver_im.iteration % 10 == 0:
        normalize_psi(psi)
    if solver_im.iteration % 50 == 0:
        logger.info(f"Imaginary Time: {solver_im.sim_time:.2f}")

# ==========================================
# PHASE 2: Imprint Vortex Phase
# ==========================================
logger.info("--- PHASE 2: Imprinting Vortex Phase ---")
# Create circulation around the z-axis (charge m=1)
psi.change_scales(1)
theta = np.arctan2(y, x)
m_charge = 1
psi['g'] *= np.exp(1j * m_charge * theta)
normalize_psi(psi)

# Visualize the phase imprint
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Density (Ground State)")
plt.imshow(np.abs(psi['g'][:,:,Nz//2])**2, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
plt.subplot(1, 2, 2)
plt.title("Phase (Imprinted Vortex)")
plt.imshow(np.angle(psi['g'][:,:,Nz//2]), extent=[-Lx/2, Lx/2, -Ly/2, Ly/2], cmap='hsv')
plt.savefig("examples/1_tor/frames_vortex/initial_state.png")
plt.close()

# ==========================================
# PHASE 3: Real Time Evolution (Dynamics)
# ==========================================
logger.info("--- PHASE 3: Real Time Dynamics ---")

# Equation: i*dt(psi) = -0.5*div(grad(psi)) + V*psi + g*|psi|^2*psi
# Multiply by -i: dt(psi) = 0.5*i*div(grad(psi)) - i*V*psi - i*g*|psi|^2*psi
problem_real = d3.IVP([psi], namespace=locals())
problem_real.add_equation("dt(psi) - 0.5*1j*div(grad(psi)) = -1j*V_trap*psi - 1j*g*psi*conj(psi)*psi")

solver_real = problem_real.build_solver(d3.RK222)
solver_real.stop_sim_time = 4.0 # Run long enough to see rotation/movement

dt_real = 2e-3 # Smaller timestep for real dynamics usually better
frame_idx = 0

while solver_real.proceed:
    solver_real.step(dt_real)
    
    if solver_real.iteration % 20 == 0:
        logger.info(f"Real Time: {solver_real.sim_time:.3f}")
        
        # Save frame: Plot Phase to see the rotation!
        psi.change_scales(1)
        z_slice_idx = Nz // 2
        
        plt.figure(figsize=(10, 4))
        
        # Plot Density
        plt.subplot(1, 2, 1)
        plt.title(f"Density t={solver_real.sim_time:.2f}")
        plt.imshow(np.abs(psi['g'][:,:,z_slice_idx])**2, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2], vmin=0, vmax=0.05)
        plt.colorbar()
        
        # Plot Phase
        plt.subplot(1, 2, 2)
        plt.title(f"Phase t={solver_real.sim_time:.2f}")
        plt.imshow(np.angle(psi['g'][:,:,z_slice_idx]), extent=[-Lx/2, Lx/2, -Ly/2, Ly/2], cmap='hsv')
        plt.colorbar()
        
        plt.savefig(f"examples/1_tor/frames_vortex/frame_{frame_idx:04d}.png")
        plt.close()
        frame_idx += 1

logger.info("Done!")
