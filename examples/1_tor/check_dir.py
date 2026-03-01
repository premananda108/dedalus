import numpy as np
import matplotlib
import sys
matplotlib.use('Agg')
import matplotlib.pyplot as plt

Lx = Ly = Lz = 2*np.pi
Nx = Ny = Nz = 64
R0 = 1.0
a = 0.25
z1 = -1.0
A0_1 = 0.35
z2 = 1.0
A0_2 = -0.35

x_1d = np.linspace(-Lx/2, Lx/2, Nx)
y_1d = np.linspace(-Ly/2, Ly/2, Ny)
z_1d = np.linspace(-Lz/2, Lz/2, Nz)

x, y, z = np.meshgrid(x_1d, y_1d, z_1d, indexing='ij')
rho = np.sqrt(x**2 + y**2)

s2_1 = (rho - R0)**2 + (z - z1)**2
E_1 = np.exp(-s2_1 / a**2)
u_z_1 = 2 * (A0_1 / R0) * E_1 - (A0_1 * rho / R0) * (2 * (rho - R0) / a**2) * E_1
# self-induced velocity: evaluate at core
core_idx_z1 = np.argmin(np.abs(z_1d - z1))
core_idx_rho = Nx//2 + int(R0 / (Lx/Nx))
uz_bot = u_z_1[core_idx_rho, Ny//2, core_idx_z1]

s2_2 = (rho - R0)**2 + (z - z2)**2
E_2 = np.exp(-s2_2 / a**2)
u_z_2 = 2 * (A0_2 / R0) * E_2 - (A0_2 * rho / R0) * (2 * (rho - R0) / a**2) * E_2
core_idx_z2 = np.argmin(np.abs(z_1d - z2))
uz_top = u_z_2[core_idx_rho, Ny//2, core_idx_z2]

with open("/Users/premananda/Desktop/MyProjects/dedalus/examples/1_tor/vel_output.txt", "w") as f:
    f.write(f"Bottom ring (z=-1) moves in Z: {uz_bot}\n")
    f.write(f"Top ring (z=1) moves in Z: {uz_top}\n")
