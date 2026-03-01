import numpy as np
Lx = Ly = Lz = 2*np.pi
Nx = Ny = Nz = 64
R0 = 1.0; a = 0.25; z1 = -1.0; A0_1 = 0.35; z2 = 1.0; A0_2 = -0.35
x_1d = np.linspace(-Lx/2, Lx/2, Nx); y_1d = np.linspace(-Ly/2, Ly/2, Ny); z_1d = np.linspace(-Lz/2, Lz/2, Nz)
x, y, z = np.meshgrid(x_1d, y_1d, z_1d, indexing='ij')
rho = np.sqrt(x**2 + y**2)

# Ring 1
s2_1 = (rho - R0)**2 + (z - z1)**2
E_1 = np.exp(-s2_1 / a**2)
u_z_1 = 2 * (A0_1 / R0) * E_1 - (A0_1 * rho / R0) * (2 * (rho - R0) / a**2) * E_1

# Find center of ring 1 core
cx = np.argmin(np.abs(x_1d - R0)); cy = np.argmin(np.abs(y_1d - 0)); cz = np.argmin(np.abs(z_1d - z1))
print(f"Bottom ring core u_z = {u_z_1[cx, cy, cz]}")

# Ring 2
s2_2 = (rho - R0)**2 + (z - z2)**2
E_2 = np.exp(-s2_2 / a**2)
u_z_2 = 2 * (A0_2 / R0) * E_2 - (A0_2 * rho / R0) * (2 * (rho - R0) / a**2) * E_2

cz2 = np.argmin(np.abs(z_1d - z2))
print(f"Top ring core u_z = {u_z_2[cx, cy, cz2]}")
