import numpy as np

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

# Ring 1
s2_1 = (rho - R0)**2 + (z - z1)**2
E_1 = np.exp(-s2_1 / a**2)
u_x_1 = x * (2 * A0_1 * (z - z1) / (R0 * a**2)) * E_1
u_y_1 = y * (2 * A0_1 * (z - z1) / (R0 * a**2)) * E_1
u_z_1 = 2 * (A0_1 / R0) * E_1 - (A0_1 * rho / R0) * (2 * (rho - R0) / a**2) * E_1

# Ring 2
s2_2 = (rho - R0)**2 + (z - z2)**2
E_2 = np.exp(-s2_2 / a**2)
u_x_2 = x * (2 * A0_2 * (z - z2) / (R0 * a**2)) * E_2
u_y_2 = y * (2 * A0_2 * (z - z2) / (R0 * a**2)) * E_2
u_z_2 = 2 * (A0_2 / R0) * E_2 - (A0_2 * rho / R0) * (2 * (rho - R0) / a**2) * E_2

uz = u_z_1 + u_z_2

print(f"u_z at z=-1: {uz[Nx//2, Ny//2, np.argmin(np.abs(z_1d - (-1.0)))]}")
print(f"u_z at z=1: {uz[Nx//2, Ny//2, np.argmin(np.abs(z_1d - (1.0)))]}")

# Now let's calculate vorticity analytical
wx_1 = -y * (2*A0_1/R0) * E_1 * ( -2*(z-z1)/a**2 )
