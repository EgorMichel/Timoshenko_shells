from Constants import Ax, Ay, Nx, Ny, Lx, Ly
from Computation import Compute_Lax_Frid as Compute
from Render import create_vts_snapshot_vtk

import numpy as np


# Creating mesh
mesh = np.zeros((Nx, Ny, 10), dtype=np.double)

# Initial conditions
# values = ["Vx", "Vy", "Wx", "Wy", "Nx", "Ny", "Nxy", "Mx", "My", "Mxy"]
mesh[99:101, 99:101, 0] = 100
mesh[99:101, 99:101, 2] = 10
# mesh[98:102, 98:102, 1] = 50
dx = Lx / Nx
dy = Ly / Ny

# Time step
eig_val_x, eig_vec_x = np.linalg.eig(Ax)
eig_val_y, eig_vec_y = np.linalg.eig(Ay)
c = np.sqrt(max(np.abs(eig_val_x)) ** 2 + max(np.abs(eig_val_y)) ** 2)
# dt = 1 / (c * np.sqrt(1/dx**2 + 1/dy**2))

rho_A = max(np.abs(np.linalg.eigvals(Ax)))
rho_B = max(np.abs(np.linalg.eigvals(Ay)))
dt = 0.5e-6
print("dt = ", dt, "; Courant dt = ", 1 / (c * np.sqrt(1/dx**2 + 1/dy**2)), "; Courant2 dt = ", 0.5 * 1 / (rho_A / dx + rho_B / dy))


DATA = Compute(mesh, Ay, Ax, dt, dx, dy, steps=2000)

print(DATA.shape)

nodes = []
for j in range(Ny):
    for i in range(Nx):
        nodes.append((i * dx - Lx / 2, j * dy - Ly / 2, 0.))

nodes = np.array(nodes)
data  = np.reshape(DATA, (DATA.shape[0], DATA.shape[1] * DATA.shape[2], DATA.shape[-1]))

step = 40
for i in range(0, DATA.shape[0] // step):
    print(i)
    create_vts_snapshot_vtk(nodes, data[i * step, :, 0:2], data[i * step, :, 2:4], (Nx, Ny, 1), i)
