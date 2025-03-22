from Constants import Ax, Ay, Nx, Ny, Lx, Ly
from Computation import Compute_Lax_Vend_step as Compute
from Render import create_vts_snapshot_vtk

import numpy as np


# Creating mesh
mesh = np.zeros((Nx, Ny, 10), dtype=np.double)

# Initial conditions
# values = ["Vx", "Vy", "Wx", "Wy", "Nx", "Ny", "Nxy", "Mx", "My", "Mxy"]
mesh[99:101, 99:101, 0] = 10
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
dt = 1e-7
print("dt = ", dt, "; Courant dt = ", 1 / (c * np.sqrt(1/dx**2 + 1/dy**2)), "; Courant2 dt = ", 0.5 * 1 / (rho_A / dx + rho_B / dy))


nodes = []
for j in range(Ny):
    for i in range(Nx):
        nodes.append((i * dx - Lx / 2, j * dy - Ly / 2, 0.))

nodes = np.array(nodes)

data  = np.reshape(mesh, (Nx * Ny, mesh.shape[-1]))

steps = 1000
step = 10
j = 0
for i in range(0, steps):
    mesh = Compute(mesh, Ay, Ax, dt, dx, dy)

    if i % step == 0:
        print(j)
        data = np.reshape(mesh, (Nx * Ny, mesh.shape[-1]))
        create_vts_snapshot_vtk(nodes, data[:, 0:2], data[:, 2:4], (Nx, Ny, 1), j, "Lax_Vend")
        j += 1
