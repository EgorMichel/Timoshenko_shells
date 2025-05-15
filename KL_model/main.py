from Constants import Ax, Ay, Nx, Ny, Lx, Ly
from Computation import Compute_Lax_Frid_step, Compute_Lax_Vend_step
from Computation import Compute_GHM, Compute_Vz, Compute_q
from Render import create_vts_snapshot_vtk

import numpy as np


# Creating mesh
mesh = np.zeros((Nx, Ny, 10), dtype=np.double)
dx = Lx / Nx
dy = Ly / Ny

# Initial conditions
# values = ["Vx", "Vy", "Wx", "Wy", "Nx", "Ny", "Nxy", "Mx", "My", "Mxy"]
mesh[99:101, 99:101, 0] = 100
mesh[99:101, 99:101, 2] = 100
# mesh[99, 100, 2] = 100
# mesh[101, 100, 2] = -100
# mesh[100, 99, 3] = 100
# mesh[100, 101, 3] = -100
#  -0.05 - 0.05 100

# Time step
rho_A = max(np.abs(np.linalg.eigvals(Ax)))
rho_B = max(np.abs(np.linalg.eigvals(Ay)))
dt = 0.5 * 1 / (rho_A / dx + rho_B / dy)
dt = 1e-6
print("dt = ", dt, "; Courant dt = ", 1 / (rho_A / dx + rho_B / dy))


nodes = []
for j in range(Ny):
    for i in range(Nx):
        nodes.append((i * dx - Lx / 2, j * dy - Ly / 2, 0.))
nodes = np.array(nodes)

steps = 300
step  = 2
j = 0

# Preliminary computations
E1, L1 = np.linalg.eig(Ax.T)
E2, L2 = np.linalg.eig(Ay.T)
L1 = L1.T
L2 = L2.T
L1_inv = np.linalg.inv(L1)
L2_inv = np.linalg.inv(L2)
nx, ny, n_vars = mesh.shape
x = np.arange(0, nx * dx, dx)
y = np.arange(0, ny * dy, dy)
X, Y = np.meshgrid(x[1:-1], y[1:-1], indexing='ij')

from time import time

t0 = time()

for i in range(0, steps):

    if i % step == 0:
        print(j)
        data = np.reshape(mesh, (Nx * Ny, mesh.shape[-1]))
        Vz = Compute_Vz(data[:, 2], data[:, 3], dx, dy).reshape(-1, 1)
        V = np.concatenate((data[:, 0:2], Vz), axis=1)
        W = data[:, 2:4]
        q = Compute_q(mesh[:, :, 7], mesh[:, :, 8], mesh[:, :, 9], dx, dy).flatten()
        M = data[:, 7:]
        N = data[:, 4:7]
        create_vts_snapshot_vtk(nodes, V, W, q, M, N, (Nx, Ny, 1), j, "CXM_3_lim")
        j += 1


    # mesh = Compute_Lax_Vend_step(mesh, Ay, Ax, dt, dx, dy, alpha=0.05)
    mesh = Compute_GHM(mesh, E1, L1, L1_inv, E2, L2, L2_inv, dt, x, y, order=3, limiter=True)

t1 = time()

print(f"Time taken   : {t1-t0:.{2}f} sec")
print(f"Time per step: {(t1-t0) / steps:.{2}f} sec")
