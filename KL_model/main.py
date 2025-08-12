from Constants import Ax, Ay, Nx, Ny, Lx, Ly
from Computation import Compute_Lax_Frid_step, Compute_Lax_Vend_step
from Computation import Compute_CXM, Compute_Vz, Compute_q
from Render import create_vts_snapshot_vtk

import numpy as np


# Creating mesh
mesh = np.zeros((Nx, Ny, 10), dtype=np.double)
dx = Lx / Nx
dy = Ly / Ny

# Initial conditions
# values = ["Vx", "Vy", "Wx", "Wy", "Nx", "Ny", "Nxy", "Mx", "My", "Mxy"]


def initial_conditions(mesh):
    """
    Initial conditions setter.
    mesh: numpy array of shape (Nx, Ny, 10).
    values = ["Vx", "Vy", "Wx", "Wy", "Nx", "Ny", "Nxy", "Mx", "My", "Mxy"]
    Returns: numpy array of shape (Nx, Ny, 10) with initial conditions
    """

    # R = 10
    # A = 100
    # t_ = np.linspace(0, 2 * np.pi, 100)

    # points = np.array([[np.round(R * np.cos(t) + 100), np.round(R * np.sin(t) + 100)] for t in t_])
    # velocities = np.array([[A * np.cos(t), A * np.sin(t)] for t in t_])

    # for i in range(len(points)):
    #     mesh[int(points[i, 0]), int(points[i, 1]), 0] = velocities[i, 0]
    #     mesh[int(points[i, 0]), int(points[i, 1]), 1] = velocities[i, 1]


    # Example of initial conditions
    # mesh[99, 100, 2] = 100 # Wx
    # mesh[101, 100, 3] = -100 # Wy
    mesh[99:101, 99:101, 0] = 100 # Vx
    # mesh[100, 99, 0] = 100
    # mesh[100, 101, 1] = -100 # Vy

    return mesh

mesh = initial_conditions(mesh)


# Time step
rho_A = max(np.abs(np.linalg.eigvals(Ax)))
rho_B = max(np.abs(np.linalg.eigvals(Ay)))
dt = 0.5 * 1 / (rho_A / dx + rho_B / dy)
dt = 1e-6
print("dt = ", dt, "; Courant dt = ", 1 / (rho_A / dx + rho_B / dy))


# Generating nodes
nodes = []
for j in range(Ny):
    for i in range(Nx):
        nodes.append((i * dx - Lx / 2, j * dy - Ly / 2, 0.))
nodes = np.array(nodes)


def preliminary_computations(mesh, Ax, Ay, dx, dy):
    """
    Performs preliminary computations for further use in calculations.
    Returns:
        E1, L1, L1_inv: eigenvalues, eigenvectors, and their inverse matrix for Ax
        E2, L2, L2_inv: eigenvalues, eigenvectors, and their inverse matrix for Ay
        x, y, X, Y: coordinate grids
    """
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
    return E1, L1, L1_inv, E2, L2, L2_inv, x, y, X, Y

E1, L1, L1_inv, E2, L2, L2_inv, x, y, X, Y = preliminary_computations(mesh, Ax, Ay, dx, dy)


def run_simulation(mesh, steps = 1000, snapshot_step = 2, dt = 1e-6, method = "Lax-Frid", filename = "test"):
    """
    Runs the simulation for a given number of steps, periodically saving snapshots.
    mesh: numpy array of shape (Nx, Ny, 10)
    steps: number of steps
    snapshot_step: step for saving snapshots
    dt: time step
    method: method of simulation
    Returns: numpy array of shape (Nx, Ny, 10) with final state
    """
    j = 0
    for i in range(0, steps):

        if i % snapshot_step == 0:
            print(j)
            data = np.reshape(mesh, (Nx * Ny, mesh.shape[-1]), order='F')
            Vz = Compute_Vz(data[:, 2], data[:, 3], dx, dy).reshape(-1, 1)
            V = np.concatenate((data[:, 0:2], Vz), axis=1)
            W = data[:, 2:4]
            q = Compute_q(mesh[:, :, 7], mesh[:, :, 8], mesh[:, :, 9], dx, dy).flatten()
            M = data[:, 7:]
            N = data[:, 4:7]
            create_vts_snapshot_vtk(nodes, V, W, q, M, N, (Nx, Ny, 1), j, filename)
            j += 1

        if   method == "Lax-Frid":
            mesh = Compute_Lax_Frid_step(mesh, Ax, Ay, dt, dx, dy, alpha=0.1)
        elif method == "Lax-Vend":
            mesh = Compute_Lax_Vend_step(mesh, Ax, Ay, dt, dx, dy, alpha=0.01)
        elif method == "CXM":
            mesh = Compute_CXM(mesh, E1, L1, L1_inv, E2, L2, L2_inv, dt, x, y, order=3, limiter=True)

    return mesh


from time import time

t0 = time()
mesh = run_simulation(mesh, steps = 100, snapshot_step = 2, dt = 1e-6, method = "CXM", filename = "CXM_Vx")
t1 = time()

print(f"Time taken   : {t1-t0:.{2}f} sec")
