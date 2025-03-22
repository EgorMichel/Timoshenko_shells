import numpy as np
from Render import create_vts_snapshot_vtk

# Constants
E = 200 * 1e9 # Pascal
v = 0.25
p = 7800
h = 0.01
D = E * h**3 / (12 * (1 - v**2))

Lx = 10
Ly = 10
Nx = 100
Ny = 100

x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)

dx = Lx / Nx
dy = Ly / Ny

I = p * h * dx**4 / 4

print(I)

Ax = np.array([
    [0, 0, 0, 0, 1 / p, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1 / p, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1 / I, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, -1 / I],
    [E / (1 - v ** 2), 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [E * v / (1 - v ** 2), 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, E / (4 * (1 + v)), 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, -D, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, -D * v, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, D / 2 * (1 - v), 0, 0, 0, 0, 0, 0]
])

Ay = np.array([
    [0, 0, 0, 0, 0, 0, 1 / p, 0, 0, 0],
    [0, 0, 0, 0, 0, 1 / p, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1 / I],
    [0, 0, 0, 0, 0, 0, 0, 0, 1 / I, 0],
    [0, E * v / (1 - v ** 2), 0, 0, 0, 0, 0, 0, 0, 0],
    [0, E / (1 - v ** 2), 0, 0, 0, 0, 0, 0, 0, 0],
    [E / (4 * (1 + v)), 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -D * v, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -D, 0, 0, 0, 0, 0, 0],
    [0, 0, D / 2 * (1 - v), 0, 0, 0, 0, 0, 0, 0]
])



mesh = np.zeros((Nx, Ny, 10))

eig_val_x, eig_vec_x = np.linalg.eig(Ax)
eig_val_y, eig_vec_y = np.linalg.eig(Ay)
#
c = np.sqrt(max(np.abs(eig_val_x)) ** 2 + max(np.abs(eig_val_y)) ** 2)
dt = 1 / (c * np.sqrt(1/dx**2 + 1/dy**2))

print(dt)

steps = 100

values = ["Vx", "Vy", "Wx", "Wy", "Nx", "Ny", "Nxy", "Mx", "My", "Mxy"]

# mesh[0, 45:55, 3] = 0
mesh[49:51, 49:51, 2] = 1
mesh[49:51, 49:51, 0] = 1
# mesh[0, 45:55, 9] = 0

DATA = np.zeros((steps, Nx, Ny, 10))

for _ in range(steps):
    mesh_new = np.zeros_like(mesh)

    y_part = (mesh[1:-1, 2:] - mesh[1:-1, :-2]) @ Ay
    x_part = (mesh[2:, 1:-1] - mesh[:-2, 1:-1]) @ Ax
    mesh_new[1:-1, 1:-1] = 0.25 * (
            mesh[2:, 1:-1] + mesh[:-2, 1:-1] + mesh[1:-1, 2:] + mesh[1:-1, :-2]
    ) - dt / (2 * dx) * x_part - dt / (2 * dy) * y_part

    mesh_new[0, :]  = mesh_new[1, :]
    mesh_new[-1, :] = mesh_new[-2, :]
    mesh_new[:, 0]  = mesh_new[:, 1]
    mesh_new[:, -1] = mesh_new[:, -2]

    mesh = mesh_new.copy()
    DATA[_] = np.copy(mesh)


from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors


for val in range(len(values)):
    fig, ax = plt.subplots()
    board = max(abs(np.min(DATA[:, :, :, val])), abs(np.max(DATA[:, :, :, val])))
    # img = ax.imshow(DATA[0, :, :, val], cmap='RdBu_r', origin='lower',
    #            extent=[x.min(), x.max(), y.min(), y.max()], vmin=-board, vmax=board)

    linthresh = 0.01
    norm = mcolors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=-board, vmax=board)

    img = ax.imshow(DATA[0, :, :, val], cmap='RdBu_r', origin='lower',
                    extent=[x.min(), x.max(), y.min(), y.max()],
                    norm=norm)

    fig.colorbar(img, label='Ð—Ð½Ð°Ñ‡ÐµÐ½Ð¸Ðµ' + values[val])
    ax.set_title(values[val])


    def update(frame):
        img.set_data(DATA[frame, :, :, val])
        return [img]

    ani = FuncAnimation(
        fig=fig,
        func=update,
        frames=DATA.shape[0],
        interval=50,
        blit=True
    )

    ani.save(str(val) + "_" + values[val] + ".gif", fps=15)