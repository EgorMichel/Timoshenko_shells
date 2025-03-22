import numpy as np

def Compute_Lax_Frid_step(data, Ax, Ay, dt, dx, dy):
    mesh_new = np.zeros_like(data)

    x_part = (data[2:, 1:-1] - data[:-2, 1:-1]) @ Ax
    y_part = (data[1:-1, 2:] - data[1:-1, :-2]) @ Ay

    alpha = 0.1

    mesh_new[1:-1, 1:-1] = (alpha / 4) * (
            data[2:, 1:-1] + data[:-2, 1:-1] + data[1:-1, 2:] + data[1:-1, :-2]
    ) + (1 - alpha) * data[1:-1, 1:-1] - dt / (2 * dx) * x_part - dt / (2 * dy) * y_part

    # Board conditions
    mesh_new[0, :] = mesh_new[1, :]
    mesh_new[-1, :] = mesh_new[-2, :]
    mesh_new[:, 0] = mesh_new[:, 1]
    mesh_new[:, -1] = mesh_new[:, -2]
    return mesh_new

def Compute_Lax_Vend_step(data, Ax, Ay, dt, dx, dy, steps=100):
    mesh_new = np.zeros_like(data)


    du_dx = (data[2:, 1:-1] - data[:-2, 1:-1]) / (2 * dx)
    du_dy = (data[1:-1, 2:] - data[1:-1, :-2]) / (2 * dy)

    d2u_dx = (data[2:, 1:-1] + data[:-2, 1:-1] - 2 * data[1:-1, 1:-1]) / (dx ** 2)
    d2u_dy = (data[1:-1, 2:] + data[1:-1, :-2] - 2 * data[1:-1, 1:-1]) / (dy ** 2)

    d2u_dxdy = (data[2:, 2:] - data[2:, :-2] - data[:-2, 2:] + data[:-2, :-2]) / (4 * dx * dy)

    x_1 = du_dx @ Ax * dt
    y_1 = du_dy @ Ay * dt

    x_2 = d2u_dx @ (Ax @ Ax) / 2 * dt ** 2
    y_2 = d2u_dy @ (Ay @ Ay) / 2 * dt ** 2

    xy_2 = d2u_dxdy @ (Ax @ Ay + Ay @ Ax) * dt**2

    alpha = 0.05

    u_ij = ((alpha / 4) * (data[2:, 1:-1] + data[:-2, 1:-1] + data[1:-1, 2:] + data[1:-1, :-2])
            + (1 - alpha) * data[1:-1, 1:-1])

    mesh_new[1:-1, 1:-1] = u_ij - x_1 - y_1 + x_2 + y_2 + xy_2

    # Board conditions
    mesh_new[0, :]  = mesh_new[1, :]
    mesh_new[-1, :] = mesh_new[-2, :]
    mesh_new[:, 0]  = mesh_new[:, 1]
    mesh_new[:, -1] = mesh_new[:, -2]
    return mesh_new
