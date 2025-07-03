import numpy as np
from numba import njit


# @njit
def Compute_Lax_Frid_step(data, Ax, Ay, dt, dx, dy, alpha=0.05):
    mesh_new = np.zeros_like(data)

    du_dx = (data[2:, 1:-1] - data[:-2, 1:-1]) / (2 * dx)
    du_dy = (data[1:-1, 2:] - data[1:-1, :-2]) / (2 * dy)

    x_1 = du_dx @ Ax * dt
    y_1 = du_dy @ Ay * dt

    u_ij = ((alpha / 4) * (data[2:, 1:-1] + data[:-2, 1:-1] + data[1:-1, 2:] + data[1:-1, :-2])
            + (1 - alpha) * data[1:-1, 1:-1])

    mesh_new[1:-1, 1:-1] = u_ij - x_1 - y_1

    # Board conditions
    mesh_new[0, :] = mesh_new[1, :]
    mesh_new[-1, :] = mesh_new[-2, :]
    mesh_new[:, 0] = mesh_new[:, 1]
    mesh_new[:, -1] = mesh_new[:, -2]
    return mesh_new



def Compute_Lax_Vend_step(data, Ax, Ay, dt, dx, dy, alpha=0.05):
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

    u_ij = ((alpha / 4) * (data[2:, 1:-1] + data[:-2, 1:-1] + data[1:-1, 2:] + data[1:-1, :-2])
            + (1 - alpha) * data[1:-1, 1:-1])

    mesh_new[1:-1, 1:-1] = u_ij - x_1 - y_1 + x_2 + y_2 + xy_2

    # Board conditions
    mesh_new[0, :]  = mesh_new[1, :]
    mesh_new[-1, :] = mesh_new[-2, :]
    mesh_new[:, 0]  = mesh_new[:, 1]
    mesh_new[:, -1] = mesh_new[:, -2]
    return mesh_new


from scipy.interpolate import RegularGridInterpolator
from random import randint

@njit(fastmath=True, cache=True)
def Newton_my(X, Y, x_all, order=1, limiter=False):

    def one_point(x):
        n = len(X)
        h = X[1] - X[0]

        # Находим индекс ближайшего узла слева от xi
        index = max(0, min(n - 1, int((x - X[0]) / h)))

        start_index = max(0, index - order // 2)
        end_index = min(n - 1, start_index + order)

        indices = np.arange(start_index, end_index + 1)

        divided_differences = np.copy(Y[indices])
        diff_matrix = np.zeros((len(indices), len(indices)))
        diff_matrix[:, 0] = np.copy(Y[indices])


        for i in range(1, len(indices)):
            diff_matrix[0:-i, i] = diff_matrix[1:len(indices) - i + 1, i - 1] - diff_matrix[0:len(indices) - i, i - 1]

        divided_differences = np.copy(diff_matrix[0])

        y = divided_differences[0]
        q = (x - X[indices[0]]) / h

        term = q
        fact = 1.0
        for i in range(1, len(divided_differences)):
            fact *= i
            y += term / fact * divided_differences[i]
            q -= 1
            term *= q

        # Limiter
        if limiter:
            if y > 0:
                y = min(y, np.max(Y[indices]))
            elif y < 0:
                y = max(y, np.min(Y[indices]))


        return y

    return np.array([one_point(x_) for x_ in x_all])


def PerformOnePart_my(data, E, L, L_inv, x, y, dt, dir, order=1, limiter=False):
    nx, ny, n_vars = data.shape
    V = np.einsum('ij,klj->kli', L, data)

    V_interpolated = np.copy(V)
    for i in range(n_vars):
        Values = V[:, :, i]
        shift  = E[i] * dt

        if dir == 'X':
            x_shifted = x - shift
            for j in range(ny):
                Values[j, :] = Newton_my(x, Values[j, :], x_shifted, order, limiter)

        if dir == 'Y':
            y_shifted = y - shift
            for j in range(nx):
                Values[:, j] = Newton_my(y, Values[:, j], y_shifted, order, limiter)

        V_interpolated[:, :, i] = np.copy(Values)

    return np.einsum('ij,klj->kli', L_inv, V_interpolated)


def Compute_CXM(data, E1, L1, L1_inv, E2, L2, L2_inv, dt, x, y, order=1, limiter=False):

    mesh_new = np.zeros_like(data)

    if randint(0, 1):
        U1 = PerformOnePart_my(data, E1, L1, L1_inv, x, y, dt, "X", order, limiter)
        U2 = PerformOnePart_my(U1,   E2, L2, L2_inv, x, y, dt, "Y", order, limiter)
    else:
        U1 = PerformOnePart_my(data, E2, L2, L2_inv, x, y, dt, "Y", order, limiter)
        U2 = PerformOnePart_my(U1,   E1, L1, L1_inv, x, y, dt, "X", order, limiter)

    mesh_new[1:-1, 1:-1] = U2[1:-1, 1:-1]

    mesh_new[0, :] = mesh_new[1, :]
    mesh_new[-1, :] = mesh_new[-2, :]
    mesh_new[:, 0] = mesh_new[:, 1]
    mesh_new[:, -1] = mesh_new[:, -2]

    return mesh_new


def Compute_Vz(Wx, Wy, dx, dy):
    return (Wx * dx + Wy * dy) / 2


def Compute_q(Mx, My, Mxy, dx, dy):
    q = np.zeros_like(Mx)

    d2Mx_dx2   = (Mx[2: , 1:-1] - 2 * Mx[1:-1, 1:-1] + Mx[ :-2, 1:-1]) / dx**2
    d2My_dy2   = (My[1:-1, 2: ] - 2 * My[1:-1, 1:-1] + My[1:-1,  :-2]) / dy**2
    d2Mxy_dxdy = (Mxy[2: , 2: ] - Mxy[2: ,  :-2] - Mxy[ :-2, 2: ] + Mxy[ :-2,  :-2]) / (4*dx*dy)

    q[1:-1, 1:-1] = -(d2Mx_dx2 + d2My_dy2 - 2 * d2Mxy_dxdy)

    q[0, :] = q[1, :]
    q[-1, :] = q[-2, :]
    q[:, 0] = q[:, 1]
    q[:, -1] = q[:, -2]

    return q