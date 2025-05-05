import numpy as np
from numba import njit


# @njit
def Newton_interpolation_vectorized(X, Y, x, order=1):
    """
    Векторизованная интерполяция Ньютона для равномерной сетки с JIT-компиляцией.

    Аргументы:
    X - отсортированный по возрастанию массив координат точек.
    Расстояния между всеми точками равны, то есть сетка регулярная.
    Y - массив значений функции в точках, координаты которых указаны в X.
    Координаты и значения соответствуют друг другу по индексу, т.е значение в точке X[i] равно Y[i].
    x - numpy массив координат, в которых нужно вычислить значения функции,
    используя интерполяционные полиномы Ньютона.
    order - порядок полинома Ньютона.

    Возвращает:
    numpy массив интерполированных значений в точках x.
    """

    n = len(X)
    h = X[1] - X[0]  # Шаг сетки

    # Векторизованный результат
    y_interp = np.zeros_like(x, dtype=np.float64)  # Инициализация массива результатов

    for i in range(x.shape[0]):
        xi = x[i]
        # Находим индекс ближайшего узла слева от xi
        if xi < X[0]:
            index = 0
        elif xi > X[-1]:
            index = n - 1
        else:
            index = int((xi - X[0]) / h)

        # Определяем индексы узлов, которые будут использоваться для интерполяции
        start_index = max(0, index - order + 1)
        end_index = min(n - 1, index + order - (index - start_index))

        # Дополнительная корректировка индексов для достижения нужного порядка
        if end_index - start_index < order:
            if start_index == 0:
                end_index = min(n - 1, order)
            else:
                start_index = max(0, n - 1 - order)
                end_index = n - 1

        selected_indices = np.arange(start_index, end_index + 1)

        # Вычисляем разделенные разности
        divided_differences = np.copy(Y[selected_indices])

        for k in range(1, order + 1):
            for j in range(order, k - 1, -1):
                divided_differences[j] = (divided_differences[j] - divided_differences[j - 1]) / (
                            X[selected_indices[j]] - X[selected_indices[j - k]])

        # Вычисляем интерполяционный полином Ньютона
        yi = divided_differences[0]
        term = 1.0

        for k in range(1, order + 1):
            term *= (xi - X[selected_indices[k - 1]])
            yi += divided_differences[k] * term

        # Limiter
        tmp = Y[selected_indices]
        yi = np.clip(yi, np.min(tmp) if yi < 0 else None, np.max(tmp) if yi > 0 else None)
        # if yi > 0:
        #     m   = np.max(tmp)
        #     if yi > m:
        #         yi = m
        #
        # if yi < 0:
        #     m   = np.min(tmp)
        #     if yi < m:
        #         yi = m


        y_interp[i] = yi  # Сохраняем результат для текущего xi

    return y_interp

@njit
def Newton_my(X, Y, x_all, order=3):

    def one_point(x):
        n = len(X)
        h = X[1] - X[0]

        # Находим индекс ближайшего узла слева от xi
        if x < X[0]:
            index = 0
        elif x > X[-1]:
            index = n - 1
        else:
            index = int((x - X[0]) / h)

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

        return y

    return np.array([one_point(x_) for x_ in x_all])





    # for i in range(len(divided_differences) - 1):
    #     for j in range()


import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import timeit

# Пример данных
F = lambda x_: np.sign(x_)
X = np.linspace(-5, 5, 100)
Y = F(X)
x = np.linspace(-5, 5, 1000)
y = F(x)


# x = np.linspace(1.5, 2.0, 100)
# y = np.array([Newton_my(X, Y, x_, order=3) for x_ in x])
#
# plt.plot(x, y)
# plt.show()
# Интерполяция с использованием вашей функции
y_newton = Newton_my(X, Y, x, order=3)

# y_newton = np.array([Newton_my(X, Y, x_, order=3) for x_ in x])

# Интерполяция с использованием scipy.interpolate.interp1d
f_linear = interp1d(X, Y, kind='linear', fill_value="extrapolate")
y_linear = f_linear(x)

f_cubic = interp1d(X, Y, kind='cubic', fill_value="extrapolate")
y_cubic = f_cubic(x)


# Визуализация результатов
plt.figure(figsize=(10, 6))
plt.plot(X, Y, 'o', label='Исходные точки')
plt.plot(x, y_newton, label='Интерполяция Ньютона (order=3)')
plt.plot(x, y_linear, label='Интерполяция Scipy Linear')
plt.plot(x, y_cubic, label='Интерполяция Scipy Cubic')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Сравнение интерполяций')
plt.legend()
plt.grid(True)


# Визуализация разницы между методами
plt.figure(figsize=(10, 6))
# plt.plot(x, y_newton - y_linear, label='Разница: Ньютон - Linear')
plt.plot(x, y_newton - y, label='Разница: Ньютон')
plt.plot(x, y_cubic - y, label='Разница: Scipy')
plt.xlabel('x')
plt.ylabel('Разница')
plt.title('Разница между интерполяциями Ньютона и Scipy')
plt.legend()
plt.grid(True)

plt.figure(figsize=(10, 6))
plt.plot(x, np.abs(y_cubic - y) - np.abs(y_newton - y), label='Error Scipy - Error Newton')
plt.xlabel('x')
plt.ylabel('Разница')
plt.title('Разница между интерполяциями Ньютона и Scipy')
plt.legend()
plt.grid(True)


plt.show()