import numpy as np

# Constants
E = 210 * 1e9 # Pascal
v = 0.3
p = 7850
h = 0.01
D = E * h**3 / (12 * (1 - v**2))

Lx = 10
Ly = 10
Nx = 200
Ny = 200

l = E * v / ((1 + v) * (1 - 2 * v))
G = E / (2 * (1 + v))

print("G = ", G)
print("l = ", l)

dx = Lx / Nx
I = p * h * dx**4 / 4

Ax = -np.array([
    [0                   , 0                , 0     , 0              , 1 / p, 0, 0    , 0    , 0, 0     ],
    [0                   , 0                , 0     , 0              , 0    , 0, 1 / p, 0    , 0, 0     ],
    [0                   , 0                , 0     , 0              , 0    , 0, 0    , 1 / I, 0, 0     ],
    [0                   , 0                , 0     , 0              , 0    , 0, 0    , 0    , 0, -1 / I],
    [E / (1 - v ** 2)    , 0                , 0     , 0              , 0    , 0, 0    , 0    , 0, 0     ],
    [E * v / (1 - v ** 2), 0                , 0     , 0              , 0    , 0, 0    , 0    , 0, 0     ],
    [0                   , E / (4 * (1 + v)), 0     , 0              , 0    , 0, 0    , 0    , 0, 0     ],
    [0                   , 0                , -D    , 0              , 0    , 0, 0    , 0    , 0, 0     ],
    [0                   , 0                , -D * v, 0              , 0    , 0, 0    , 0    , 0, 0     ],
    [0                   , 0                , 0     , D / 2 * (1 - v), 0    , 0, 0    , 0    , 0, 0     ]
])

Ay = -np.array([
    [0                , 0                   , 0              , 0     , 0, 0    , 1 / p, 0, 0    , 0    ],
    [0                , 0                   , 0              , 0     , 0, 1 / p, 0    , 0, 0    , 0    ],
    [0                , 0                   , 0              , 0     , 0, 0    , 0    , 0, 0    , 1 / I],
    [0                , 0                   , 0              , 0     , 0, 0    , 0    , 0, 1 / I, 0    ],
    [0                , E * v / (1 - v ** 2), 0              , 0     , 0, 0    , 0    , 0, 0    , 0    ],
    [0                , E / (1 - v ** 2)    , 0              , 0     , 0, 0    , 0    , 0, 0    , 0    ],
    [E / (4 * (1 + v)), 0                   , 0              , 0     , 0, 0    , 0    , 0, 0    , 0    ],
    [0                , 0                   , 0              , -D * v, 0, 0    , 0    , 0, 0    , 0    ],
    [0                , 0                   , 0              , -D    , 0, 0    , 0    , 0, 0    , 0    ],
    [0                , 0                   , D / 2 * (1 - v), 0     , 0, 0    , 0    , 0, 0    , 0    ]
])


cs = (E / (2 * (1 + v) * p))**0.5
cp = (E * (1 - v) / ( (1 + v) * (1 - 2 * v) * p))**0.5

print("Cp = ", cp)
print("Cs = ", cs)
print(np.abs(np.linalg.eig(Ay)[0]))
print(np.abs(np.linalg.eig(Ax)[0]))

print("Time = ", 5 / cp)
print("Steps  before Cp touch wall = ", 5 / cp / 2e-6)
print("Frames before Cp touch wall = ", 5 / cp / 2e-6 / 10)
print("Steps  before Cs touch wall = ", 5 / cs / 2e-6)
print("Frames before Cs touch wall = ", 5 / cs / 2e-6 / 10)

# print(np.sum(Ax @ Ay - Ay @ Ax))
