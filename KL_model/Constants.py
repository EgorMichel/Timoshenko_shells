import numpy as np

# Constants
E = 210 * 1e9 # Pa # Young's modulus
v = 0.3 # Poisson's ratio
p = 7800 # kg/m^3 # density
h = 1 # m   # thickness
D = E * h**3 / (12 * (1 - v**2)) # bending stiffness

Lx = 10 # m
Ly = 10 # m
Nx = 200 # number of nodes in x direction
Ny = 200 # number of nodes in y direction

l = E * v / ((1 + v) * (1 - 2 * v)) # shear modulus
G = E / (2 * (1 + v)) # shear modulus


dx = Lx / Nx # grid spacing in x direction
I = p * h**3 / 12 # moment of inertia


Ax = -np.array([
    [0                   , 0                , 0     , 0              , 1 / p, 0, 0    , 0    , 0, 0     ],
    [0                   , 0                , 0     , 0              , 0    , 0, 1 / p, 0    , 0, 0     ],
    [0                   , 0                , 0     , 0              , 0    , 0, 0    , 1 / I, 0, 0     ],
    [0                   , 0                , 0     , 0              , 0    , 0, 0    , 0    , 0, 1 / I ],
    [E / (1 - v ** 2)    , 0                , 0     , 0              , 0    , 0, 0    , 0    , 0, 0     ],
    [E * v / (1 - v ** 2), 0                , 0     , 0              , 0    , 0, 0    , 0    , 0, 0     ],
    [0                   , E / (2 * (1 + v)), 0     , 0              , 0    , 0, 0    , 0    , 0, 0     ],
    [0                   , 0                , D     , 0              , 0    , 0, 0    , 0    , 0, 0     ],
    [0                   , 0                , D * v , 0              , 0    , 0, 0    , 0    , 0, 0     ],
    [0                   , 0                , 0     , D * (1 - v) / 2, 0    , 0, 0    , 0    , 0, 0     ]
])

Ay = -np.array([
    [0                , 0                   , 0              , 0     , 0, 0    , 1 / p, 0, 0    , 0    ],
    [0                , 0                   , 0              , 0     , 0, 1 / p, 0    , 0, 0    , 0    ],
    [0                , 0                   , 0              , 0     , 0, 0    , 0    , 0, 0    , 1 / I],
    [0                , 0                   , 0              , 0     , 0, 0    , 0    , 0, 1 / I, 0    ],
    [0                , E * v / (1 - v ** 2), 0              , 0     , 0, 0    , 0    , 0, 0    , 0    ],
    [0                , E / (1 - v ** 2)    , 0              , 0     , 0, 0    , 0    , 0, 0    , 0    ],
    [E / (2 * (1 + v)), 0                   , 0              , 0     , 0, 0    , 0    , 0, 0    , 0    ],
    [0                , 0                   , 0              , D * v , 0, 0    , 0    , 0, 0    , 0    ],
    [0                , 0                   , 0              , D     , 0, 0    , 0    , 0, 0    , 0    ],
    [0                , 0                   , D * (1 - v) / 2, 0     , 0, 0    , 0    , 0, 0    , 0    ]
])


cs = (E / (2 * (1 + v) * p))**0.5 # shear wave speed
cp = (E * (1 - v) / ( (1 + v) * (1 - 2 * v) * p))**0.5 # compressional wave speed
