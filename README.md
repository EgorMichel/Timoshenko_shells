# Timoshenko_shells
Solver for Timoshenko shell model.

Install dependencies:

    pip install -r requirements.txt

Run:

    python3 main.py


Code:

Initial conditions:

Initials sets in function initial_conditions at main.py.
There is an example for circle conditions and commented examples.
Variables vector: ["Vx", "Vy", "Wx", "Wy", "Nx", "Ny", "Nxy", "Mx", "My", "Mxy"]. So for example:

    mesh[x_coord, y_coord, n] = N

x_coord - pixel at x direction (from 0 to Nx)

y_coord - pixel at y direction (from 0 to Ny)

n - index in variable vector (for example 3 is Wy)

N - value.

Computation run:

Computation runs in function run_simulation at main.py.
Just need to specify steps, snapshot step, dt, method and filename.

CXM is 3-order with limiter by default. You can edit it in code, for example:

    order=5, limiter=False

Limiter is recommended.

View the results via ParaView. Snapshots are in Results foulder.

