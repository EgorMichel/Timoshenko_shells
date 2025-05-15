import vtk
import numpy as np


def create_vts_snapshot_vtk(nodes, values_vel, values_ang, Q, Moments, Stresses, dims, snapshot_number, name):
    expected_points = dims[0] * dims[1] * dims[2]
    if len(nodes) != expected_points:
        raise ValueError(f"Узлы: получено {len(nodes)}, ожидается {expected_points}")
    if len(values_vel) != expected_points:
        raise ValueError(f"Значения: получено {len(values_vel)}, ожидается {expected_points}")

    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(dims)

    vtk_points = vtk.vtkPoints()
    vtk_points.SetDataTypeToDouble()

    for pt in nodes:
        vtk_points.InsertNextPoint(pt)
    grid.SetPoints(vtk_points)

    # Add velocity array
    velocities = vtk.vtkDoubleArray()
    velocities.SetName("velocity")
    velocities.SetNumberOfComponents(3)
    values_vel = np.asarray(values_vel, dtype=np.float64)
    for vx, vy, vz in values_vel:
        velocities.InsertNextTuple3(vx, vy, vz)
    grid.GetPointData().AddArray(velocities)

    # Add angular velocity array
    ang_vels = vtk.vtkDoubleArray()
    ang_vels.SetName("ang_vel")
    ang_vels.SetNumberOfComponents(3)
    values_ang = np.asarray(values_ang, dtype=np.float64)
    for wx, wy in values_ang:
        ang_vels.InsertNextTuple3(wx, wy, 0.)
    grid.GetPointData().AddArray(ang_vels)

    # Add angular moments array
    moments = vtk.vtkDoubleArray()
    moments.SetName("Moments")
    moments.SetNumberOfComponents(3)
    Moments = np.asarray(Moments, dtype=np.float64)
    for Mx, My, Mxy in Moments:
        moments.InsertNextTuple3(Mx, My, Mxy)
    grid.GetPointData().AddArray(moments)

    # Add angular stresses array
    stresses = vtk.vtkDoubleArray()
    stresses.SetName("Stresses")
    stresses.SetNumberOfComponents(3)
    Stresses = np.asarray(Stresses, dtype=np.float64)
    for Sx, Sy, Sxy in Stresses:
        stresses.InsertNextTuple3(Sx, Sy, Sxy)
    grid.GetPointData().AddArray(stresses)

    # Add Q array
    q_array = vtk.vtkDoubleArray()
    q_array.SetName("Q")
    q_array.SetNumberOfComponents(1)  # Q is a scalar field
    Q = np.asarray(Q, dtype=np.float64)
    for q in Q:
        q_array.InsertNextValue(q)  # Insert each scalar value
    grid.GetPointData().AddArray(q_array)

    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(f"Results/" + name + f"_{snapshot_number}.vts")
    writer.SetInputData(grid)
    writer.SetDataModeToAscii()
    writer.Write()
