import vtk
import numpy as np


def create_vts_snapshot_vtk(nodes, values_vel, values_ang, dims, snapshot_number):
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

    # velocities = vtk.vtkDoubleArray()
    # velocities.SetName("velocity")
    # velocities.SetNumberOfComponents(3)
    # values_vel = np.asarray(values_vel, dtype=np.float64)
    # for vx, vy in values_vel:
    #     velocities.InsertNextTuple3(vx, vy, 0.)
    # grid.GetPointData().SetVectors(velocities)
    #
    # ang_vels = vtk.vtkDoubleArray()
    # ang_vels.SetName("ang_vel")
    # ang_vels.SetNumberOfComponents(3)
    # values_ang = np.asarray(values_ang, dtype=np.float64)
    # for wx, wy in values_ang:
    #     ang_vels.InsertNextTuple3(wx, wy, 0.)
    # grid.GetPointData().SetVectors(ang_vels)

    # Add velocity array
    velocities = vtk.vtkDoubleArray()
    velocities.SetName("velocity")
    velocities.SetNumberOfComponents(3)
    values_vel = np.asarray(values_vel, dtype=np.float64)
    for vx, vy in values_vel:
        velocities.InsertNextTuple3(vx, vy, 0.)
    grid.GetPointData().AddArray(velocities)  # Changed to AddArray

    # Add angular velocity array
    ang_vels = vtk.vtkDoubleArray()
    ang_vels.SetName("ang_vel")
    ang_vels.SetNumberOfComponents(3)
    values_ang = np.asarray(values_ang, dtype=np.float64)
    for wx, wy in values_ang:
        ang_vels.InsertNextTuple3(wx, wy, 0.)
    grid.GetPointData().AddArray(ang_vels)  # Changed to AddArray

    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(f"Results/Lax_Frid3_{snapshot_number}.vts")
    writer.SetInputData(grid)
    writer.SetDataModeToAscii()
    writer.Write()
