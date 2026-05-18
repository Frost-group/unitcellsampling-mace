import numpy as np
import plotly.graph_objects as go

grid = np.load("LGPS-tetra_Li_probe_mace_polar_grid.npy").astype(float)
fill_value = np.nanmax(grid) + 5.0
grid[np.isnan(grid)] = fill_value

nx, ny, nz = grid.shape
x, y, z = np.mgrid[0:nx, 0:ny, 0:nz]

fig = go.Figure(data=go.Isosurface(
    x=x.flatten(),
    y=y.flatten(),
    z=z.flatten(),
    value=grid.flatten(),
    isomin=0.0,
    isomax=10.0,
    surface_count=5,
    caps=dict(x_show=False, y_show=False, z_show=False),
))
fig.update_layout(title="Li probe PES isosurfaces")
fig.show()