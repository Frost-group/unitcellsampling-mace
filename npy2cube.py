import numpy as np
from ase.io import read
from ase.io.cube import write_cube

# framework structure with the correct cell
atoms = read("TaCl6_framework.cif")   # or whatever file you are using

grid = np.load("NaTaCl6_Na_probe_mace_grid.npy").astype(float)

# cube files should not contain NaNs, so replace excluded points
fill_value = np.nanmax(grid) + 5.0
grid[np.isnan(grid)] = fill_value

with open("NaTaCl6_Na_probe_mace_grid.cube", "w") as f:
    write_cube(f, atoms, data=grid)