import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from ase.io import read, write
import numpy as np
import types

# ---- temporary torch compatibility workaround ----
# Your torch 2.2.2 on this machine has torch.compiler but not is_compiling
if not hasattr(torch, "compiler"):
    torch.compiler = types.SimpleNamespace()
if not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False

from mace.calculators import mace_mp
from unitcellsampling.sample import UnitCellSampler

# 1) load full structure
atoms = read("NaTaCl6.cif")

# 2) remove mobile ions to make the rigid framework
framework = atoms[[atom.index for atom in atoms if atom.symbol != "Na"]]

# optional: save for inspection
write("TaCl6_framework.cif", framework)

print("Original structure:", atoms)
print("Framework only:", framework)

# 3) make sampler on framework
sampler = UnitCellSampler(framework)

# 4) generate a tiny test grid first
sampler.generate_grid_vectors(
    n_frac=(20, 20, 20),
    cutoff_radii=1.0,
    vdw_scale=None,
    midvox=True,
)

# 5) build MACE ASE calculator
mace_calc = mace_mp(
    device="cpu",
    default_dtype="float64",
)

# 6) callable hook for unitcellsampling
def mace_method(atoms_with_probe):
    atoms_with_probe = atoms_with_probe.copy()
    atoms_with_probe.calc = mace_calc
    return atoms_with_probe.get_potential_energy()

# 7) sample one Na over the empty framework
energies = sampler.calculate_energies(
    method=mace_method,
    atom="Na",
    exploit_symmetry=False,
    normalize=True,
)

# current unitcellsampling code may return flat output
energies = np.asarray(energies).reshape(sampler.n_frac)

print("Grid shape:", energies.shape)
print("Min energy:", energies.min())
print("Max energy:", energies.max())

np.save("NaTaCl6_Na_probe_mace_grid.npy", energies)
np.save("NaTaCl6_Na_probe_mask.npy", sampler.included_grid_vectors)
print("Saved grid to NaTaCl6_Na_probe_mace_grid.npy")