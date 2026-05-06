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
if not hasattr(torch, "compiler"):
    torch.compiler = types.SimpleNamespace()
if not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False

from mace.calculators import mace_polar
from unitcellsampling.sample import UnitCellSampler

# -----------------------------
# User settings
# -----------------------------
STRUCTURE_FILE = "Li10Ge(PS6)2.cif"
FRAMEWORK_FILE = "GePS_framework.cif"

GRID_SHAPE = (44, 44, 64)
CUTOFF_RADII = 1.0
MIDVOX = True

POLAR_MODEL = "polar-1-m"   # official options include polar-1-s, polar-1-m, polar-1-l
DEVICE = "cpu"              # change to "cuda" on a working GPU setup
DTYPE = "float64"

# IMPORTANT:
# This is the total charge/spin of the framework + probe configuration.
# You need to choose this convention consciously for your stripped-framework setup.
TOTAL_CHARGE = 0
TOTAL_SPIN = 0
EXTERNAL_FIELD = [0.0, 0.0, 0.0]

OUT_GRID = "Li10Ge(PS6)2_Li_probe_mace_polar_grid.npy"
OUT_MASK = "Li10Ge(PS6)2_Li_probe_mace_polar_mask.npy"


# 1) load full structure
atoms = read(STRUCTURE_FILE)

# 2) remove mobile ions to make the rigid framework
framework = atoms[[atom.index for atom in atoms if atom.symbol != "Li"]]

# optional: save for inspection
write(FRAMEWORK_FILE, framework)

print("Original structure:", atoms)
print("Framework only:", framework)

# 3) make sampler on framework
sampler = UnitCellSampler(framework)

# 4) generate grid
sampler.generate_grid_vectors(
    n_frac=GRID_SHAPE,
    cutoff_radii=CUTOFF_RADII,
    vdw_scale=None,
    midvox=MIDVOX,
)

# 5) build MACE-POLAR-1 ASE calculator
polar_calc = mace_polar(
    model=POLAR_MODEL,
    device=DEVICE,
    default_dtype=DTYPE,
)

print(f"Using MACE-POLAR-1 model={POLAR_MODEL}, device={DEVICE}, dtype={DTYPE}")
print(f"Total charge={TOTAL_CHARGE}, total spin={TOTAL_SPIN}, external_field={EXTERNAL_FIELD}")

# 6) callable hook for unitcellsampling
def mace_polar_method(atoms_with_probe):
    atoms_with_probe = atoms_with_probe.copy()

    # Required by PolarMACE
    atoms_with_probe.info["charge"] = TOTAL_CHARGE
    atoms_with_probe.info["spin"] = TOTAL_SPIN
    atoms_with_probe.info["external_field"] = EXTERNAL_FIELD

    atoms_with_probe.calc = polar_calc
    return atoms_with_probe.get_potential_energy()

# 7) sample one Li over the empty framework
energies = sampler.calculate_energies(
    method=mace_polar_method,
    atom="Li",
    exploit_symmetry=False,
    normalize=True,
)

energies = np.asarray(energies).reshape(sampler.n_frac)

finite = np.isfinite(energies)
print("Grid shape:", energies.shape)
print("Finite points:", finite.sum(), "/", energies.size)
if finite.any():
    print("Min finite energy:", np.nanmin(energies))
    print("Max finite energy:", np.nanmax(energies))

np.save(OUT_GRID, energies)
np.save(OUT_MASK, sampler.included_grid_vectors)
print(f"Saved grid to {OUT_GRID}")
print(f"Saved mask to {OUT_MASK}")