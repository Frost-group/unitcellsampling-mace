import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from ase.io import read, write
from ase.calculators.lammpslib import LAMMPSlib
import numpy as np

from unitcellsampling.sample import UnitCellSampler


# 1) load full structure
atoms = read("LGPS-tetra.cif")

# 2) remove mobile Li to make rigid framework
framework = atoms[[atom.index for atom in atoms if atom.symbol != "Li"]]
write("GePS-tetra.cif", framework)

print("Original structure:", atoms)
print("Framework only:", framework)

# 3) sampler on framework
sampler = UnitCellSampler(framework)

sampler.generate_grid_vectors(
    n_frac=(44, 44, 64),
    cutoff_radii=1.0,
    vdw_scale=None,
    midvox=True,
)

# 4) LGPS-specific UFF-like / LAMMPS callable
#
# IMPORTANT:
# - You must fill in physically sensible charges and LJ params for Li/Ge/P/S.
# - LAMMPS "real" units expect:
#     epsilon in kcal/mol
#     sigma in Å
#     charges in e
#
# This is a TEMPLATE, not a validated force field.

ATOM_TYPES = {
    "Li": 1,
    "Ge": 2,
    "P": 3,
    "S": 4,
}

ATOM_MASSES = {
    "Li": 6.941,
    "Ge": 72.630,
    "P": 30.973761998,
    "S": 32.06,
}

# ---- FILL THESE IN ----
CHARGES = {
    "Li": +1.0,   # placeholder
    "Ge":  0.0,   # placeholder
    "P":   0.0,   # placeholder
    "S":   0.0,   # placeholder
}

# diagonal LJ terms only; pair_modify mix arithmetic handles cross terms
# values below are placeholders and must be replaced
LJ = {
    "Li": (0.0440, 2.571134),  # epsilon, sigma
    "Ge": (0.1000, 3.700000),
    "P":  (0.1000, 3.400000),
    "S":  (0.2500, 3.600000),
}


def make_lammps_calc():
    lmpcmds = [
        "pair_style lj/cut/coul/long 12.500",
        "pair_modify tail yes mix arithmetic",
        "special_bonds lj/coul 0.0 0.0 1.0",
        "dielectric 1.0",
        "kspace_style ewald 1.0e-5",
    ]

    # pair_coeff type type epsilon sigma
    for sym, typ in ATOM_TYPES.items():
        eps, sig = LJ[sym]
        lmpcmds.append(f"pair_coeff {typ} {typ} {eps:.8f} {sig:.8f}")

    amendments = []
    for sym, typ in ATOM_TYPES.items():
        amendments.append(f"set type {typ} charge {CHARGES[sym]}")

    calc = LAMMPSlib(
        lmpcmds=lmpcmds,
        atom_types=ATOM_TYPES,
        atom_type_masses=ATOM_MASSES,
        log_file="lgps_uff_probe.log",
        lammps_header=[
            "units real",
            "atom_style full",
            "boundary p p p",
        ],
        amendments=amendments,
        post_changebox_cmds=None,
    )
    return calc


def lgps_uff_method(atoms_with_probe):
    atoms_with_probe = atoms_with_probe.copy()
    atoms_with_probe.calc = make_lammps_calc()
    return atoms_with_probe.get_potential_energy()


# 5) sample one Li probe over rigid framework
energies = sampler.calculate_energies(
    method=lgps_uff_method,
    atom="Li",
    exploit_symmetry=False,
    normalize=True,
)

energies = np.asarray(energies).reshape(sampler.n_frac)

print("Grid shape:", energies.shape)
print("Min energy:", np.nanmin(energies))
print("Max energy:", np.nanmax(energies))

np.save("LGPS-tetra_probe_uff_grid.npy", energies)
np.save("LGPS-tetra_probe_uff_mask.npy", sampler.included_grid_vectors)

print("Saved grid to LGPS-tetra_probe_uff_grid.npy")
print("Saved mask to LGPS-tetra_probe_uff_mask.npy")