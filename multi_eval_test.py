from ase.io import read
from ase import Atom
import numpy as np
import torch
import types

if not hasattr(torch, "compiler"):
    torch.compiler = types.SimpleNamespace()
if not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False

torch.set_num_threads(1)

from mace.calculators import mace_mp

atoms = read("NaTaCl6.cif")
framework = atoms[[atom.index for atom in atoms if atom.symbol != "Na"]]

calc = mace_mp(device="cpu", default_dtype="float64", model="medium")

fw = framework.copy()
fw.calc = calc
e_framework = fw.get_potential_energy()
print("Framework energy:", e_framework)

test_fracs = [
    [0.125, 0.125, 0.125],
    [0.125, 0.125, 0.375],
    [0.375, 0.375, 0.375],
    [0.875, 0.875, 0.875],
]

for i, frac in enumerate(test_fracs):
    cart = frac[0] * framework.cell[0] + frac[1] * framework.cell[1] + frac[2] * framework.cell[2]
    s = framework.copy()
    s += Atom("Na", position=cart)
    s.calc = calc
    e = s.get_potential_energy()
    print(i, frac, e - e_framework)