from ase.io import read, write
from ase import Atom
import torch
import types

# ---- temporary workaround for your torch/compiler mismatch ----
if not hasattr(torch, "compiler"):
    torch.compiler = types.SimpleNamespace()
if not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False

# Optional: reduce threading weirdness on older laptops
torch.set_num_threads(1)

from mace.calculators import mace_mp


def main():
    # 1) Read the full structure
    atoms = read("NaTaCl6.cif")
    print("Loaded full structure:", atoms)

    # 2) Remove all mobile Na ions to create the rigid framework
    framework = atoms[[atom.index for atom in atoms if atom.symbol != "Na"]]
    print("Framework only:", framework)

    # Optional: save it so you can inspect it
    write("TaCl6_framework.cif", framework)

    # 3) Build MACE calculator
    calc = mace_mp(
        device="cpu",
        default_dtype="float64",
        model="medium",   # explicit, avoids ambiguity
    )

    # 4) Test A: framework energy only
    fw = framework.copy()
    fw.calc = calc
    e_framework = fw.get_potential_energy()
    print("\nFramework energy:", e_framework)

    # 5) Test B: add ONE probe Na at a single fractional position
    # Pick something simple and far from exact corners
    frac = [0.125, 0.125, 0.125]
    cart = frac[0] * framework.cell[0] + frac[1] * framework.cell[1] + frac[2] * framework.cell[2]

    probe_structure = framework.copy()
    probe_structure += Atom("Na", position=cart)
    probe_structure.calc = calc

    e_probe = probe_structure.get_potential_energy()
    print("Probe fractional coord:", frac)
    print("Probe Cartesian coord:", cart)
    print("Framework + one Na energy:", e_probe)
    print("Energy difference (probe - framework):", e_probe - e_framework)


if __name__ == "__main__":
    main()