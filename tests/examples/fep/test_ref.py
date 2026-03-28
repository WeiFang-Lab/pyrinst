import pickle

from pyrinst.geometries import HarmRef
from pyrinst.io.xyz import load
from pyrinst.potentials import MACE
from pyrinst.utils.units import Energy

symbols, x, _ = load("C2H2.xyz", energy_pattern=False)
pes = MACE(atoms=symbols, model_paths="MACE-OFF24_medium.model")
reference = HarmRef(x, pes.symbols)
pes.compute(reference, task=2)
reference.calc_freq()
reference.norm_dimensionless_modes()
print("max imaginary freq. :", reference.freqs[0] * Energy(1, "au").get("cm-1"), "cm-1")
with open("a.pkl", "wb") as f:
    pickle.dump(reference, f)
