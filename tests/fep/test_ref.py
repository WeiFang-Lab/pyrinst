import pickle

from pyrinst.core.pes.mace import MACEPES
from pyrinst.io.xyz import load
from pyrinst.utils.reference import FEPRef
from pyrinst.utils.units import Energy

pes = MACEPES(atoms="C2H2.xyz", model_paths="MACE-OFF24_medium.model")
x = load("C2H2.xyz")
reference = FEPRef(x, pes=pes)
reference.recalc_hess()
reference.norm_dimensionless_modes()
print("max imaginary freq. :", reference.freqs[0] * Energy(1, "au").get("cm-1"), "cm-1")
with open("a.pkl", "wb") as f:
    pickle.dump(reference, f)
