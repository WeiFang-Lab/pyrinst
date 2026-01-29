import numpy as np
import pickle

from pyrinst.config.constants import KB
from pyrinst.utils.fep import *
from pyrinst.utils.units import Energy

h2ev = Energy(1, 'Hartree').get('ev')
# load beads energies and harm energies (preserve original shapes/transpose)
with open('a.pkl', 'rb') as f: input = pickle.load(f)
f = [f"simulation.pos_eval_{i:02d}.xyz" for i in range(24)]
# all(map(os.path.exists, f)) or sys.exit(2)
r = [np.loadtxt(x, usecols=0).ravel() for x in f]
# len(set(map(len, r))) == 1 or sys.exit(3)
bes = np.vstack(r).T
bes = bes[1:] - bes[0]
aes = np.average(bes, axis=1)
# bhs = np.loadtxt('harm_energies.txt').T
bhs = input.harm_energies * h2ev
des = (aes - bhs) * Energy(1, 'eV').get('Hartree') # eV -> Hartree

beta = 1.0 / (250 * KB) # 1/Hartree


freqs_complex = to_complex(input.freq)
df0 = dF(freqs_complex, beta, 24)
df1, var1 = free_energy_perturbation(des, beta, 0)
print('ref:', df0.real * h2ev, 'eV')
print('correction:', df1 * h2ev, 'eV')
print(f"delta F({int(250)}K)=: {df0.real * h2ev + df1 * h2ev:.10e} eV")
print('error: ', var1 * h2ev, 'eV')