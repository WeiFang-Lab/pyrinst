import pickle

from pyrinst.utils.pimc import HarmFEP, InstantonFEP
from pyrinst.io.xyz import save

with open('a.pkl', 'rb') as f: input = pickle.load(f)
polymer = HarmFEP(input)
sampled_nm_pos = polymer.sample_normal_modes(temperature=250, n_samples=8192)
sampled_bead_pos = polymer.get_cart_pos(sampled_nm_pos, 250)

input.freq = polymer.freqs
input.harm_energies = polymer.harm_energies
with open('a.pkl', 'wb') as f: pickle.dump(input, f)

for bead_idx in range(24):
    filename = f'simulation.pos_{str(bead_idx).zfill(len(str(24)))}.xyz'
    
    # Extract positions for this bead across all samples
    bead_positions = sampled_bead_pos[:, bead_idx, :]  # Shape: (N, natoms*3)
    
    # Create list of x
    x_list = [input.x, ]  # write centroid position as the first frame
    for sample_idx in range(8192):
        pos_3d = bead_positions[sample_idx].reshape(-1, 3)  # Shape: (natoms, 3)
        x_list.append(pos_3d) # Shape: (frames, natoms, 3)

    save(filename, x_list, input.atoms, comment=' ')