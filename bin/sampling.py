import argparse
import pickle
import numpy as np

from pyrinst.utils.pimc import HarmFEP, InstantonFEP
from pyrinst.io.xyz import save

def main():
    parser = argparse.ArgumentParser(description='Generate distribution via quasi random number.')
    parser.add_argument('input', help='pkl file.')
    parser.add_argument('-T', type=float, default=300, help='Temperature (K).')
    parser.add_argument('-N', type=int, default=4096, help='The number of configurations sampled.')
    parser.add_argument('-n', '--nbeads', type=int, default=24, help='The number of beads.')
    parser.add_argument('-o', '--output', type=str, default='simulation.pos', help='The prefix of output configuration files.')
    parser.add_argument('-l', '--lmd_val', type=float, default=1.0, help='The mass scaling factor.')
    parser.add_argument('--nprandom', action='store_true', help='Use numpy function to generate gaussian samples')
    parser.add_argument('--inst', help='pkl file of fix-centroid instanton')
    args = parser.parse_args()

    input = np.load(args.input, allow_pickle=True)
    if args.inst:
        inst = np.load(args.inst, allow_pickle=True)
        polymer = InstantonFEP(ref=None, inst=inst)
    else:
        polymer = HarmFEP(input,nbeads = args.n)

    sampled_nm_pos = polymer.sample_normal_modes(temperature=args.T, n_samples=args.N)
    sampled_bead_pos = polymer.get_cart_pos(nm_pos=sampled_nm_pos, temperature=args.T)

    # refresh freqs and add harm_energies
    if args.inst:
        input.harm_energies = polymer.harm_energies
        with open(args.input, 'wb') as f: pickle.dump(input, f)
    else: 
        input.freq = polymer.freqs
        input.harm_energies = polymer.harm_energies
        with open(args.input, 'wb') as f: pickle.dump(input, f)

    for bead_idx in range(args.n):

        filename = f'{args.o}_{str(bead_idx).zfill(len(str(args.n)))}.xyz'
        # Extract positions for this bead across all samples
        bead_positions = sampled_bead_pos[:, bead_idx, :]  # Shape: (N, natoms*3)
        
        # Create list of x
        x_list = [input.x, ]  # write centroid position as the first frame
        if type(polymer) is InstantonFEP:
            x_list[0] = (polymer.npos[bead_idx])
        for sample_idx in range(args.N):
            pos_3d = bead_positions[sample_idx].reshape(-1, 3)  # Shape: (natoms, 3)
            x_list.append(pos_3d) # Shape: (frames, natoms, 3)

        # Write file
        save(filename, x_list, input.atoms, comment=' ')

if __name__ == "__main__":
    main()