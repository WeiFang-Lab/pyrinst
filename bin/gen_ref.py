import argparse

from pyrinst.utils.reference import FEPRef
from pyrinst.io.xyz import load
from pyrinst.core.pes.mace import MACEPES

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Centroid structure in xyz format.')
    parser.add_argument('-o', '--output', default='ref', help='filename of reference pkl.')
    parser.add_argument('--phase', choices=('gas', 'liquid', 'solid'), default='gas', help='Phase of the system')
    parser.add_argument('-N', '--beads', type=int, help='Number of  beads.')
    parser.add_argument('-P', '--PES', type=str, choices=('MACE',), help='Potential energy surface')
    # parameters for mace calculator
    parser.add_argument('--model_path', help='path of MACE model path', type=str)
    parser.add_argument('--dtype', help='dtype of MACE model', type=str, default='float64')
    parser.add_argument('--device', help='device which model runs on', type=str, default='cuda')
    parser.add_argument('--enable_cueq', action='store_true', help='Enable CUEQ (default: disabled)')

    args = parser.parse_args()

    x =  load(args.input)
    if args.PES == 'MACE':
        mace_pes = MACEPES(x, model_paths = args.model_path, default_dtype = args.dtype, device = args.device, enable_cueq = args.enable_cueq)

    reference = FEPRef(x, pes = mace_pes, phase = args.phase)
    reference.recalc_hess()
    reference.norm_dimensionless_modes()
    print('max imaginary freq:', reference.freq[0])
    reference.save(args.output)

if __name__ == "__main__":
    main()