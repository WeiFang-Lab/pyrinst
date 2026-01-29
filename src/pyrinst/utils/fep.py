import re
import numpy as np
from scipy import signal

from pyrinst.io.xyz import load


def autocorr(data, maxlag):
    """Calculate autocorrelation function and error estimates"""
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    ndata = data - mean
    var = np.var(data)

    # Use scipy to calculate the autocorrelation
    acorr = signal.correlate(ndata, ndata, 'full')[len(ndata)-1:]
    acorr = acorr / acorr[0]  # Normalize

    # Calculate autocorrelation time and the corrected error
    if maxlag == 0:
        # Auto-determine maxlag if 0 is provided
        maxlag_indices = np.where(acorr < 0.01)[0]
        maxlag = maxlag_indices[0] if len(maxlag_indices) > 0 else min(len(acorr) // 3, 100)

    tau = 2*np.sum(acorr[1:maxlag]) + 1
    error = np.sqrt(tau * var / n)

    return mean, tau, error

def free_energy_perturbation(potential_energies, beta, maxlag):
    # Calculate the boltzmann factor
    boltzmann_factors = np.exp(-beta*potential_energies)

    # Calculate autocorrelation and error for the Boltzmann factors
    mean_exp, tau_exp, error_exp = autocorr(boltzmann_factors, maxlag)

    # Convert to free energy
    deltaF = -1/beta * np.log(mean_exp)

    # Error propagation for the logarithm
    # δ(ΔF) = δ(ln(<exp(-βΔU)>)) / β = δ(<exp(-βΔU)>) / (β * <exp(-βΔU)>)
    deltaF_error = error_exp / (beta * mean_exp)

    return deltaF, deltaF_error

def ana(x):
    em2x = np.exp(-2 * x)
    return x + np.log(1 - em2x) - np.log(2 * x)

def dF0(ws, beta, N=24):
    hbfs = beta * ws / 2
    temp = np.arcsinh(hbfs / N) * N
    return np.sum(ana(temp)) / beta

def dF(ws, beta, N=24):
    hbfs = beta * ws / 2
    temp = np.arcsinh(hbfs / N) * N
    return np.sum(np.log(np.sinh(temp)) - np.log(hbfs)) / beta

def to_complex(freqs):
    '''
    
    '''
    real = np.where(freqs > 0, freqs, 0.0)
    imag = np.where(freqs < 0, -freqs, 0.0)
    return real + 1j * imag


def extract_energy_from_xyz(prefix: str, nbeads: int, is_instanton: bool):
    '''

    '''
    energies = []
    if not is_instanton:
        for bead_idx in range(nbeads):
            filename = f'{prefix}_{str(bead_idx).zfill(len(str(nbeads)))}.xyz'
            _, _, raw_comment = load(filename, return_all=True)

            if isinstance(raw_comment, str):
                comments = [raw_comment]
            else:
                comments = raw_comment

            pattern = r"energy\s*=\s*['\"]?([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)['\"]?"

            for c in comments:
                match = re.search(pattern, c)
                if match:
                    energies.append(float(match.group(1)))
                else:
                    raise ValueError(f"Energy not found in comment: {c}")

        energies = np.array(energies)  # Shape: (nbeads * nframes,)
        nframes = len(energies) // nbeads
        energies = energies.reshape(nbeads, nframes) # Shape: (nbeads , nframes)

    else:
        # Inst. FEP
        pass
    return energies