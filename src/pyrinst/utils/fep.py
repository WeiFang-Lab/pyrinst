import numpy as np
from scipy import signal


def autocorr(data, maxlag):
    """Calculate autocorrelation function and error estimates"""
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    ndata = data - mean
    var = np.var(data)

    # Use scipy to calculate the autocorrelation
    acorr = signal.correlate(ndata, ndata, "full")[len(ndata) - 1 :]
    acorr = acorr / acorr[0]  # Normalize

    # Calculate autocorrelation time and the corrected error
    if maxlag == 0:
        # Auto-determine maxlag if 0 is provided
        maxlag_indices = np.where(acorr < 0.01)[0]
        maxlag = maxlag_indices[0] if len(maxlag_indices) > 0 else min(len(acorr) // 3, 100)

    tau = 2 * np.sum(acorr[1:maxlag]) + 1
    error = np.sqrt(tau * var / n)

    return mean, tau, error


def free_energy_perturbation(potential_energies, beta, maxlag):
    # Calculate the boltzmann factor
    boltzmann_factors = np.exp(-beta * potential_energies)

    # Calculate autocorrelation and error for the Boltzmann factors
    mean_exp, tau_exp, error_exp = autocorr(boltzmann_factors, maxlag)

    # Convert to free energy
    deltaF = -1 / beta * np.log(mean_exp)

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
    """ """
    real = np.where(freqs > 0, freqs, 0.0)
    imag = np.where(freqs < 0, -freqs, 0.0)
    return real + 1j * imag
