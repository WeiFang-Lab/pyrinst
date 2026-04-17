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


def free_energy_perturbation(potential_energies, beta, maxlag=None, weights=1):
    # Calculate the boltzmann factor
    boltzmann_factors = np.exp(-beta * potential_energies) * weights

    # Calculate autocorrelation and error for the Boltzmann factors
    if maxlag is not None:
        mean_exp, _, error_exp = autocorr(boltzmann_factors, maxlag)
    else:
        mean_exp = np.mean(boltzmann_factors)
        error_exp = np.std(boltzmann_factors) / np.sqrt(len(boltzmann_factors))

    # Convert to free energy
    delta_free_energy = -1 / beta * np.log(mean_exp)

    # Error propagation for the logarithm
    # δ(ΔF) = δ(ln(<exp(-βΔU)>)) / β = δ(<exp(-βΔU)>) / (β * <exp(-βΔU)>)
    error = error_exp / (beta * mean_exp)

    return delta_free_energy, error
