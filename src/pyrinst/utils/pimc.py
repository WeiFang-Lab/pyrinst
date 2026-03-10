import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc

from pyrinst.opt import proj_eig
from pyrinst.utils.coordinates import mass_weight
from pyrinst.utils.units import AMU, HBAR, KB, Energy


def box_muller(U1, U2):
    """
    Box-Muller transformation,
    a method to convert two independent uniform random variables (U1, U2)
    into two independent standard normal random variables.
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-15
    U1 = np.clip(U1, epsilon, 1.0 - epsilon)
    R = np.sqrt(-2 * np.log(U1))
    T = 2 * np.pi * U2
    return R * np.cos(T), R * np.sin(T)


def sobol_gaussian_sample(loc, scale, M0):
    """
    Generate Gaussian samples using Sobol sequences with Box-Muller transform.

    Parameters:
        loc: Mean values (can be multidimensional array)
        scale: Standard deviations (same shape as loc)
        M0: Number of samples (must be a power of 2 for Sobol sequences)

    Returns:
        Samples with shape (M0, *loc.shape)
    """
    # Flatten the arrays for sampling
    loc_flat = np.array(loc).flatten()
    scale_flat = np.array(scale).flatten()

    d = len(loc_flat)
    M = int(np.log2(M0))

    # Generate Sobol sequences
    sampler = qmc.Sobol(d=2 * d, scramble=True, seed=18)
    samp = sampler.random_base2(M)

    # Apply Box-Muller transform
    Z1, Z2 = box_muller(samp[:, :d], samp[:, d:])

    # Use Z1 for the samples (Z2 could be used for additional samples if needed)
    samples = Z1 * scale_flat + loc_flat

    # Reshape back to original shape
    original_shape = np.array(loc).shape
    return samples.reshape((2**M, *original_shape))


def mk_o_nm_matrix(nbeads):
    """
    Makes a matrix that transforms between the bead and the (open path) normal mode
    representations.
    """
    # here define the orthogonal transformation matrix for the open path
    b2o_nm = np.zeros((nbeads, nbeads))
    b2o_nm[0, :] = np.sqrt(1.0)
    for j in range(0, nbeads):
        for i in range(1, nbeads):
            b2o_nm[i, j] = np.sqrt(2.0) * np.cos(np.pi * (j + 0.5) * i / float(nbeads))
    return b2o_nm / np.sqrt(nbeads)


class nm_fft:  # ! TODO add (matrix-version) of the open path transformation here
    """Uses Fast Fourier transforms to do normal mode transformations.

    Attributes:
       fft: The fast-Fourier transform function to transform between the
          bead and normal mode representations.
       ifft: The inverse fast-Fourier transform function to transform
          between the normal mode and bead representations.
       qdummy: A matrix to hold a copy of the bead positions to transform
          them to the normal mode representation.
       qnmdummy: A matrix to hold a copy of the normal modes to transform
          them to the bead representation.
       nbeads: The number of beads.
       natoms: The number of atoms.
    """

    def __init__(self, nbeads, natoms, open_paths=None, n_threads=1, single_precision=False):
        """Initializes nm_trans.

        Args:
           nbeads: The number of beads.
           natoms: The number of atoms.
        """

        self.nbeads = nbeads
        self.natoms = natoms
        self.n_threads = n_threads
        self.single_precision = single_precision
        if open_paths is None:
            open_paths = []
        self._open = open_paths
        # for atoms with open path we still use the matrix transformation
        self._b2o_nm = mk_o_nm_matrix(nbeads)
        self._o_nm2b = self._b2o_nm.T

        self.qdummy = np.zeros(
            (self.nbeads, 3 * self.natoms),
            dtype="float32" if self.single_precision else "float64",
        )
        self.qnmdummy = np.zeros(
            (self.nbeads // 2 + 1, 3 * self.natoms),
            dtype="complex64" if self.single_precision else "complex128",
        )

        def dummy_fft(self):
            self.qnmdummy = np.fft.rfft(self.qdummy, axis=0)

        def dummy_ifft(self):
            self.qdummy = np.fft.irfft(self.qnmdummy, n=self.nbeads, axis=0)

        self.fft = lambda: dummy_fft(self)
        self.ifft = lambda: dummy_ifft(self)

    def b2nm(self, q):
        """Transforms a matrix to the normal mode representation.

        Args:
           q: A matrix with nbeads rows and 3*natoms columns,
              in the bead representation.
        """

        if self.nbeads == 1:
            return q
        self.qdummy[:] = q
        self.fft()
        if self.nbeads == 2:
            return self.qnmdummy.real / np.sqrt(self.nbeads)

        nmodes = self.nbeads // 2

        self.qnmdummy /= np.sqrt(self.nbeads)
        qnm = np.zeros(q.shape)
        qnm[0, :] = self.qnmdummy[0, :].real

        if self.nbeads % 2 == 0:
            self.qnmdummy[1:-1, :] *= np.sqrt(2)
            (qnm[1:nmodes, :], qnm[self.nbeads : nmodes : -1, :]) = (
                self.qnmdummy[1:-1, :].real,
                self.qnmdummy[1:-1, :].imag,
            )
            qnm[nmodes, :] = self.qnmdummy[nmodes, :].real
        else:
            self.qnmdummy[1:, :] *= np.sqrt(2)
            (qnm[1 : nmodes + 1, :], qnm[self.nbeads : nmodes : -1, :]) = (
                self.qnmdummy[1:, :].real,
                self.qnmdummy[1:, :].imag,
            )

        for io in self._open:  # does separately the transformation for the atom that are marked as open paths
            qnm[:, 3 * io] = np.dot(self._b2o_nm, q[:, 3 * io])
            qnm[:, 3 * io + 1] = np.dot(self._b2o_nm, q[:, 3 * io + 1])
            qnm[:, 3 * io + 2] = np.dot(self._b2o_nm, q[:, 3 * io + 2])
        return qnm

    def nm2b(self, qnm):
        """Transforms a matrix to the bead representation.

        Args:
           qnm: A matrix with nbeads rows and 3*natoms columns,
              in the normal mode representation.
        """

        nbeads = self.nbeads
        if nbeads == 1:
            return qnm
        if nbeads == 2:
            self.qnmdummy[:] = qnm
            self.ifft()
            return self.qdummy * np.sqrt(nbeads)

        nmodes = nbeads // 2
        odd = nbeads - 2 * nmodes  # 0 if even, 1 if odd

        isqrt2 = np.sqrt(0.5)
        qnm_complex = self.qnmdummy
        qnm_complex[0, :] = qnm[0, :]
        if not odd:
            qnm_complex[1:-1, :].real = qnm[1:nmodes, :] * isqrt2
            qnm_complex[1:-1, :].imag = qnm[nbeads:nmodes:-1, :] * isqrt2
            qnm_complex[nmodes, :] = qnm[nmodes, :]
        else:
            qnm_complex[1:, :].real = qnm[1 : nmodes + 1, :] * isqrt2
            qnm_complex[1:, :].imag = qnm[nbeads:nmodes:-1, :] * isqrt2

        self.ifft()
        q = self.qdummy * np.sqrt(nbeads)
        for io in self._open:  # does separately the transformation for the atom that are marked as open paths
            q[:, 3 * io] = np.dot(self._o_nm2b, qnm[:, 3 * io])
            q[:, 3 * io + 1] = np.dot(self._o_nm2b, qnm[:, 3 * io + 1])
            q[:, 3 * io + 2] = np.dot(self._o_nm2b, qnm[:, 3 * io + 2])
        return q


class HarmFEP:
    def __init__(self, ref, nbeads=24, lmd=None):
        self.ref = ref
        self.natoms = len(ref.masses)
        self.nbeads = nbeads
        self.freqs, self.modes, self.masses = ref.freqs, ref.modes.reshape([ref.modes.shape[0], -1]), ref.masses

        # Mass scaling factor
        self.lmd = lmd if lmd is not None else 1.0
        self.resize(ref.x, self.masses, nbeads, self.lmd)

        # Normal mode <-> Cartesian coordinates transformation
        self.transform = nm_fft(nbeads=self.nbeads, natoms=self.natoms)
        self._nm_freq = None

    def resize(self, x, masses, nbeads, lmd):
        """Initialize ring polymer positions and masses."""
        # Ring polymer position
        pos_flat = x.reshape(-1)  # Shape: (natoms*3,)
        self.npos = np.tile(pos_flat, (nbeads, 1))  # Shape: (nbeads, natoms*3)

        # Get masses and repeat for x,y,z coordinates
        mass3 = np.repeat(masses, 3)  # Shape: (natoms*3,)

        # Apply mass scaling if provided
        if lmd != 1.0:
            raise
            mass3 = mass3 / (lmd**2)

        # Ring polymer mass
        self.nmass3 = np.tile(mass3, (nbeads, 1))  # Shape: (nbeads, natoms*3)

    def get_nm_freq(self):
        """Calculate normal mode frequencies (dimensionless)."""
        if self._nm_freq is None:
            self._nm_freq = 2 * np.array([np.sin(k * np.pi / self.nbeads) for k in range(self.nbeads)])
        return self._nm_freq

    def calculate_variance(self, ret_freq=False):
        """
        Calculate variance for normal mode coordinates.

        Returns:
            sigma: Standard deviation array in Angstrom, shape (nbeads, natoms*3)
        """
        # Get dimensionless normal mode frequencies
        nm_freq_dimensionless = self.get_nm_freq()

        # Calculate beta
        temperature: float = self.ref.T
        beta = 1.0 / (KB * temperature)  # 1/Hartree

        # Calculate the ring polymer frequency scale, omega_P = P / (beta*hbar)
        omega_P = self.nbeads / (beta * HBAR)  # 1/au.time

        # Calculate actual normal mode frequencies
        nm_freq_with_units = omega_P * nm_freq_dimensionless  # 1/au.time

        # Hartree -> 1/ au.time
        mode_freq_with_units = np.where(self.freqs > 0, self.freqs, 0.0) / HBAR
        imag_freq_with_units = np.where(self.freqs < 0, self.freqs, 0.0) / HBAR

        for i in range(len(imag_freq_with_units)):
            imag_freq_with_units[i] = -min(-imag_freq_with_units[i], nm_freq_with_units[1] * 0.8)  #
        self.freqs = imag_freq_with_units * HBAR + mode_freq_with_units

        # Calculate variance combining normal modes and harmonic modes
        # Normal mode variance: sigma^2 = P / (m * beta * omega^2)
        # Effective harmonic frequencies: omega_eff^2 = omega_real^2 - omega_imag^2 (should be stable with small imaginary frequencies)
        variance = np.zeros_like(self.nmass3)

        for k in range(self.nbeads):
            if nm_freq_dimensionless[k] == 0:  # centroid mode
                variance[k, :] = 0  # fix centroid motion
                # Pure harmonic variance with effective frequency squared
                self.var0 = np.sqrt(
                    self.nbeads
                    / (AMU * beta)
                    / (
                        (lambda s: np.array([1e-200 if abs(x) < 1e-20 else x for x in s]))(
                            mode_freq_with_units**2 - imag_freq_with_units**2
                        )
                    ),
                    dtype=complex,
                )  # au.length

            else:
                omega_k = nm_freq_with_units[k]  # 1/au.time
                variance[k, :] = np.sqrt(
                    self.nbeads / (AMU * beta) / (omega_k**2 + mode_freq_with_units**2 - imag_freq_with_units**2)
                )  # au.length

        if ret_freq:
            return variance, nm_freq_with_units
        else:
            return variance

    def sample_normal_modes(self, n_samples):
        """
        Sample normal mode coordinates using quasi-random numbers.

        Parameters:
            n_samples: Number of samples (should be power of 2)

        Returns:
            sampled_nm_pos: Array of shape (n_samples, nbeads, natoms*3)
        """
        # Transform current positions to normal modes
        nm_pos = self.transform.b2nm(self.npos)  # Shape: (nbeads, natoms*3)

        # Calculate variance (standard deviation)
        sigma = self.calculate_variance()  # Shape: (nbeads, natoms*3)
        self.sigma = sigma

        # Generate samples
        sampled_nm_pos = sobol_gaussian_sample(loc=nm_pos * 0, scale=sigma, M0=n_samples)

        return sampled_nm_pos  # Shape: (n_samples, nbeads, natoms*3)

    def nm_to_beads(self, nm_coordinates):
        """
        Transform normal mode coordinates back to bead coordinates.

        Parameters:
            nm_coordinates: Array of shape (n_samples, nbeads, natoms*3)

        Returns:
            bead_coordinates: Array of shape (n_samples, nbeads, natoms*3)
        """
        n_samples = nm_coordinates.shape[0]
        bead_coords = np.zeros_like(nm_coordinates)

        for i in range(n_samples):
            bead_coords[i] = self.transform.nm2b(nm_coordinates[i])

        return bead_coords

    def get_cart_pos(self, nm_pos: NDArray) -> NDArray:
        # Transform back to bead coordinates of displacement
        sampled_bead_pos = self.nm_to_beads(nm_pos)

        # Calculate harmonic energies
        sampled_mean_square = np.average(sampled_bead_pos**2, axis=1)
        # Force constant: k = P*kB*T/sigma^2
        temp = self.nbeads * KB * self.ref.T / np.real(self.var0**2)  # Hartree / au.length
        self.harm_energies = np.sum(sampled_mean_square * temp, axis=1) / 2 * self.lmd**2  # Hartree

        # Real position with displacement plus centroid coordinates
        return np.einsum("nbi,ij->nbj", sampled_bead_pos, self.modes) + self.npos


class InstFEP(HarmFEP):
    def __init__(self, inst, nbeads=24, lmd=None):
        assert nbeads == inst.N
        super().__init__(inst, nbeads, lmd)

        self.npos: NDArray = np.concatenate((inst.x, inst.x[::-1]))

        self.beta: float = inst.beta

        hess: NDArray = self.rp.hessian_full

        centroid_mode = np.tile(np.eye(np.prod(self.npos.shape[1:], dtype=int)), (1, nbeads))
        centroid_mode *= np.sqrt(self.nmass3.reshape(1, -1))
        centroid_mode /= np.linalg.norm(centroid_mode, axis=1, keepdims=True)
        hess = mass_weight(hess, self.npos.shape, self.masses)

        # nm: Cartesian normal modes in columns with permutation mode removed
        freq_rp2, self.nm = proj_eig(
            x=self.npos, hess=hess, mass=self.masses, constr_vecs=centroid_mode
        )  # 1 / au.time **2

        slc = slice(1, None) if np.isclose(self.ref.BN, 0) else slice(None, None)

        self.freq_rp = np.sqrt(freq_rp2[slc]) * HBAR  # Hartree

        self.nm = self.nm[:, slc] / np.sqrt(self.nmass3.reshape(-1, 1))

    def calculate_variance(self, ret_freq=False):
        """
        Calculate variance for normal mode coordinates.

        Returns:
            sigma: Standard deviation array in Angstrom, shape (nbeads-1)*natoms*3-1
        """
        variance: NDArray = np.sqrt(self.nbeads / (AMU * self.beta * self.freq_rp**2))  # au.length
        if ret_freq:
            return variance, self.freq_rp
        else:
            return variance

    def sample_normal_modes(self, n_samples):
        """
        Sample normal mode coordinates using quasi-random numbers.

        Parameters:
            temperature: Temperature in Kelvin
            n_samples: Number of samples (should be power of 2)

        Returns:
            sampled_nm_pos: Array of shape (n_samples, (nbeads-1)*natoms*3-1)
        """
        self.sigma = self.calculate_variance()  # Shape: (nbeads-1)*natoms*3
        return sobol_gaussian_sample(loc=np.zeros_like(self.sigma), scale=self.sigma, M0=n_samples)

    def nm_to_cart(self, nm_coordinates):
        """
        Transform normal mode coordinates back to Cartesian coordinates.

        Parameters:
            nm_coordinates: Array of shape (n_samples, (nbeads-1)*natoms*3-1)

        Returns:
            Array of shape (n_samples, nbeads, natoms, 3)
        """
        return (nm_coordinates @ self.nm.T).reshape(-1, self.nbeads, self.natoms, 3) + self.npos

    def get_cart_pos(self, nm_pos: NDArray) -> NDArray:
        freq: NDArray = self.freq_rp * Energy(1, "au").get("eV")
        cart_pos: NDArray = self.nm_to_cart(nm_pos)
        vhs: NDArray = 0.5 * np.sum((nm_pos * freq) ** 2, axis=1)
        springs: float = self.ref.springs.potential(self.npos)
        for i in range(len(vhs)):
            vhs[i] = (vhs[i] - self.ref.springs.potential(cart_pos[i]) + springs) / self.nbeads
        self.harm_energies = vhs
        return cart_pos
