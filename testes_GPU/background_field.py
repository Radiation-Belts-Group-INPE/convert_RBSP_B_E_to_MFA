import numpy as np
from PyEMD import EMD
from numba import jit
import warnings
from joblib import Parallel, delayed


class BackgroundFieldCalculator:
    """
    Calculate background magnetic field using Empirical Mode Decomposition (EMD).
    Based on Regi et al. 2016 methodology.
    """

    def __init__(self, Tmax=2048, Rmin=0.1, n_jobs=-1):
        """
        Parameters:
        -----------
        Tmax : int
            Maximum period threshold for mean field identification (in samples)
        Rmin : float
            Minimum energy ratio threshold for mean field classification
        n_jobs : int
            Number of parallel CPU jobs for EMD (-1 uses all cores)
        """
        self.Tmax = Tmax
        self.Rmin = Rmin
        self.n_jobs = n_jobs
        
        print("="*60)
        print("Background Field Calculator (EMD-based)")
        print(f"  Tmax: {Tmax} samples")
        print(f"  Rmin: {Rmin}")
        print(f"  Parallel cores: {'all' if n_jobs == -1 else n_jobs}")
        print("="*60)

    def _extract_single_component_emd(self, signal_1d):
        """
        Extract background field from a single magnetic field component using EMD.
        
        Returns:
        --------
        B0 : array
            Background/mean field
        dB : array
            Perturbation/fluctuation field
        """
        emd = EMD()
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                IMFs = emd.emd(signal_1d)
                residue = signal_1d - np.sum(IMFs, axis=0)
        except Exception as e:
            print(f"  Warning: EMD failed, returning original signal")
            return signal_1d, np.zeros_like(signal_1d)

        mean_field_imfs = []
        perturbation_imfs = []

        # Classify each IMF as mean field or perturbation
        for imf in IMFs:
            avg_period, energy_ratio = self._analyze_imf(imf, self.Tmax)
            
            if avg_period > self.Tmax or energy_ratio > self.Rmin:
                mean_field_imfs.append(imf)
            else:
                perturbation_imfs.append(imf)

        # Background field = residue + mean field IMFs
        B0 = residue.copy()
        if mean_field_imfs:
            B0 += np.sum(mean_field_imfs, axis=0)

        # Perturbation field = high-frequency IMFs
        dB = np.zeros_like(signal_1d)
        if perturbation_imfs:
            dB = np.sum(perturbation_imfs, axis=0)

        return B0, dB

    @staticmethod
    @jit(nopython=True, cache=True)
    def _analyze_imf(imf, Tmax):
        """
        Analyze IMF to determine if it belongs to mean field or perturbation.
        
        Returns:
        --------
        avg_period : float
            Average period of the IMF
        energy_ratio : float
            Ratio of energy in periods > Tmax to total energy
        """
        # Find zero crossings
        sign_changes = np.diff(np.sign(imf))
        zero_crossings = np.where(sign_changes != 0)[0]

        if len(zero_crossings) < 2:
            return np.inf, 1.0

        periods = np.zeros(len(zero_crossings) - 1)
        energies = np.zeros(len(zero_crossings) - 1)

        # Calculate period and energy for each oscillation
        for i in range(len(zero_crossings) - 1):
            start = zero_crossings[i]
            end = zero_crossings[i + 1]
            periods[i] = 2 * (end - start)  # Full period = 2 * half period
            energies[i] = np.sum(imf[start:end]**2)

        avg_period = np.mean(periods)
        total_energy = np.sum(energies)
        
        # Calculate energy ratio for long periods
        if total_energy > 0:
            long_period_energy = 0.0
            for i in range(len(periods)):
                if periods[i] > Tmax:
                    long_period_energy += energies[i]
            energy_ratio = long_period_energy / total_energy
        else:
            energy_ratio = 0.0

        return avg_period, energy_ratio

    def calculate_background_field(self, b_field_3d):
        """
        Calculate background magnetic field from 3D magnetic field data.
        
        Parameters:
        -----------
        b_field_3d : array, shape (n_samples, 3)
            Magnetic field in GSM coordinates [Bx, By, Bz]
        
        Returns:
        --------
        B0 : array, shape (n_samples, 3)
            Background/mean magnetic field
        dB : array, shape (n_samples, 3)
            Perturbation/fluctuation field
        """
        print("\nCalculating background field using parallel EMD...")
        
        # Process each component (Bx, By, Bz) in parallel
        results = Parallel(n_jobs=self.n_jobs, backend='loky', verbose=5)(
            delayed(self._extract_single_component_emd)(b_field_3d[:, i])
            for i in range(3)
        )
        
        # Combine results
        B0 = np.column_stack([r[0] for r in results])
        dB = np.column_stack([r[1] for r in results])
        
        print("\n✓ Background field calculation complete!")
        print(f"  Original field magnitude: {np.mean(np.linalg.norm(b_field_3d, axis=1)):.2f} nT")
        print(f"  Background field magnitude: {np.mean(np.linalg.norm(B0, axis=1)):.2f} nT")
        print(f"  Perturbation RMS: {np.sqrt(np.mean(dB**2)):.2f} nT")
        
        return B0, dB


# Simple usage function
def calculate_background_field(b_gsm, Tmax=2048, Rmin=0.1, n_jobs=-1):
    """
    Simple function to calculate background magnetic field.
    
    Parameters:
    -----------
    b_gsm : array, shape (n_samples, 3)
        Magnetic field in GSM coordinates [Bx, By, Bz] in nT
    Tmax : int
        Maximum period threshold (in samples)
    Rmin : float
        Minimum energy ratio threshold
    n_jobs : int
        Number of CPU cores (-1 = all cores)
    
    Returns:
    --------
    B0 : array, shape (n_samples, 3)
        Background magnetic field
    dB : array, shape (n_samples, 3)
        Perturbation field
    
    Example:
    --------
    >>> B0, dB = calculate_background_field(b_gsm, Tmax=2048, n_jobs=-1)
    >>> # B0 is your background field
    >>> # dB is the perturbation/fluctuation
    """
    calculator = BackgroundFieldCalculator(Tmax=Tmax, Rmin=Rmin, n_jobs=n_jobs)
    return calculator.calculate_background_field(b_gsm)