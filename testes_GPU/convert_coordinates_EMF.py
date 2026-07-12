#%%
import numpy as np
from scipy import signal
from PyEMD import EMD
from numba import jit, prange
from joblib import Parallel, delayed
import warnings

class VanAllenProbesCoordinateConverterEMD:
    """
    Optimized converter with parallelization and Numba JIT compilation.
    """

    def __init__(self, Tmax=2048, Rmin=0.1, alpha=0.05, theta1=0.05, theta2=0.50, n_jobs=-1):
        """
        Parameters:
        -----------
        n_jobs : int
            Number of parallel jobs (-1 uses all cores)
        """
        self.Tmax = Tmax
        self.Rmin = Rmin
        self.alpha = alpha
        self.theta1 = theta1
        self.theta2 = theta2
        self.n_jobs = n_jobs

    @staticmethod
    @jit(nopython=True, cache=True)
    def _running_average_numba(data_1d, window_size):
        """Numba-optimized running average for 1D array."""
        n = len(data_1d)
        result = np.zeros(n)
        half_window = window_size // 2
        
        for i in range(n):
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)
            result[i] = np.mean(data_1d[start:end])
        
        return result

    @staticmethod
    def running_average(data, window_size):
        """Fast running average using Numba."""
        if data.ndim == 1:
            return VanAllenProbesCoordinateConverterEMD._running_average_numba(data, window_size)
        else:
            return np.array([VanAllenProbesCoordinateConverterEMD._running_average_numba(data[:, i], window_size)
                           for i in range(data.shape[1])]).T

    def _extract_single_component_emd(self, signal_1d):
        """Extract mean field from single component (for parallel processing)."""
        emd = EMD()
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                IMFs = emd.emd(signal_1d)
                residue = signal_1d - np.sum(IMFs, axis=0)
        except:
            return signal_1d, np.zeros_like(signal_1d)

        mean_field_imfs = []
        perturbation_imfs = []

        for imf in IMFs:
            avg_period, energy_ratio = self._analyze_imf_numba(imf, self.Tmax)
            
            if avg_period > self.Tmax or energy_ratio > self.Rmin:
                mean_field_imfs.append(imf)
            else:
                perturbation_imfs.append(imf)

        B0_emd = residue.copy()
        if mean_field_imfs:
            B0_emd += np.sum(mean_field_imfs, axis=0)

        perturbation_emd = np.zeros_like(signal_1d)
        if perturbation_imfs:
            perturbation_emd = np.sum(perturbation_imfs, axis=0)

        return B0_emd, perturbation_emd

    def extract_mean_field_emd_parallel(self, b_field_3d):
        """
        Extract mean field from 3D magnetic field using parallel EMD.
        
        Parameters:
        -----------
        b_field_3d : array-like, shape (n_samples, 3)
            3-component magnetic field
            
        Returns:
        --------
        B0_emd : array-like, shape (n_samples, 3)
            Mean field for all components
        perturbation_emd : array-like, shape (n_samples, 3)
            Perturbation for all components
        """
        # Process all 3 components in parallel
        results = Parallel(n_jobs=self.n_jobs, backend='loky')(
            delayed(self._extract_single_component_emd)(b_field_3d[:, i])
            for i in range(3)
        )
        
        B0_emd = np.column_stack([r[0] for r in results])
        perturbation_emd = np.column_stack([r[1] for r in results])
        
        return B0_emd, perturbation_emd

    @staticmethod
    @jit(nopython=True, cache=True)
    def _analyze_imf_numba(imf, Tmax):
        """Numba-optimized IMF analysis."""
        # Find zero crossings
        sign_changes = np.diff(np.sign(imf))
        zero_crossings = np.where(sign_changes != 0)[0]

        if len(zero_crossings) < 2:
            return np.inf, 1.0

        periods = np.zeros(len(zero_crossings) - 1)
        energies = np.zeros(len(zero_crossings) - 1)

        for i in range(len(zero_crossings) - 1):
            start = zero_crossings[i]
            end = zero_crossings[i + 1]
            
            periods[i] = 2 * (end - start)
            energies[i] = np.sum(imf[start:end]**2)

        avg_period = np.mean(periods)
        
        total_energy = np.sum(energies)
        if total_energy > 0:
            long_period_energy = 0.0
            for i in range(len(periods)):
                if periods[i] > Tmax:
                    long_period_energy += energies[i]
            energy_ratio = long_period_energy / total_energy
        else:
            energy_ratio = 0.0

        return avg_period, energy_ratio

    def _analyze_imf(self, imf):
        """Wrapper for Numba-optimized IMF analysis."""
        return self._analyze_imf_numba(imf, self.Tmax)

    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def _transform_to_mfa_numba(b_field, b_mean, sat_position):
        """Numba-optimized MFA transformation."""
        n_samples = b_field.shape[0]
        b_mfa = np.zeros_like(b_field)
        
        for i in prange(n_samples):
            # Parallel direction
            b_mean_norm = np.sqrt(b_mean[i, 0]**2 + b_mean[i, 1]**2 + b_mean[i, 2]**2)
            if b_mean_norm > 1e-10:
                e_parallel = b_mean[i] / b_mean_norm
            else:
                e_parallel = np.array([0.0, 0.0, 1.0])

            # Azimuthal direction
            e_azimuthal = np.array([
                e_parallel[1] * sat_position[i, 2] - e_parallel[2] * sat_position[i, 1],
                e_parallel[2] * sat_position[i, 0] - e_parallel[0] * sat_position[i, 2],
                e_parallel[0] * sat_position[i, 1] - e_parallel[1] * sat_position[i, 0]
            ])
            e_azimuthal_norm = np.sqrt(e_azimuthal[0]**2 + e_azimuthal[1]**2 + e_azimuthal[2]**2)
            if e_azimuthal_norm > 1e-10:
                e_azimuthal = e_azimuthal / e_azimuthal_norm
            else:
                e_azimuthal = np.array([0.0, 1.0, 0.0])

            # Radial direction
            e_radial = np.array([
                e_azimuthal[1] * e_parallel[2] - e_azimuthal[2] * e_parallel[1],
                e_azimuthal[2] * e_parallel[0] - e_azimuthal[0] * e_parallel[2],
                e_azimuthal[0] * e_parallel[1] - e_azimuthal[1] * e_parallel[0]
            ])

            # Transform
            b_mfa[i, 0] = b_field[i, 0] * e_parallel[0] + b_field[i, 1] * e_parallel[1] + b_field[i, 2] * e_parallel[2]
            b_mfa[i, 1] = b_field[i, 0] * e_radial[0] + b_field[i, 1] * e_radial[1] + b_field[i, 2] * e_radial[2]
            b_mfa[i, 2] = b_field[i, 0] * e_azimuthal[0] + b_field[i, 1] * e_azimuthal[1] + b_field[i, 2] * e_azimuthal[2]
        
        return b_mfa

    def gsm_to_mfa_emd(self, b_field, sat_position):
        """
        Optimized GSM to MFA conversion using parallel EMD.
        """
        # Extract mean field using parallel EMD
        b_mean, _ = self.extract_mean_field_emd_parallel(b_field)
        
        # Transform to MFA using Numba
        b_mfa = self._transform_to_mfa_numba(b_field, b_mean, sat_position)
        
        return b_mfa, b_mean

    # Keep other methods unchanged...
    @staticmethod
    def gsm_to_mfa_mavg(b_field, sat_position, avg_window=2048):
        """Legacy moving average method."""
        n_samples = b_field.shape[0]
        b_mfa = np.zeros_like(b_field)
        b_mean = VanAllenProbesCoordinateConverterEMD.running_average(b_field, avg_window)
        b_mfa = VanAllenProbesCoordinateConverterEMD._transform_to_mfa_numba(b_field, b_mean, sat_position)
        return b_mfa

    @staticmethod
    def gse_to_mgse(e_field, spin_axis_gse):
        """Convert electric field from GSE to mGSE coordinates."""
        n_samples = e_field.shape[0]
        e_mgse = np.zeros_like(e_field)
        z_gse = np.array([0, 0, 1])

        for i in range(n_samples):
            x_mgse = spin_axis_gse[i] / np.linalg.norm(spin_axis_gse[i])
            y_mgse = np.cross(z_gse, x_mgse)
            y_mgse = y_mgse / np.linalg.norm(y_mgse)
            z_mgse = np.cross(x_mgse, y_mgse)
            transform_matrix = np.array([x_mgse, y_mgse, z_mgse])
            e_mgse[i] = transform_matrix @ e_field[i]

        return e_mgse

    @staticmethod
    def calculate_psd(data, dt, nperseg, overlap_fraction=0.8):
        """Calculate Power Spectral Density using overlapped FFT."""
        noverlap = int(nperseg * overlap_fraction)

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_components = data.shape[1]
        psd_list = []
        for i in range(n_components):
            f, t, Sxx = signal.spectrogram(data[:, i], fs=1/dt,
                                          nperseg=nperseg,
                                          noverlap=noverlap,
                                          scaling='density')
            psd_list.append(Sxx)

        psd = np.array(psd_list).transpose(1, 2, 0)
        return f, psd, t

    @staticmethod
    def remove_vxb_field(e_field, sat_velocity, b_field):
        """Remove V_sc × B electric field."""
        e_corrected = e_field.copy()
        for i in range(e_field.shape[0]):
            v_ms = sat_velocity[i] * 1000
            b_t = b_field[i] * 1e-9
            vxb = np.cross(v_ms, b_t)
            e_corrected[i] -= vxb * 1000
        return e_corrected

    @staticmethod
    def remove_corotation_field(e_field, sat_position, b_field, omega_earth=7.2921e-5):
        """Remove co-rotation electric field."""
        e_corrected = e_field.copy()
        omega_vec = np.array([0, 0, omega_earth])
        R_earth = 6371e3

        for i in range(e_field.shape[0]):
            pos_m = sat_position[i] * R_earth
            v_corot = np.cross(omega_vec, pos_m)
            b_t = b_field[i] * 1e-9
            e_corot = -np.cross(v_corot, b_t)
            e_corrected[i] -= e_corot * 1000

        return e_corrected


def process_emfisis_data_emd(b_gsm, sat_position, time_array, Tmax=2048, use_emd=True, n_jobs=-1):
    """
    Optimized EMFISIS data processing with parallelization.
    
    Parameters:
    -----------
    n_jobs : int
        Number of parallel jobs (-1 uses all CPU cores)
    """
    converter = VanAllenProbesCoordinateConverterEMD(Tmax=Tmax, n_jobs=n_jobs)

    # Step 1: Running average
    b_smoothed = converter.running_average(b_gsm, window_size=11)

    # Step 2: Convert to MFA
    if use_emd:
        b_mfa, b_mean = converter.gsm_to_mfa_emd(b_smoothed, sat_position)
    else:
        b_mfa = converter.gsm_to_mfa_mavg(b_smoothed, sat_position, avg_window=Tmax)
        b_mean = None

    # Step 3: Calculate PSD
    dt = np.mean(np.diff(time_array))
    if hasattr(dt, 'total_seconds'):
        dt = dt.total_seconds()
    elif isinstance(dt, np.timedelta64):
        dt = dt / np.timedelta64(1, 's')

    frequencies, psd, psd_times = converter.calculate_psd(
        b_mfa, dt=dt, nperseg=2048, overlap_fraction=0.8
    )

    frequencies_mhz = frequencies * 1000

    if use_emd:
        return frequencies_mhz, psd, psd_times, b_mfa, b_mean
    else:
        return frequencies_mhz, psd, psd_times, b_mfa


def process_efw_data_emd(e_mgse, b_mgse, sat_velocity, sat_position, time_array):
    """Process EFW electric field data."""
    converter = VanAllenProbesCoordinateConverterEMD()

    e_corrected = converter.remove_vxb_field(e_mgse, sat_velocity, b_mgse)
    e_corrected = converter.remove_corotation_field(e_corrected, sat_position, b_mgse)

    dt = np.mean(np.diff(time_array))
    if hasattr(dt, 'total_seconds'):
        dt = dt.total_seconds()
    elif isinstance(dt, np.timedelta64):
        dt = dt / np.timedelta64(1, 's')

    frequencies, psd, psd_times = converter.calculate_psd(
        e_corrected, dt=dt, nperseg=256, overlap_fraction=0.8
    )

    frequencies_mhz = frequencies * 1000
    return frequencies_mhz, psd, psd_times, e_corrected