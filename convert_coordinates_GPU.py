import numpy as np
from scipy import signal
from PyEMD import EMD
from numba import cuda, jit
import warnings

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from joblib import Parallel, delayed


class VanAllenProbesCoordinateConverterGPU:
    """
    GPU-accelerated converter using CuPy and CUDA.
    """

    def __init__(self, Tmax=2048, Rmin=0.1, alpha=0.05, theta1=0.05, theta2=0.50, 
                 n_jobs=-1, use_gpu=True):
        """
        Parameters:
        -----------
        use_gpu : bool
            Use GPU acceleration if available (requires CuPy and CUDA)
        n_jobs : int
            Number of parallel CPU jobs for EMD (-1 uses all cores)
        """
        self.Tmax = Tmax
        self.Rmin = Rmin
        self.alpha = alpha
        self.theta1 = theta1
        self.theta2 = theta2
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu and self._check_gpu_available()
        
        if self.use_gpu:
            device = cp.cuda.Device()
            # Get device name properly
            device_name = cp.cuda.runtime.getDeviceProperties(device.id)['name'].decode('utf-8')
            print(f"GPU acceleration enabled: {device_name}")
            print(f"GPU memory: {device.mem_info[1] / 1e9:.2f} GB total")
        else:
            if use_gpu and not CUPY_AVAILABLE:
                print("CuPy not installed. Install with: pip install cupy-cuda12x")
            print("Using CPU acceleration")

    @staticmethod
    def _check_gpu_available():
        """Check if GPU is available."""
        if not CUPY_AVAILABLE:
            return False
        try:
            cp.cuda.Device(0).compute_capability
            return True
        except Exception as e:
            print(f"GPU check failed: {e}")
            return False

    @staticmethod
    def running_average_gpu(data, window_size):
        """GPU-accelerated running average using CuPy."""
        if data.ndim == 1:
            data_gpu = cp.asarray(data)
            kernel = cp.ones(window_size) / window_size
            result = cp.convolve(data_gpu, kernel, mode='same')
            return cp.asnumpy(result)
        else:
            data_gpu = cp.asarray(data)
            kernel = cp.ones(window_size) / window_size
            result = cp.zeros_like(data_gpu)
            for i in range(data.shape[1]):
                result[:, i] = cp.convolve(data_gpu[:, i], kernel, mode='same')
            return cp.asnumpy(result)

    @staticmethod
    def running_average_cpu(data, window_size):
        """CPU fallback for running average."""
        kernel = np.ones(window_size) / window_size
        if data.ndim == 1:
            return np.convolve(data, kernel, mode='same')
        else:
            return np.array([np.convolve(data[:, i], kernel, mode='same')
                           for i in range(data.shape[1])]).T

    def running_average(self, data, window_size):
        """Adaptive running average (GPU or CPU)."""
        if self.use_gpu:
            return self.running_average_gpu(data, window_size)
        else:
            return self.running_average_cpu(data, window_size)

    def _extract_single_component_emd(self, signal_1d):
        """Extract mean field from single component (CPU-based EMD)."""
        emd = EMD()
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                IMFs = emd.emd(signal_1d)
                residue = signal_1d - np.sum(IMFs, axis=0)
        except Exception as e:
            print(f"EMD failed: {e}, returning original signal")
            return signal_1d, np.zeros_like(signal_1d)

        mean_field_imfs = []
        perturbation_imfs = []

        for imf in IMFs:
            avg_period, energy_ratio = self._analyze_imf_cpu(imf, self.Tmax)
            
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

    @staticmethod
    @jit(nopython=True, cache=True)
    def _analyze_imf_cpu(imf, Tmax):
        """CPU-optimized IMF analysis with Numba."""
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

    def extract_mean_field_emd_parallel(self, b_field_3d):
        """
        Extract mean field using parallel EMD (CPU-based, EMD doesn't support GPU).
        """
        print(f"Running parallel EMD on {self.n_jobs} cores...")
        results = Parallel(n_jobs=self.n_jobs, backend='loky', verbose=5)(
            delayed(self._extract_single_component_emd)(b_field_3d[:, i])
            for i in range(3)
        )
        
        B0_emd = np.column_stack([r[0] for r in results])
        perturbation_emd = np.column_stack([r[1] for r in results])
        
        return B0_emd, perturbation_emd

    def _transform_to_mfa_cupy(self, b_field, b_mean, sat_position):
        """GPU-accelerated MFA transformation using CuPy (vectorized)."""
        n_samples = b_field.shape[0]
        
        # Transfer to GPU
        b_field_gpu = cp.asarray(b_field, dtype=cp.float64)
        b_mean_gpu = cp.asarray(b_mean, dtype=cp.float64)
        sat_position_gpu = cp.asarray(sat_position, dtype=cp.float64)
        
        # Vectorized operations on GPU
        b_mean_norm = cp.linalg.norm(b_mean_gpu, axis=1, keepdims=True)
        b_mean_norm = cp.where(b_mean_norm > 1e-10, b_mean_norm, 1.0)
        e_parallel = b_mean_gpu / b_mean_norm
        
        # Azimuthal direction (cross product)
        e_azimuthal = cp.cross(e_parallel, sat_position_gpu)
        e_azimuthal_norm = cp.linalg.norm(e_azimuthal, axis=1, keepdims=True)
        e_azimuthal_norm = cp.where(e_azimuthal_norm > 1e-10, e_azimuthal_norm, 1.0)
        e_azimuthal = e_azimuthal / e_azimuthal_norm
        
        # Radial direction
        e_radial = cp.cross(e_azimuthal, e_parallel)
        
        # Transform to MFA (vectorized dot products)
        b_mfa_gpu = cp.zeros_like(b_field_gpu)
        b_mfa_gpu[:, 0] = cp.sum(b_field_gpu * e_parallel, axis=1)
        b_mfa_gpu[:, 1] = cp.sum(b_field_gpu * e_radial, axis=1)
        b_mfa_gpu[:, 2] = cp.sum(b_field_gpu * e_azimuthal, axis=1)
        
        return cp.asnumpy(b_mfa_gpu)

    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def _transform_to_mfa_cpu(b_field, b_mean, sat_position):
        """CPU fallback with Numba parallel."""
        n_samples = b_field.shape[0]
        b_mfa = np.zeros_like(b_field)
        
        for i in range(n_samples):
            b_mean_norm = np.sqrt(b_mean[i, 0]**2 + b_mean[i, 1]**2 + b_mean[i, 2]**2)
            if b_mean_norm > 1e-10:
                e_parallel = b_mean[i] / b_mean_norm
            else:
                e_parallel = np.array([0.0, 0.0, 1.0])

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

            e_radial = np.array([
                e_azimuthal[1] * e_parallel[2] - e_azimuthal[2] * e_parallel[1],
                e_azimuthal[2] * e_parallel[0] - e_azimuthal[0] * e_parallel[2],
                e_azimuthal[0] * e_parallel[1] - e_azimuthal[1] * e_parallel[0]
            ])

            b_mfa[i, 0] = b_field[i, 0] * e_parallel[0] + b_field[i, 1] * e_parallel[1] + b_field[i, 2] * e_parallel[2]
            b_mfa[i, 1] = b_field[i, 0] * e_radial[0] + b_field[i, 1] * e_radial[1] + b_field[i, 2] * e_radial[2]
            b_mfa[i, 2] = b_field[i, 0] * e_azimuthal[0] + b_field[i, 1] * e_azimuthal[1] + b_field[i, 2] * e_azimuthal[2]
        
        return b_mfa

    def gsm_to_mfa_emd(self, b_field, sat_position):
        """
        GSM to MFA conversion with GPU acceleration.
        """
        # EMD is CPU-based (parallel across components)
        b_mean, _ = self.extract_mean_field_emd_parallel(b_field)
        
        # MFA transformation on GPU or CPU
        print("Transforming to MFA coordinates...")
        if self.use_gpu:
            b_mfa = self._transform_to_mfa_cupy(b_field, b_mean, sat_position)
        else:
            b_mfa = self._transform_to_mfa_cpu(b_field, b_mean, sat_position)
        
        return b_mfa, b_mean

    @staticmethod
    def calculate_psd(data, dt, nperseg, overlap_fraction=0.8):
        """Calculate PSD (CPU-based, scipy doesn't support GPU)."""
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
    def gse_to_mgse(e_field, spin_axis_gse):
        """Convert GSE to mGSE."""
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
    def remove_vxb_field(e_field, sat_velocity, b_field):
        """Remove V×B field."""
        e_corrected = e_field.copy()
        for i in range(e_field.shape[0]):
            v_ms = sat_velocity[i] * 1000
            b_t = b_field[i] * 1e-9
            vxb = np.cross(v_ms, b_t)
            e_corrected[i] -= vxb * 1000
        return e_corrected

    @staticmethod
    def remove_corotation_field(e_field, sat_position, b_field, omega_earth=7.2921e-5):
        """Remove co-rotation field."""
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


def process_emfisis_data_gpu(b_gsm, sat_position, time_array, Tmax=2048, 
                              use_emd=True, use_gpu=True, n_jobs=-1):
    """
    GPU-accelerated EMFISIS data processing.
    
    Parameters:
    -----------
    use_gpu : bool
        Enable GPU acceleration (requires CuPy and CUDA)
    n_jobs : int
        Number of CPU cores for parallel EMD
    """
    converter = VanAllenProbesCoordinateConverterGPU(Tmax=Tmax, n_jobs=n_jobs, use_gpu=use_gpu)

    # Running average (GPU-accelerated if available)
    print("Applying running average...")
    b_smoothed = converter.running_average(b_gsm, window_size=11)

    # Convert to MFA (GPU-accelerated transformation)
    if use_emd:
        b_mfa, b_mean = converter.gsm_to_mfa_emd(b_smoothed, sat_position)
    else:
        # Simple moving average fallback
        b_mean = converter.running_average(b_smoothed, window_size=Tmax)
        if use_gpu and converter.use_gpu:
            b_mfa = converter._transform_to_mfa_cupy(b_smoothed, b_mean, sat_position)
        else:
            b_mfa = converter._transform_to_mfa_cpu(b_smoothed, b_mean, sat_position)

    # Calculate PSD
    print("Calculating PSD...")
    dt = np.mean(np.diff(time_array))
    if hasattr(dt, 'total_seconds'):
        dt = dt.total_seconds()
    elif isinstance(dt, np.timedelta64):
        dt = dt / np.timedelta64(1, 's')

    frequencies, psd, psd_times = converter.calculate_psd(
        b_mfa, dt=dt, nperseg=2048, overlap_fraction=0.8
    )

    frequencies_mhz = frequencies * 1000

    print("Processing complete!")
    if use_emd:
        return frequencies_mhz, psd, psd_times, b_mfa, b_mean
    else:
        return frequencies_mhz, psd, psd_times, b_mfa


def process_efw_data_gpu(e_mgse, b_mgse, sat_velocity, sat_position, time_array):
    """Process EFW data."""
    converter = VanAllenProbesCoordinateConverterGPU()

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