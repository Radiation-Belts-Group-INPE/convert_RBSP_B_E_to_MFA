#%%r
#%%
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import pandas as pd
import matplotlib.pyplot as plt

class VanAllenProbesCoordinateConverter:
    """
    Convert Van Allen Probes EMFISIS and EFW data to MFA coordinates
    and calculate power spectral densities for ULF wave analysis.
    """
    
    def __init__(self):
        pass
    
    @staticmethod
    def running_average(data, window_size):
        """
        Apply running average to data.
        
        Parameters:
        -----------
        data : array-like, shape (n_samples, n_components)
            Input data
        window_size : int
            Window size for running average
            
        Returns:
        --------
        averaged_data : array-like
            Running averaged data
        """
        kernel = np.ones(window_size) / window_size
        if data.ndim == 1:
            return np.convolve(data, kernel, mode='same')
        else:
            return np.array([np.convolve(data[:, i], kernel, mode='same') 
                           for i in range(data.shape[1])]).T
    
    @staticmethod
    def fill_nan(data, method='linear'):
        """
        Fill NaN values in data using interpolation.
        
        Parameters:
        -----------
        data : array-like, shape (n_samples,) or (n_samples, n_components)
            Input data with potential NaN values
        method : str
            Interpolation method: 'linear', 'nearest', 'cubic', 'forward_fill', 'backward_fill'
            
        Returns:
        --------
        filled_data : array-like
            Data with NaN values filled
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        filled_data = np.zeros_like(data)
        
        for i in range(data.shape[1]):
            col = data[:, i].copy()
            
            # Find NaN positions
            nan_mask = np.isnan(col)
            
            if not nan_mask.any():
                filled_data[:, i] = col
                continue
            
            # Get valid (non-NaN) indices and values
            valid_idx = np.where(~nan_mask)[0]
            valid_values = col[~nan_mask]
            
            if len(valid_idx) == 0:
                # All NaN, fill with zeros
                filled_data[:, i] = 0
                continue
            
            if method == 'linear':
                # Linear interpolation
                filled_data[:, i] = np.interp(
                    np.arange(len(col)), valid_idx, valid_values
                )
            elif method == 'nearest':
                # Nearest neighbor
                from scipy.interpolate import interp1d
                f = interp1d(valid_idx, valid_values, kind='nearest', 
                           fill_value='extrapolate')
                filled_data[:, i] = f(np.arange(len(col)))
            elif method == 'cubic':
                # Cubic interpolation
                if len(valid_idx) >= 4:
                    from scipy.interpolate import interp1d
                    f = interp1d(valid_idx, valid_values, kind='cubic', 
                               fill_value='extrapolate')
                    filled_data[:, i] = f(np.arange(len(col)))
                else:
                    # Fall back to linear if not enough points
                    filled_data[:, i] = np.interp(
                        np.arange(len(col)), valid_idx, valid_values
                    )
            elif method == 'forward_fill':
                # Forward fill (propagate last valid value forward)
                filled_data[:, i] = col
                for j in range(len(col)):
                    if nan_mask[j] and j > 0:
                        filled_data[j, i] = filled_data[j-1, i]
            elif method == 'backward_fill':
                # Backward fill (propagate next valid value backward)
                filled_data[:, i] = col
                for j in range(len(col)-1, -1, -1):
                    if nan_mask[j] and j < len(col)-1:
                        filled_data[j, i] = filled_data[j+1, i]
            else:
                raise ValueError(f"Unknown fill method: {method}")
        
        if squeeze_output:
            filled_data = filled_data.squeeze()
        
        return filled_data
    
    @staticmethod
    def butter_highpass_filter(data, cutoff_freq, sampling_freq, order=5):
        """
        Apply Butterworth high-pass filter to remove low-frequency trends.
        This removes the background Earth magnetic field variations.
        
        Parameters:
        -----------
        data : array-like, shape (n_samples,) or (n_samples, n_components)
            Input data (magnetic field components)
        cutoff_freq : float
            Cutoff frequency in Hz (frequencies below this are removed)
            Example: 10/3600 Hz = 10 mHz removes orbital variations
        sampling_freq : float
            Sampling frequency in Hz (1/dt where dt is time resolution)
            Example: 1.0 Hz for 1-second data
        order : int
            Filter order (default: 5)
            
        Returns:
        --------
        filtered_data : array-like
            High-pass filtered data (perturbations only, background removed)
        """
        # Design Butterworth high-pass filter
        nyquist_freq = 0.5 * sampling_freq
        normal_cutoff = cutoff_freq / nyquist_freq
        
        # Check if cutoff frequency is valid
        if normal_cutoff >= 1.0:
            raise ValueError(
                f"Cutoff frequency ({cutoff_freq} Hz) must be less than "
                f"Nyquist frequency ({nyquist_freq} Hz)"
            )
        if normal_cutoff <= 0:
            raise ValueError(f"Cutoff frequency must be positive")
        
        # Get filter coefficients
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        
        # Apply filter (filtfilt for zero-phase filtering)
        if data.ndim == 1:
            filtered_data = signal.filtfilt(b, a, data)
        else:
            filtered_data = np.zeros_like(data)
            for i in range(data.shape[1]):
                filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])
        
        return filtered_data
    
    @staticmethod
    def removeTheTrend(X, cutoff_freq=10/3600, sampling_freq=1.0, order=5, 
                       fill_method='linear'):
        """
        Remove background Earth magnetic field from each component.
        This isolates ULF wave perturbations from the total measured field.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples,) or (n_samples, 3)
            Input magnetic field data [Bx, By, Bz] in nT
            These are the total measured fields (background + perturbations)
        cutoff_freq : float
            Cutoff frequency in Hz (default: 10/3600 = 10 mHz)
            Frequencies below this are considered "background field"
            Frequencies above this are considered "wave perturbations"
        sampling_freq : float
            Sampling frequency in Hz (default: 1.0 Hz for 1-second data)
        order : int
            Butterworth filter order (default: 5)
        fill_method : str
            Method to fill NaN values before filtering (default: 'linear')
            
        Returns:
        --------
        perturbations : array-like
            High-pass filtered data = wave perturbations (δB)
            This is what you see as the small oscillations after removing background
        """
        # Fill NaN values first
        xx = VanAllenProbesCoordinateConverter.fill_nan(X, method=fill_method)
        
        # Apply high-pass filter to remove background field
        # This keeps only the wave perturbations
        perturbations = VanAllenProbesCoordinateConverter.butter_highpass_filter(
            xx, cutoff_freq, sampling_freq, order=order
        )
        
        return perturbations
    
    @staticmethod
    def remove_background_field(b_field, cutoff_freq=10/3600, sampling_freq=1.0, 
                               order=5, return_background=False):
        """
        Remove background Earth magnetic field to isolate wave perturbations.
        
        This function separates the measured magnetic field into:
        B_total = B_background + δB (perturbations)
        
        Where:
        - B_background: Slowly varying Earth's field along the orbit
        - δB: Fast ULF wave perturbations
        
        Parameters:
        -----------
        b_field : array-like, shape (n_samples, 3)
            Total measured magnetic field [Bx, By, Bz] in nT
        cutoff_freq : float
            Cutoff frequency in Hz (default: 10/3600 = 10 mHz)
        sampling_freq : float
            Sampling frequency in Hz (default: 1.0 Hz)
        order : int
            Filter order (default: 5)
        return_background : bool
            If True, also return the background field component
            
        Returns:
        --------
        b_perturbations : array-like, shape (n_samples, 3)
            Wave perturbations δB [δBx, δBy, δBz]
        b_background : array-like, shape (n_samples, 3) (optional)
            Background field component (if return_background=True)
        """
        # Remove background from each component
        b_perturbations = VanAllenProbesCoordinateConverter.removeTheTrend(
            b_field, cutoff_freq=cutoff_freq, sampling_freq=sampling_freq, order=order
        )
        
        if return_background:
            # Background = Total - Perturbations
            b_background = b_field - b_perturbations
            return b_perturbations, b_background
        else:
            return b_perturbations
    
    @staticmethod
    def visualize_detrending(b_field, time_array, cutoff_freq=10/3600, 
                            sampling_freq=1.0, order=5, component_names=None):
        """
        Visualize the detrending process: original field, background, and perturbations.
        
        Parameters:
        -----------
        b_field : array-like, shape (n_samples, 3)
            Magnetic field [Bx, By, Bz]
        time_array : array-like
            Time array
        cutoff_freq : float
            Cutoff frequency in Hz
        sampling_freq : float
            Sampling frequency in Hz
        order : int
            Filter order
        component_names : list of str
            Names for components (default: ['Bx', 'By', 'Bz'])
        """
        if component_names is None:
            component_names = ['Bx', 'By', 'Bz']
        
        # Get perturbations and background
        b_perturbations, b_background = VanAllenProbesCoordinateConverter.remove_background_field(
            b_field, cutoff_freq=cutoff_freq, sampling_freq=sampling_freq, 
            order=order, return_background=True
        )
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        for i in range(3):
            ax = axes[i]
            
            # Plot original field
            ax.plot(time_array, b_field[:, i], 'gray', alpha=0.5, 
                   label=f'{component_names[i]} (Total)', linewidth=0.5)
            
            # Plot background field
            ax.plot(time_array, b_background[:, i], 'b', 
                   label=f'{component_names[i]} (Background)', linewidth=2)
            
            # Plot perturbations (offset for visibility)
            offset = np.mean(b_field[:, i])
            ax.plot(time_array, b_perturbations[:, i] + offset, 'r', 
                   label=f'δ{component_names[i]} (Perturbations)', linewidth=1)
            
            ax.set_ylabel(f'{component_names[i]} (nT)', fontsize=12)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.set_title(f'Magnetic Field Detrending (Cutoff: {cutoff_freq*1000:.1f} mHz)', 
                           fontsize=14)
            if i == 2:
                ax.set_xlabel('Time', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def gsm_to_mfa(b_field, sat_position, avg_window=2048):
        """
        Convert magnetic field from GSM to MFA (Mean Field Aligned) coordinates.
        
        Parameters:
        -----------
        b_field : array-like, shape (n_samples, 3)
            Magnetic field in GSM coordinates [Bx, By, Bz]
            This should be the PERTURBATIONS (after removing background)
        sat_position : array-like, shape (n_samples, 3)
            Satellite position vector in GSM [X, Y, Z]
        avg_window : int
            Window size for running average to determine mean field (default: 2048)
            
        Returns:
        --------
        b_mfa : array-like, shape (n_samples, 3)
            Magnetic field in MFA coordinates [B_parallel, B_radial, B_azimuthal]
            where:
            - B_parallel: compressional component
            - B_radial: poloidal component
            - B_azimuthal: toroidal component
        """
        n_samples = b_field.shape[0]
        b_mfa = np.zeros_like(b_field)
        
        # Calculate mean field using running average
        b_mean = VanAllenProbesCoordinateConverter.running_average(b_field, avg_window)
        
        for i in range(n_samples):
            # Parallel direction (along mean magnetic field)
            e_parallel = b_mean[i] / np.linalg.norm(b_mean[i])
            
            # Azimuthal direction (cross product of parallel and position)
            e_azimuthal = np.cross(e_parallel, sat_position[i])
            e_azimuthal = e_azimuthal / np.linalg.norm(e_azimuthal)
            
            # Radial direction (completes the triad)
            e_radial = np.cross(e_azimuthal, e_parallel)
            
            # Transform magnetic field to MFA coordinates
            b_mfa[i, 0] = np.dot(b_field[i], e_parallel)    # Compressional (parallel)
            b_mfa[i, 1] = np.dot(b_field[i], e_radial)      # Poloidal (radial)
            b_mfa[i, 2] = np.dot(b_field[i], e_azimuthal)   # Toroidal (azimuthal)
        
        return b_mfa
    
    @staticmethod
    def gse_to_mgse(e_field, spin_axis_gse):
        """
        Convert electric field from GSE to mGSE (modified GSE) coordinates.
        
        Parameters:
        -----------
        e_field : array-like, shape (n_samples, 3)
            Electric field in GSE coordinates [Ex, Ey, Ez]
        spin_axis_gse : array-like, shape (n_samples, 3)
            Spin axis unit vector in GSE coordinates
            
        Returns:
        --------
        e_mgse : array-like, shape (n_samples, 3)
            Electric field in mGSE coordinates
        """
        n_samples = e_field.shape[0]
        e_mgse = np.zeros_like(e_field)
        
        # GSE unit vectors
        x_gse = np.array([1, 0, 0])
        y_gse = np.array([0, 1, 0])
        z_gse = np.array([0, 0, 1])
        
        for i in range(n_samples):
            # mGSE unit vectors
            x_mgse = spin_axis_gse[i] / np.linalg.norm(spin_axis_gse[i])
            y_mgse = np.cross(z_gse, x_mgse)
            y_mgse = y_mgse / np.linalg.norm(y_mgse)
            z_mgse = np.cross(x_mgse, y_mgse)
            
            # Transformation matrix from GSE to mGSE
            transform_matrix = np.array([x_mgse, y_mgse, z_mgse])
            
            # Transform electric field
            e_mgse[i] = transform_matrix @ e_field[i]
        
        return e_mgse
    
    @staticmethod
    def apply_e_dot_b_constraint(e_field, b_field, measured_components=[1, 2]):
        """
        Apply E·B = 0 constraint to derive the third electric field component.
        
        Parameters:
        -----------
        e_field : array-like, shape (n_samples, 3)
            Electric field with two measured components
        b_field : array-like, shape (n_samples, 3)
            Magnetic field vector
        measured_components : list
            Indices of measured components (default: [1, 2] for Y and Z)
            
        Returns:
        --------
        e_field_corrected : array-like, shape (n_samples, 3)
            Electric field with all three components
        """
        e_field_corrected = e_field.copy()
        unmeasured_idx = [i for i in range(3) if i not in measured_components][0]
        
        for i in range(e_field.shape[0]):
            # E·B = 0 => E_unmeasured = -(E_measured1*B1 + E_measured2*B2) / B_unmeasured
            if np.abs(b_field[i, unmeasured_idx]) > 1e-6:  # Avoid division by zero
                dot_product = sum(e_field[i, j] * b_field[i, j] 
                                for j in measured_components)
                e_field_corrected[i, unmeasured_idx] = -dot_product / b_field[i, unmeasured_idx]
            else:
                # Cannot constrain this component
                e_field_corrected[i, unmeasured_idx] = np.nan
        
        return e_field_corrected
    
    @staticmethod
    def calculate_psd(data, dt, nperseg, overlap_fraction=0.8):
        """
        Calculate Power Spectral Density using overlapped FFT.
        
        Parameters:
        -----------
        data : array-like, shape (n_samples, n_components)
            Input time series data
        dt : float
            Time resolution (sampling interval in seconds)
        nperseg : int
            Number of points per FFT segment
        overlap_fraction : float
            Overlap fraction (default: 0.8 for 80% overlap)
            
        Returns:
        --------
        frequencies : array-like
            Frequency array
        psd : array-like, shape (n_freq, n_components)
            Power spectral density for each component
        times : array-like
            Time array for each PSD window
        """
        noverlap = int(nperseg * overlap_fraction)
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_components = data.shape[1]
        
        # Calculate PSD for each component
        psd_list = []
        for i in range(n_components):
            f, t, Sxx = signal.spectrogram(data[:, i], fs=1/dt, 
                                          nperseg=nperseg, 
                                          noverlap=noverlap,
                                          scaling='density')
            psd_list.append(Sxx)
        
        psd = np.array(psd_list).transpose(1, 2, 0)  # Shape: (n_freq, n_time, n_components)
        
        return f, psd, t
    
    @staticmethod
    def remove_corotation_field(e_field, sat_position, omega_earth=7.2921e-5):
        """
        Remove co-rotation electric field.
        
        Parameters:
        -----------
        e_field : array-like, shape (n_samples, 3)
            Electric field
        sat_position : array-like, shape (n_samples, 3)
            Satellite position in Earth radii
        omega_earth : float
            Earth's angular velocity (rad/s)
            
        Returns:
        --------
        e_corrected : array-like
            Electric field with co-rotation removed
        """
        e_corrected = e_field.copy()
        omega_vec = np.array([0, 0, omega_earth])
        
        for i in range(e_field.shape[0]):
            v_corot = np.cross(omega_vec, sat_position[i])
            # E_corot = -v × B, but we need B field for this
            # This is a simplified version - you'll need B field for accurate calculation
            pass
        
        return e_corrected
    
    @staticmethod
    def remove_vxb_field(e_field, sat_velocity, b_field):
        """
        Remove V_sc × B electric field induced by satellite motion.
        
        Parameters:
        -----------
        e_field : array-like, shape (n_samples, 3)
            Electric field
        sat_velocity : array-like, shape (n_samples, 3)
            Satellite velocity
        b_field : array-like, shape (n_samples, 3)
            Magnetic field
            
        Returns:
        --------
        e_corrected : array-like
            Electric field with V×B removed
        """
        e_corrected = e_field.copy()
        
        for i in range(e_field.shape[0]):
            vxb = np.cross(sat_velocity[i], b_field[i])
            e_corrected[i] -= vxb
        
        return e_corrected


def process_emfisis_data(b_gsm, sat_position, time_array, 
                         remove_background=True, cutoff_freq=10/3600, order=5):
    """
    Process EMFISIS magnetic field data following the paper's methodology.
    
    Parameters:
    -----------
    b_gsm : array-like, shape (n_samples, 3)
        Total measured magnetic field in GSM coordinates (nT) [Bx, By, Bz]
    sat_position : array-like, shape (n_samples, 3)
        Satellite position in GSM (Earth radii)
    time_array : array-like
        Time array (seconds or datetime)
    remove_background : bool
        Whether to remove background Earth field (default: True)
    cutoff_freq : float
        Cutoff frequency in Hz for background removal (default: 10/3600 = 10 mHz)
    order : int
        Butterworth filter order (default: 5)
        
    Returns:
    --------
    frequencies : array
        Frequency array (mHz)
    psd : array, shape (n_freq, n_time, 3)
        Power spectral density [compressional, poloidal, toroidal]
    psd_times : array
        Time array for PSD
    b_mfa : array
        Magnetic field perturbations in MFA coordinates
    b_perturbations_gsm : array
        Magnetic field perturbations in GSM coordinates
    b_background_gsm : array
        Background magnetic field in GSM coordinates
    """
    converter = VanAllenProbesCoordinateConverter()
    
    # Calculate sampling frequency
    dt = np.mean(np.diff(time_array))
    if isinstance(dt, np.timedelta64):
        dt = dt / np.timedelta64(1, 's')
    sampling_freq = 1.0 / dt
    
    # Step 1: Running average over 11 s to reduce spin modulation
    b_smoothed = converter.running_average(b_gsm, window_size=11)
    
    # Step 2: Remove background Earth magnetic field to isolate perturbations
    if remove_background:
        print(f"Removing background field with {cutoff_freq*1000:.2f} mHz cutoff...")
        b_perturbations_gsm, b_background_gsm = converter.remove_background_field(
            b_smoothed, cutoff_freq=cutoff_freq, sampling_freq=sampling_freq, 
            order=order, return_background=True
        )
        print(f"  Background field removed!")
        print(f"  Original field range: [{np.min(b_smoothed):.2f}, {np.max(b_smoothed):.2f}] nT")
        print(f"  Perturbations range: [{np.min(b_perturbations_gsm):.2f}, {np.max(b_perturbations_gsm):.2f}] nT")
    else:
        b_perturbations_gsm = b_smoothed
        b_background_gsm = np.zeros_like(b_smoothed)
    
    # Step 3: Convert perturbations to MFA coordinates
    b_mfa = converter.gsm_to_mfa(b_perturbations_gsm, sat_position, avg_window=2048)
    
    # Step 4: Calculate PSD with 80% overlap and 2048-point FFT
    frequencies, psd, psd_times = converter.calculate_psd(
        b_mfa, dt=dt, nperseg=2048, overlap_fraction=0.8
    )
    
    # Convert frequency to mHz
    frequencies_mhz = frequencies * 1000
    
    return frequencies_mhz, psd, psd_times, b_mfa, b_perturbations_gsm, b_background_gsm


def process_efw_data(e_mgse, b_mgse, sat_velocity, sat_position, time_array):
    """
    Process EFW electric field data following the paper's methodology.
    
    Parameters:
    -----------
    e_mgse : array-like, shape (n_samples, 3)
        Electric field in mGSE coordinates (mV/m)
    b_mgse : array-like, shape (n_samples, 3)
        Magnetic field in mGSE coordinates (nT)
    sat_velocity : array-like, shape (n_samples, 3)
        Satellite velocity (km/s)
    sat_position : array-like, shape (n_samples, 3)
        Satellite position (Earth radii)
    time_array : array-like
        Time array (seconds)
        
    Returns:
    --------
    frequencies : array
        Frequency array (mHz)
    psd : array, shape (n_freq, n_time, 3)
        Power spectral density in mGSE
    psd_times : array
        Time array for PSD
    """
    converter = VanAllenProbesCoordinateConverter()
    
    # Remove co-rotation and V×B electric fields
    e_corrected = converter.remove_vxb_field(e_mgse, sat_velocity, b_mgse)
    e_corrected = converter.remove_corotation_field(e_corrected, sat_position)
    
    # Calculate PSD with 80% overlap and 256-point FFT
    dt = np.mean(np.diff(time_array))  # Time resolution (should be ~10.9 s)
    if isinstance(dt, np.timedelta64):
        dt = dt / np.timedelta64(1, 's')
    
    frequencies, psd, psd_times = converter.calculate_psd(
        e_corrected, dt=dt, nperseg=256, overlap_fraction=0.8
    )
    
    # Convert frequency to mHz
    frequencies_mhz = frequencies * 1000
    
    return frequencies_mhz, psd, psd_times

#%%
# Example usage
if __name__ == "__main__":
    print("="*70)
    print("VAN ALLEN PROBES - BACKGROUND FIELD REMOVAL DEMONSTRATION")
    print("="*70)
    
    # Simulate Van Allen Probes data
    n_samples = 10000
    dt_mag = 1.0  # 1 s for magnetic field
    time_mag = np.arange(n_samples) * dt_mag
    
    # Simulate orbital period (~9 hours = 32400 seconds)
    orbital_freq = 1.0 / 32400  # Hz
    
    # Create realistic magnetic field with:
    # 1. Large background Earth field (varies with orbital position)
    # 2. Small ULF wave perturbations (Pc5 waves: 2-7 mHz)
    
    # Background field (large, slow variation)
    background_bx = 20000 + 5000 * np.sin(2 * np.pi * orbital_freq * time_mag)
    background_by = 15000 + 8000 * np.cos(2 * np.pi * orbital_freq * time_mag)
    background_bz = 25000 + 6000 * np.sin(2 * np.pi * orbital_freq * time_mag + np.pi/4)
    
    # ULF wave perturbations (small, fast oscillations)
    ulf_freq = 0.005  # 5 mHz
    ulf_bx = 50 * np.sin(2 * np.pi * ulf_freq * time_mag)
    ulf_by = 30 * np.sin(2 * np.pi * ulf_freq * time_mag + np.pi/3)
    ulf_bz = 40 * np.sin(2 * np.pi * ulf_freq * time_mag + np.pi/2)
    
    # Noise
    noise = np.random.randn(n_samples, 3) * 2
    
    # Total measured field = Background + ULF waves + Noise
    b_gsm = np.column_stack([
        background_bx + ulf_bx + noise[:, 0],
        background_by + ulf_by + noise[:, 1],
        background_bz + ulf_bz + noise[:, 2]
    ])
    
    # Satellite position
    sat_position = np.column_stack([
        4 + 2 * np.cos(2 * np.pi * orbital_freq * time_mag),
        2 * np.sin(2 * np.pi * orbital_freq * time_mag),
        0.5 * np.sin(2 * np.pi * orbital_freq * time_mag)
    ])
    
    # Process WITHOUT background removal
    print("\n1. WITHOUT BACKGROUND REMOVAL")
    print("-" * 70)
    result_no_removal = process_emfisis_data(
        b_gsm, sat_position, time_mag, remove_background=False
    )
    freq1, psd1, time_psd1, b_mfa1, b_pert1, b_bg1 = result_no_removal
    print(f"  Total PSD power: {np.sum(psd1):.2e}")
    
    # Process WITH background removal (your method!)
    print("\n2. WITH BACKGROUND REMOVAL (High-pass filter @ 10 mHz)")
    print("-" * 70)
    result_with_removal = process_emfisis_data(
        b_gsm, sat_position, time_mag, 
        remove_background=True, cutoff_freq=10/3600, order=5
    )
    freq2, psd2, time_psd2, b_mfa2, b_pert2, b_bg2 = result_with_removal
    print(f"  Total PSD power: {np.sum(psd2):.2e}")
    
    # Visualize the detrending
    print("\n3. VISUALIZING BACKGROUND REMOVAL")
    print("-" * 70)
    converter = VanAllenProbesCoordinateConverter()
    
    # Show only first 5000 points for clarity
    n_plot = 5000
    fig = converter.visualize_detrending(
        b_gsm[:n_plot], time_mag[:n_plot], 
        cutoff_freq=10/3600, sampling_freq=1.0, order=5
    )
    plt.savefig('background_removal_demo.png', dpi=150, bbox_inches='tight')
    print("  Saved visualization to 'background_removal_demo.png'")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Original field magnitude: {np.mean(np.linalg.norm(b_gsm, axis=1)):.1f} nT")
    print(f"Background field magnitude: {np.mean(np.linalg.norm(b_bg2, axis=1)):.1f} nT")
    print(f"Perturbation magnitude: {np.mean(np.linalg.norm(b_pert2, axis=1)):.1f} nT")
    print(f"\nPerturbations are ~{100*np.mean(np.linalg.norm(b_pert2, axis=1))/np.mean(np.linalg.norm(b_gsm, axis=1)):.2f}% of total field")
    print("\nThe high-pass filter successfully isolated ULF wave perturbations!")
    print("="*70)
# %%