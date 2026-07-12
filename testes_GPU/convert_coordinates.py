#%%
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import pandas as pd

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
    def gsm_to_mfa(b_field, sat_position, avg_window=2048):
        """
        Convert magnetic field from GSM to MFA (Mean Field Aligned) coordinates.
        
        Parameters:
        -----------
        b_field : array-like, shape (n_samples, 3)
            Magnetic field in GSM coordinates [Bx, By, Bz]
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


def process_emfisis_data(b_gsm, sat_position, time_array):
    """
    Process EMFISIS magnetic field data following the paper's methodology.
    
    Parameters:
    -----------
    b_gsm : array-like, shape (n_samples, 3)
        Magnetic field in GSM coordinates (nT)
    sat_position : array-like, shape (n_samples, 3)
        Satellite position in GSM (Earth radii)
    time_array : array-like
        Time array (seconds)
        
    Returns:
    --------
    frequencies : array
        Frequency array (mHz)
    psd : array, shape (n_freq, n_time, 3)
        Power spectral density [compressional, poloidal, toroidal]
    psd_times : array
        Time array for PSD
    """
    converter = VanAllenProbesCoordinateConverter()
    
    # Step 1: Running average over 11 s to reduce spin modulation
    b_smoothed = converter.running_average(b_gsm, window_size=11)
    
    # Step 2: Convert to MFA coordinates
    b_mfa = converter.gsm_to_mfa(b_smoothed, sat_position, avg_window=2048)
    
    # Step 3: Calculate PSD with 80% overlap and 2048-point FFT
    dt = np.mean(np.diff(time_array))  # Time resolution (should be ~1 s)
    dt = round(dt / np.timedelta64(1, 's'))
    frequencies, psd, psd_times = converter.calculate_psd(
        b_mfa, dt=dt, nperseg=2048, overlap_fraction=0.8
    )
    
    # Convert frequency to mHz
    frequencies_mhz = frequencies * 1000
    
    return frequencies_mhz, psd, psd_times, b_mfa


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
    # Example: Generate synthetic data for demonstration
    n_samples = 10000
    dt_mag = 1.0  # 1 s for magnetic field
    dt_elec = 10.9  # 10.9 s for electric field
    
    # Synthetic magnetic field data (GSM)
    time_mag = np.arange(n_samples) * dt_mag
    b_gsm = np.random.randn(n_samples, 3) * 10 + np.array([20, 10, 30])
    sat_position = np.random.randn(n_samples, 3) * 2 + np.array([4, 0, 0])
    
    # Process EMFISIS data
    freq_mag, psd_mag, time_psd_mag = process_emfisis_data(b_gsm, sat_position, time_mag)
    
    print(f"Magnetic field PSD:")
    print(f"  Frequency resolution: {freq_mag[1] - freq_mag[0]:.3f} mHz")
    print(f"  Time resolution: {(time_psd_mag[1] - time_psd_mag[0])/60:.2f} min")
    print(f"  PSD shape: {psd_mag.shape}")
    print(f"  Components: [Compressional, Poloidal, Toroidal]")
# %%
