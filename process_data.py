#%%
import os
import sys


import speasy as spz
from speasy.core.inventory import *
from convert_coordinates2 import *
from convert_coordinates_EMF import *
from convert_coordinates_GPU import *
from background_field import *
from PyEMD import EMD
import matplotlib.pyplot as plt

def resample_to_cadence(time_array, data_array, cadence_seconds=11):
    """
    Resample data to a specific cadence.
    
    Parameters:
    -----------
    time_array : array of datetime64
        Original time array
    data_array : array, shape (n_samples, n_components)
        Data to resample (can be 1D or 2D)
    cadence_seconds : int
        Target cadence in seconds
    
    Returns:
    --------
    time_resampled : array
        Resampled time array
    data_resampled : array
        Resampled data
    """
    # Create DataFrame with time index
    df = pd.DataFrame(data_array, index=pd.DatetimeIndex(time_array))
    
    # Resample to target cadence (using mean for averaging)
    df_resampled = df.resample(f'{cadence_seconds}s').mean()
    
    # Remove any NaN rows (if resampling created gaps)
    df_resampled = df_resampled.dropna()
    
    time_resampled = df_resampled.index.to_numpy()
    data_resampled = df_resampled.values
    
    return time_resampled, data_resampled
# %%
cda_tree = spz.inventories.tree.cda

time_interval = [f"2016-10-12T00:00",f"2016-10-16T00:00"]

productsEMFISIS = [
            cda_tree.Van_Allen_Probes_RBSP.RBSPA.EMFISIS.RBSP_A_MAGNETOMETER_1SEC_GSM_EMFISIS_L3.Mag,
            cda_tree.Van_Allen_Probes_RBSP.RBSPA.EMFISIS.RBSP_A_MAGNETOMETER_1SEC_GSM_EMFISIS_L3.coordinates,
        ]
#%%

emfisis = spz.get_data(productsEMFISIS, time_interval)
# %%
FILL_EMFISIS = -99999
FILL_EFW = -9.999999e+30
FILL_REPT = -9.999999e+30
intervals = [time_interval]
for i in range(len(emfisis)):
    emfisis[i].values[emfisis[i].values<=FILL_EMFISIS] = np.nan
# %%
time_mag = emfisis[0].time
b_gsm = emfisis[0]['Bx_GSM', 'By_GSM', 'Bz_GSM'].values
sat_position = emfisis[1]['X_GSM', 'Y_GSM', 'Z_GSM'].values
time_mag_11s, b_gsm_11s = resample_to_cadence(time_mag, b_gsm, cadence_seconds=11)
time_pos_11s, sat_position_11s = resample_to_cadence(time_mag, sat_position, cadence_seconds=11)
# %%
# freq_mag, psd_mag, time_psd_mag, b_mfa = process_emfisis_data(b_gsm, sat_position, time_mag)
# # %%
# result_with_removal = process_emfisis_data(
#         b_gsm, sat_position, time_mag, 
#         remove_background=True, cutoff_freq=10/3600, order=5
#     )
# # %%
# freq2, psd2, time_psd2, b_mfa2, b_pert2, b_bg2 = result_with_removal
# # %%
# converter = VanAllenProbesCoordinateConverter()
# # %%
# n_plot = 5000
# fig = converter.visualize_detrending(
#     b_gsm[:], time_mag[:]
# )
# # %%
# frequencies, psd, psd_times, b_mfa, b_mean = process_emfisis_data_emd(
#     b_gsm, sat_position, time_mag, Tmax=2048, use_emd=True, n_jobs=-1
# )
# %%

# GPU-accelerated processing
B0, dB = calculate_background_field(b_gsm_11s, Tmax=2048, n_jobs=-1)

# %%
