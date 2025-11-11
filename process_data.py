#%%
import speasy as spz
from speasy.core.inventory import *
from convert_coordinates2 import *
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
# %%
freq_mag, psd_mag, time_psd_mag, b_mfa = process_emfisis_data(b_gsm, sat_position, time_mag)
# %%
result_with_removal = process_emfisis_data(
        b_gsm, sat_position, time_mag, 
        remove_background=True, cutoff_freq=10/3600, order=5
    )
# %%
freq2, psd2, time_psd2, b_mfa2, b_pert2, b_bg2 = result_with_removal
# %%
converter = VanAllenProbesCoordinateConverter()
# %%
n_plot = 5000
fig = converter.visualize_detrending(
    b_gsm[:], time_mag[:]
)
# %%
