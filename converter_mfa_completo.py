#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import speasy as spz
from scipy import signal
#%%
class SpacePhysicsCoordinateConverter:
    @staticmethod
    def butter_highpass_filter(data, cutoff_freq, sampling_freq, order=5):
        nyquist = 0.5 * sampling_freq
        normal_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        if data.ndim == 1:
            return signal.filtfilt(b, a, data)
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])
        return filtered_data

    @staticmethod
    def remove_vxb_field(e_field, sat_velocity, b_field):
        # Fator 1e-3 converte (km/s) * nT para mV/m diretamente
        vxb = np.cross(sat_velocity, b_field) * 1e-3
        return e_field - vxb

    def gsm_to_mfa_matrices(self, b_background, sat_position):
        n_samples = b_background.shape[0]
        rot_matrices = np.zeros((n_samples, 3, 3))
        for i in range(n_samples):
            mag_b0 = np.linalg.norm(b_background[i])
            e_parallel = b_background[i] / mag_b0 if mag_b0 > 0 else np.array([0, 0, 1])
            
            e_azimuthal = np.cross(e_parallel, sat_position[i])
            mag_az = np.linalg.norm(e_azimuthal)
            e_azimuthal = e_azimuthal / mag_az if mag_az > 0 else np.array([0, 1, 0])
            
            e_radial = np.cross(e_azimuthal, e_parallel)
            rot_matrices[i] = np.vstack([e_parallel, e_radial, e_azimuthal])
        return rot_matrices

    def transform_vector_to_mfa(self, vector, rot_matrices):
        n_samples = vector.shape[0]
        vector_mfa = np.zeros_like(vector)
        for i in range(n_samples):
            if np.isnan(vector[i]).any():
                vector_mfa[i] = np.nan
            else:
                vector_mfa[i] = rot_matrices[i] @ vector[i]
        return vector_mfa

def resample_to_cadence(time_array, data_array, cadence_seconds=11):
    """Resampla os dados para uma cadência comum usando o Pandas."""
    df = pd.DataFrame(data_array, index=pd.DatetimeIndex(time_array))
    df_resampled = df.resample(f'{cadence_seconds}s').mean()
    # Interpola pequenos gaps gerados pelo resample se houver nans pontuais
    df_resampled = df_resampled.interpolate(method='linear', limit=2)
    return df_resampled.index.to_numpy(), df_resampled.values

#%%
# =============================================================================
# FLUXO PRINCIPAL: DOWNLOAD VIA SPEASY E PROCESSAMENTO
# =============================================================================
if __name__ == "__main__":
    cda_tree = spz.inventories.tree.cda

    # Definindo o intervalo de tempo (Exemplo de Outubro de 2016)
    time_interval = ["2016-10-13T00:00:00", "2016-10-14T00:00:00"]
    print(f"Baixando dados para o intervalo: {time_interval}")

    # Produtos do CDAWeb via Speasy (RBSP-A)
    products = [
        cda_tree.Van_Allen_Probes_RBSP.RBSPA.EMFISIS.RBSP_A_MAGNETOMETER_1SEC_GSM_EMFISIS_L3.Mag,          # B (GSM)
        cda_tree.Van_Allen_Probes_RBSP.RBSPA.EMFISIS.RBSP_A_MAGNETOMETER_1SEC_GSM_EMFISIS_L3.coordinates,  # Posição Satélite (GSM)
        cda_tree.Van_Allen_Probes_RBSP.RBSPA.EFW.RBSP_A_EFW_L3_E_SPINFIT_MGSE.E_spinfit_mgse,             # E (mGSE)
        cda_tree.Van_Allen_Probes_RBSP.RBSPA.EFW.RBSP_A_EFW_L3_E_SPINFIT_MGSE.v_gsm,                      # Velocidade Satélite (GSM)
    ]

    # Download de dados
    data = spz.get_data(products, time_interval)
    
    # Tratamento de Fill Values / Nulos
    FILL_EMFISIS = -99999.0
    FILL_EFW = -9.999999e+30
    
    data[0].values[data[0].values <= FILL_EMFISIS] = np.nan
    data[2].values[data[2].values <= FILL_EFW] = np.nan

    print("Dados baixados com sucesso. Ajustando cadência para 11s...")

    # Extração de variáveis originais
    time_mag = data[0].time
    b_gsm_raw = data[0]['Bx_GSM', 'By_GSM', 'Bz_GSM'].values
    sat_pos_raw = data[1]['X_GSM', 'Y_GSM', 'Z_GSM'].values
    
    time_efw = data[2].time
    e_mgse_raw = data[2]['E1', 'E2', 'E3'].values  # Nota: e_field spinfit costuma vir em mGSE
    sat_vel_raw = data[3]['Vx_GSM', 'Vy_GSM', 'Vz_GSM'].values

    # Resampling para unificar na cadência comum de 11s
    t_11s, b_gsm = resample_to_cadence(time_mag, b_gsm_raw, 11)
    _, sat_pos = resample_to_cadence(time_mag, sat_pos_raw, 11)
    _, e_gsm = resample_to_cadence(time_efw, e_mgse_raw, 11) # Assumindo aproximação direta para o demo
    _, sat_vel = resample_to_cadence(time_efw, sat_vel_raw, 11)
    #%%
    # -------------------------------------------------------------------------
    # PIPELINE DE CONVERSÃO COORDINATES
    # -------------------------------------------------------------------------
    converter = SpacePhysicsCoordinateConverter()
    fs_11s = 1.0 / 11.0 # Frequência de amostragem real (~0.09 Hz)

    # 1. Limpeza do Campo Elétrico (E_corrected = E - V_sc x B)
    e_corrected_gsm = converter.remove_vxb_field(e_gsm, sat_vel, b_gsm)

    # 2. Filtro Passa-Alta para isolar flutuações ULF (Banda Pc5, Cutoff = 10 mHz)
    cutoff = 10 / 3600  # 10 mHz convertido para Hz
    db_gsm = converter.butter_highpass_filter(b_gsm, cutoff, sampling_freq=fs_11s)
    b0_gsm = b_gsm - db_gsm # Campo médio de fundo

    de_gsm = converter.butter_highpass_filter(e_corrected_gsm, cutoff, sampling_freq=fs_11s)

    # 3. Geração das Matrizes e rotação para MFA
    rot_matrices = converter.gsm_to_mfa_matrices(b0_gsm, sat_pos)
    b_mfa = converter.transform_vector_to_mfa(db_gsm, rot_matrices)
    e_mfa = converter.transform_vector_to_mfa(de_gsm, rot_matrices)

    print("Conversão Concluída! Gerando gráficos de validação...")
    #%%
    # -------------------------------------------------------------------------
    # PLOT
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    
    axes[0].plot(t_11s, b_mfa[:, 0], label=r'$\delta B_\parallel$ (Compressional)', color='black')
    axes[0].plot(t_11s, b_mfa[:, 1], label=r'$\delta B_r$ (Poloidal)', color='crimson')
    axes[0].plot(t_11s, b_mfa[:, 2], label=r'$\delta B_\phi$ (Toroidal)', color='royalblue')
    axes[0].set_ylabel("$\delta$B - MFA (nT)")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Dados RBSP-A Convertidos para MFA (Filtro Passa-Alta de 10 mHz)")

    axes[1].plot(t_11s, e_mfa[:, 0], label=r'$\delta E_\parallel$', color='black', linestyle='--')
    axes[1].plot(t_11s, e_mfa[:, 1], label=r'$\delta E_r$', color='crimson')
    axes[1].plot(t_11s, e_mfa[:, 2], label=r'$\delta E_\phi$', color='royalblue')
    axes[1].set_ylabel("$\delta$E - MFA (mV/m)")
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()