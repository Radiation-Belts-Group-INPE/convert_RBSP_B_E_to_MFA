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
        """Aplica um filtro digital passa-alta para isolar perturbações ULF."""
        nyquist = 0.5 * sampling_freq
        normal_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        if data.ndim == 1:
            return signal.filtfilt(b, a, data)
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])
        return filtered_data

    def gsm_to_mfa_matrices(self, b_background, sat_position):
        """Calcula as matrizes de rotação ponto a ponto de GSM para MFA."""
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
        """Projeta um vetor 3D qualquer no referencial MFA usando as matrizes."""
        n_samples = vector.shape[0]
        vector_mfa = np.zeros_like(vector)
        for i in range(n_samples):
            if np.isnan(vector[i]).any():
                vector_mfa[i] = np.nan
            else:
                vector_mfa[i] = rot_matrices[i] @ vector[i]
        return vector_mfa

def resample_to_cadence(time_array, data_array, cadence_seconds=11):
    """Resampla os dados para uma cadência comum tratando NaNs corretamente."""
    df = pd.DataFrame(data_array, index=pd.DatetimeIndex(time_array))
    
    # O .mean() do Pandas já ignora NaNs por padrão (skipna=True),
    # calculando a média apenas com os pontos válidos daquela janela.
    df_resampled = df.resample(f'{cadence_seconds}s').mean()
    
    # limit_direction='both' garante que se houver NaNs no início ou fim, 
    # eles sejam extrapolados/preenchidos mais próximos e não estraguem o sinal.
    df_resampled = df_resampled.interpolate(method='linear', limit_direction='both')
    
    # Caso ainda sobre algum NaN residual difícil na borda, preenchemos com o vizinho mais próximo
    df_resampled = df_resampled.bfill().ffill()
    
    return df_resampled.index.to_numpy(), df_resampled.values
#%%

# =============================================================================
# FLUXO PRINCIPAL: DOWNLOAD VIA SPEASY E PROCESSAMENTO
# =============================================================================
if __name__ == "__main__":
    cda_tree = spz.inventories.tree.cda

    # Intervalo de tempo curto para teste rápido (4 horas)
    time_interval = ["2016-10-13T00:00:00", "2016-10-13T04:00:00"]
    print(f"Baixando dados das Van Allen Probes para: {time_interval}")

    # Lista de produtos atualizada para usar os dados L3 e a árvore correta do seu ambiente
    products = [
        # 0. Campo Magnético L3 (1 segundo, em GSM)
        cda_tree.Van_Allen_Probes_RBSP.RBSPA.EMFISIS.RBSP_A_MAGNETOMETER_1SEC_GSM_EMFISIS_L3.Mag,

        # 1. Coordenadas de Posição do satélite em GSM
        cda_tree.Van_Allen_Probes_RBSP.RBSPA.EMFISIS.RBSP_A_MAGNETOMETER_1SEC_GSM_EMFISIS_L3.coordinates,

        # 2. Campo Elétrico L3 (Inercial / calibrado / restrição E.B=0 aplicada) em mGSE
        cda_tree.Van_Allen_Probes_RBSP.RBSP_A.EFW.RBSPA_EFW_L3.efield_in_inertial_frame_spinfit_edotb_mgse
    ]

    # Download dos dados via API
    data = spz.get_data(products, time_interval)
    
    # Tratamento de Fill Values / Dados Nulos padrão dos arquivos CDF
    FILL_EMFISIS = -99999.0
    FILL_EFW = -9.999999e+30
    
    data[0].values[data[0].values <= FILL_EMFISIS] = np.nan
    data[2].values[data[2].values <= FILL_EFW] = np.nan

    print("Dados baixados com sucesso. Ajustando cadência para 11s...")

    # Extração das matrizes de dados brutas (.values evita problemas com nomes de colunas)
    time_mag = data[0].time
    b_gsm_raw = data[0].values
    sat_pos_raw = data[1].values
    
    time_efw = data[2].time
    e_mgse_raw = data[2].values

    # Resampling para sincronizar ambos os instrumentos na mesma linha do tempo (11s)
    t_11s, b_gsm = resample_to_cadence(time_mag, b_gsm_raw, 11)
    _, sat_pos = resample_to_cadence(time_mag, sat_pos_raw, 11)
    _, e_mgse = resample_to_cadence(time_efw, e_mgse_raw, 11) 
#%%
    # -------------------------------------------------------------------------
    # PIPELINE DE PROCESSAMENTO E CONVERSÃO COORDINATES
    # -------------------------------------------------------------------------
    converter = SpacePhysicsCoordinateConverter()
    fs_11s = 1.0 / 11.0 # Frequência de amostragem após o resample (~0.09 Hz)

    # Nota: Não precisamos do passo "e_mgse - vxb" porque escolhemos o produto 'in_inertial_frame'

    # 1. Isolamento das Ondas ULF (Filtro Passa-Alta Pc5: Cutoff em 10 mHz)
    cutoff_hz = 10 / 3600  # 10 mHz convertido para Hz
    
    db_gsm = converter.butter_highpass_filter(b_gsm, cutoff_hz, sampling_freq=fs_11s)
    b0_gsm = b_gsm - db_gsm # Campo médio de fundo usado para o alinhamento
    
    de_mgse = converter.butter_highpass_filter(e_mgse, cutoff_hz, sampling_freq=fs_11s)

    # 2. Geração das Matrizes de Rotação baseadas na geometria do campo de fundo B0
    rot_matrices = converter.gsm_to_mfa_matrices(b0_gsm, sat_pos)
    
    # 3. Rotação das perturbações de ambos os vetores para o referencial MFA
    b_mfa = converter.transform_vector_to_mfa(db_gsm, rot_matrices)
    e_mfa = converter.transform_vector_to_mfa(de_mgse, rot_matrices)

    print("Conversão concluída com sucesso! Gerando gráficos...")
#%%
    # -------------------------------------------------------------------------
    # GERAÇÃO DOS GRÁFICOS DE VALIDAÇÃO
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    
    # Painel do Campo Magnético Alinhado (MFA)
    axes[0].plot(t_11s, b_mfa[:, 0], label=r'$\delta B_\parallel$ (Compressional)', color='black', linewidth=1)
    axes[0].plot(t_11s, b_mfa[:, 1], label=r'$\delta B_r$ (Poloidal)', color='crimson', linewidth=1)
    axes[0].plot(t_11s, b_mfa[:, 2], label=r'$\delta B_\phi$ (Toroidal)', color='royalblue', linewidth=1)
    axes[0].set_ylabel(r"$\delta$B - MFA (nT)", fontsize=11)
    axes[0].legend(loc='upper right', frameon=True)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Sinais das Van Allen Probes em Coordenadas MFA (Mean Field-Aligned)", fontsize=12)

    # Painel do Campo Elétrico Alinhado (MFA)
    axes[1].plot(t_11s, e_mfa[:, 0], label=r'$\delta E_\parallel$', color='black', linestyle='--', linewidth=1)
    axes[1].plot(t_11s, e_mfa[:, 1], label=r'$\delta E_r$', color='crimson', linewidth=1)
    axes[1].plot(t_11s, e_mfa[:, 2], label=r'$\delta E_\phi$', color='royalblue', linewidth=1)
    axes[1].set_ylabel(r"$\delta$E - MFA (mV/m)", fontsize=11)
    axes[1].set_xlabel("Tempo (UTC)", fontsize=11)
    axes[1].legend(loc='upper right', frameon=True)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# %%
de_mgse
# %%
