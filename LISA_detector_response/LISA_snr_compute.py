import numpy as np
from astropy import units as un
from scipy.interpolate import InterpolatedUnivariateSpline
from ldc.waveform.waveform import HpHc
from ldc.lisa.orbits import Orbits
from ldc.lisa.projection.projectedstrain import ProjectedStrain
from ldc.lisa.noise import get_noise_model
# Nouveaux imports pour la méthode ldc
from ldc.common.series.timeseries import TimeSeries, FrequencySeries
from ldc.common.tools.snr import compute_tdi_snr
import utils_GRB_afterglow_phenom as ugrb
import matplotlib.pyplot as plt
import os

# ===================================================================
# 1. CLASSE D'INTERFACE (Inchangée)
# ===================================================================
class NumericWaveform(HpHc):
    """
    Classe d'interface pour permettre au module ldc de traiter un signal 
    temporel fourni sous forme de tableaux NumPy.
    """
    def set_param(self, param, units=None):
        self.p = param
        for key, value in param.items():
            setattr(self, key, value)
        if 'basis' in self.p:
            self.basis = self.p['basis']
        self.check_param()

    def check_param(self):
        needed = ['t', 'hp', 'hc']
        if not all(k in self.p for k in needed):
            raise ValueError(f"Paramètres manquants. {needed} sont requis.")
        self.hp_interp = InterpolatedUnivariateSpline(self.p['t'], self.p['hp'], ext='zeros')
        self.hc_interp = InterpolatedUnivariateSpline(self.p['t'], self.p['hc'], ext='zeros')

    def compute_hphc_td(self, t, **kwargs):
        hp_t = self.hp_interp(t)
        hc_t = self.hc_interp(t)
        return hp_t, hc_t

# ===================================================================
# 2. FONCTION PRINCIPALE DE CALCUL DU SNR 
# ===================================================================
def calculate_lisa_snr(t, hp, hc, k, u, v, plot_results=False):
    """
    Calcule le Signal-to-Noise Ratio (SNR) pour un signal d'onde gravitationnelle
    donné (h(t)) dans le détecteur LISA.
    """
    print("Initialisation du calcul du SNR pour LISA...")

    # --- Étape 1: Configuration de LISA ---
    dt = t[1] - t[0]
    config = {
        "orbit_type": "analytic", 
        "nominal_arm_length": 2.5e9 * un.m,
        "initial_rotation": 0 * un.rad,
        "initial_position": 0 * un.rad
    }
    orbits = Orbits.type(config)
    proj = ProjectedStrain(orbits)
    print("Configuration LISA terminée.")

    # --- Étape 2: Préparation de la source GW ---
    gw_source = NumericWaveform("custom_signal", "Numeric", "TD")
    params = {'t': t, 'hp': hp, 'hc': hc, 'd_L': 1.0 * un.Gpc, 'basis': (k, v, u)}
    gw_source.set_param(params)
    print("Source GW préparée.")

    # --- Étape 3: Calcul de la réponse des bras et des variables TDI ---
    print("Calcul de la réponse des bras de LISA...")
    proj.arm_response(t_min=t[0], t_max=t[-1], dt=dt, GWs=[gw_source]) 
    print("Calcul des variables TDI...")
    tdi_X = proj.compute_tdi_x(t)
    tdi_Y = proj.compute_tdi_y(t)
    tdi_Z = proj.compute_tdi_z(t)
    print("Variables TDI calculées.")

    # --- Étape 4: Conversion au format FrequencySeries de LDC ---
    print("Conversion des signaux au format LDC FrequencySeries...")
    X_fs = TimeSeries(tdi_X, t0=t[0], dt=dt).ts.fft()
    Y_fs = TimeSeries(tdi_Y, t0=t[0], dt=dt).ts.fft()
    Z_fs = TimeSeries(tdi_Z, t0=t[0], dt=dt).ts.fft()
    source_tdi = {"X": X_fs, "Y": Y_fs, "Z": Z_fs}
    
    # --- Étape 5: Obtention du modèle de bruit LDC ---
    freq_dim_name = source_tdi["X"].dims[0]
    freqs = source_tdi["X"].coords[freq_dim_name].values
    
    # On demande explicitement les PSD croisées (XY) nécessaires pour le calcul du SNR total
    noise_model = get_noise_model("SciRDv1", freqs)
    # On utilise SciRDv1 qui inclut les PSD croisées, ce qui est nécessaire pour un calcul de SNR total correct.
    # à tester avec d'autres modèles de bruit si besoin : model can be: “Proposal”, “SciRDv1”, “SciRDdeg1”, “MRDv1”,”MRD_MFR”, “mldc”, “newdrs”, “LCESAcall”, “redbook”
    print("Modèle de bruit obtenu.")

    # --- Étape 6: Calcul du SNR avec la fonction de LDC ---
    print("Calcul du SNR via ldc.common.tools.snr.compute_tdi_snr...")
    
    # La fonction gère elle-même fmin, pas besoin de le spécifier.
    # Elle retourne un dictionnaire avec les SNR au carré.
    results_dict = compute_tdi_snr(source=source_tdi, noise=noise_model)
    print("Summary dict returned by compute_tdi_snr:")
    print(results_dict)
    if not results_dict:
        print("ERREUR: Le calcul du SNR n'a retourné aucun résultat.")
        return np.nan

    # On extrait les SNR au carré et on prend la racine
    snr2_X = results_dict.get('X2', 0.0)
    snr2_Y = results_dict.get('Y2', 0.0)
    snr2_Z = results_dict.get('Z2', 0.0)
    snr2_total = results_dict.get('tot2', 0.0)
    snr_total = np.sqrt(snr2_total)
    print("Calcul du SNR terminé.")
    outdir = "results"
    # verif si le dossier existe, sinon le créer
    import os
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if plot_results:
        print("Génération des graphiques...")

        # --- Préparation des données pour le plot ---
        valid_freq_mask = freqs > 0
        freqs_valid = freqs[valid_freq_mask]
        
        # Calcul de la densité spectrale d'amplitude (ASD) du bruit pour les fréquences valides
        asd_noise_X = np.sqrt(noise_model.psd(option='X', freq=freqs_valid))
        
        # Sélection du signal TDI X pour les fréquences valides
        X_fs_valid = X_fs.isel({freq_dim_name: valid_freq_mask})

        # --- Création de la figure ---
        fig, ax2 = plt.subplots(1, 1, figsize=(10, 8))
        

        # --- Graphe 2: Densité Spectrale d'Amplitude ---
        ax2.loglog(freqs_valid, np.abs(X_fs_valid), label="Signal TDI (X)", color='dodgerblue', linewidth=2)
        ax2.loglog(freqs_valid, asd_noise_X, label="LISA noise (SciRDv1)", color='grey')
        ax2.set_xlabel("f [Hz]")
        ax2.set_ylabel("ASD [1/√Hz]")
        ax2.legend(frameon=False, fontsize = 13, loc='upper left')


        # --- Finalisation ---
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "LISA_signal_vs_noise.png"), dpi=150)
        plt.show()

    print("\n--- RÉSULTATS ---")
    print(f"SNR² (X) = {snr2_X:.5f}")
    print(f"SNR² (Y) = {snr2_Y:.5f}")
    print(f"SNR² (Z) = {snr2_Z:.5f}")
    print(f"SNR Total = {snr_total:.5f}")
    
    return snr_total


# ===================================================================
# 3. EXEMPLE D'UTILISATION 
# ===================================================================
if __name__ == '__main__':
    # ... (Cette section reste exactement la même) ...
    # On importe le module contenant les fonctions de génération de signal GRB
    import utils_GRB_afterglow_phenom as ugrb

    print("--- DÉBUT DE L'EXEMPLE AVEC UN SIGNAL DE GRB ---")
    
    # --- 1. Définition des paramètres physiques pour le GRB ---
    # Ces valeurs sont des exemples typiques que vous pouvez modifier
    Ej_erg = 1e52            # Énergie cinétique du jet [erg]
    E_aft = 1e52           # Puissance d'injection de l'afterglow [erg/s]
    t_end = 2    # Durée de l'injection [s]
    distance_pc = 50 * 1e6     # Distance de la source [pc] (ex: 50 Mpc)
    
    # Angles (en radians)
    theta_ej_rad = np.deg2rad(10) # Angle de vue (ex: 10 degrés)
    phi_ej_rad = np.deg2rad(0)    # Angle azimutal 


    # --- 2. Calcul des amplitudes des composantes du signal ---
    h_in = ugrb.memory_initial_acceleration_GRB(Ej=Ej_erg, 
                                                theta_ej=theta_ej_rad, 
                                                phi_ej=phi_ej_rad, 
                                                d=distance_pc)
    
    h_m = ugrb.memory_afterglow(E_aft, theta_ej_rad, phi_ej_rad, distance_pc)

    print(f"Amplitude initiale h_in: {h_in:.2e}")
    print(f"Amplitude d'afterglow h_m: {h_m:.2e}")

    # --- 3. Génération de la forme d'onde temporelle ---
    # On définit un axe de temps. Il doit être assez long pour couvrir la montée du signal.
    # La durée de la montée (t_m) dépend de t_end_injection. Prenons une marge.
    t_max_s = 1e5  # Durée totale du signal à simuler (en secondes)~ 1j
    # ! si t_max >> snr diminue. Est ce que cela est causé par le projection de LISA (mouvement propre de LISA)
    # normalement, l'impact de t_max devrait être minime tant que le signal est entièrement contenu dans la fenêtre temporelle.
    # alors pk est ce que le snr diminue ? est ce que c'est un artefact du calcul du snr ? ou est ce que c'est un effet physique réel (modulation de LISA) ?
    
    t_obs = np.linspace(0, t_max_s, int(5_001))
    
    # Génération du signal h(t)
    t_signal, h_signal = ugrb.memory_total_waveform(t_obs=t_obs,
                                                    h_in=h_in,
                                                    h_m=0,
                                                    t_end_injection=t_end,
                                                    theta_ej=theta_ej_rad,
                                                    d=distance_pc, 
                                                    radius=0)

    # On assigne le signal à la polarisation 'plus' et on met 'cross' à zéro (symétrie azimutale)
    hp_signal = h_signal
    hc_signal = np.zeros_like(h_signal)
    # direction de propagation = direction theta_ej, phi_ej (la source est dans cette direction, les ondes se propagent vers nous)
    theta_source = theta_ej_rad  # Angle polaire
    phi_source = phi_ej_rad      # Angle azimutal

    k = np.array([
        np.sin(theta_source) * np.cos(phi_source),
        np.sin(theta_source) * np.sin(phi_source),
        np.cos(theta_source)
    ])

    # Définition de la base de polarisation (u, v) perpendiculaire à k
    # On s'assure que la base est toujours bien définie, même aux pôles.
    if np.allclose(k, [0, 0, 1]) or np.allclose(k, [0, 0, -1]):
        # Si la source est à un pôle, on choisit une direction de référence pour u.
        u = np.array([1., 0., 0.])
        v = np.cross(k, u)
    else:
        # Cas général
        z_axis = np.array([0., 0., 1.])
        u = np.cross(z_axis, k)
        u /= np.linalg.norm(u)
        v = np.cross(k, u)

    print(f"\nDirection de la source (k) définie par les angles du jet : {k}")

    # --- 5. Appel de la fonction principale de calcul du SNR ---
    print("------ VERSION AVEC MODULATION ORBITALE ------")
    snr = calculate_lisa_snr(t_signal, hp_signal, hc_signal, k, u, v, plot_results=True)
    print(f"SNR calculé pour le signal de GRB : {snr:.5f}")

    print("--- FIN DE L'EXEMPLE ---")
  