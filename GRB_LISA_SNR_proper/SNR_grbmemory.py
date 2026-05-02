import numpy as np
import matplotlib.pyplot as plt
from astropy import units as un
from scipy.interpolate import InterpolatedUnivariateSpline
from ldc.waveform.waveform import HpHc
from ldc.lisa.orbits import Orbits
from ldc.lisa.projection.projectedstrain import ProjectedStrain
from ldc.lisa.noise import get_noise_model
import utils_signal as usig

# Import du module utils fourni
import utils_grb_aft_new as ugrb

class NumericWaveform(HpHc):
    """Classe pour gérer un signal h(t) fourni sous forme de tableaux NumPy."""
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

def plot_tdi_vs_noise(freqs, tdi_X, tdi_Y, tdi_Z, dt):
    """
    Calcule la FFT des variables TDI et les affiche avec le bruit LISA.
    """
    # FFT des signaux TDI[cite: 2]
    tilde_X = np.fft.rfft(tdi_X) * dt
    tilde_Y = np.fft.rfft(tdi_Y) * dt
    tilde_Z = np.fft.rfft(tdi_Z) * dt
    
    # Modèle de bruit LISA[cite: 2]
    noise_model = get_noise_model("SciRDv1", freqs)
    Sn_X = noise_model.psd(option='X')
    
    plt.figure(figsize=(10, 6))
    plt.loglog(freqs, np.sqrt(Sn_X), label="Bruit LISA (TDI X)", color='black', lw=2)
    plt.loglog(freqs, np.abs(tilde_X), label="|TDI X|", alpha=0.7)
    plt.loglog(freqs, np.abs(tilde_Y), label="|TDI Y|", alpha=0.7)
    plt.loglog(freqs, np.abs(tilde_Z), label="|TDI Z|", alpha=0.7)
    
    plt.title("Signal TDI et Bruit LISA")
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Strain (1/√Hz)")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()

def calculate_lisa_snr_for_grb(E_GRB, T_90, E_aft, beta, t_dec, t_jet_break, d, theta, theta_j, t_max=86450, dt=100):
    """
    Fonction unique qui génère le signal GRB, le projette sur LISA, 
    calcule les TDI et retourne le SNR.
    """
    # 1. Grille temporelle
    t = np.arange(0, t_max, dt)
    
    # 2. Génération du signal temporel via le modèle[cite: 1]
    hp, hp_GRB, hp_aft = ugrb.grb_afterglow_model(t, E_GRB, T_90, E_aft, beta, t_dec, t_jet_break, d, theta, theta_j)
    hc = np.zeros_like(t)
    
    # 3. Configuration de la source numérique
    k, u, v = np.array([0., 0., 1.]), np.array([1., 0., 0.]), np.array([0., 1., 0.])
    gw_source = NumericWaveform("grb_memory", "Numeric", "TD")
    gw_source.set_param({'t': t, 'hp': hp, 'hc': hc, 'd_L': d * un.pc, 'basis': (k, v, u)})
    
    # 4. Configuration des orbites et projection
    config = {"orbit_type": "analytic", "nominal_arm_length": 2.5e9 * un.m, "initial_rotation": 0 * un.rad, "initial_position": 0 * un.rad}
    orbits = Orbits.type(config)
    proj = ProjectedStrain(orbits)
    
    # Projection sur les bras
    yArm = proj.arm_response(0, t_max, dt, GWs=[gw_source])
    
    # 5. Calcul des variables TDI[cite: 2]
    tdi_X = proj.compute_tdi_x(t)
    tdi_Y = proj.compute_tdi_y(t)
    tdi_Z = proj.compute_tdi_z(t)
    
    
    # Pad and tapper the signal for spectral leakage reduction
    tdi_list = [tdi_X, tdi_Y, tdi_Z]
    for tdi in tdi_list:
        t, tdi = usig.pad_and_taper(tdi, t, padding_length = 1000, window_type='tukey', alpha=0.2)
        
    # 6. Domaine fréquentiel et SNR[cite: 2]
    freqs = np.fft.rfftfreq(len(t), d=dt)
    df = freqs[1] - freqs[0]
    
    
    
    
    tilde_X = np.fft.rfft(tdi_X) * dt
    tilde_Y = np.fft.rfft(tdi_Y) * dt
    tilde_Z = np.fft.rfft(tdi_Z) * dt
    
    
    
    noise_model = get_noise_model("SciRDv1", freqs)
    Sn_X = noise_model.psd(option='X')
    Sn_Y = noise_model.psd(option='Y')
    Sn_Z = noise_model.psd(option='Z')
    
    # Calcul SNR^2 = 4 * sum(|h_tilde|^2 / Sn) * df[cite: 2]
    snr2_X = 4 * np.sum(np.abs(tilde_X[1:])**2 / Sn_X[1:]) * df
    snr2_Y = 4 * np.sum(np.abs(tilde_Y[1:])**2 / Sn_Y[1:]) * df
    snr2_Z = 4 * np.sum(np.abs(tilde_Z[1:])**2 / Sn_Z[1:]) * df
    
    snr_total = np.sqrt(snr2_X + snr2_Y + snr2_Z)
    
    # Optionnel : Plot automatique pour vérification
    plot_tdi_vs_noise(freqs, tdi_X, tdi_Y, tdi_Z, dt)
    
    print(f"--- Résultats SNR LISA ---")
    print(f"SNR X: {np.sqrt(snr2_X):.2f}")
    print(f"SNR Y: {np.sqrt(snr2_Y):.2f}")
    print(f"SNR Z: {np.sqrt(snr2_Z):.2f}")
    print(f"SNR Total combiné: {snr_total:.2f}")
    
    return snr_total

if __name__ == "__main__":
    # Paramètres d'exemple pour un GRB
    E_GRB_val, T_90_val = 1e53, 100
    E_aft_val, beta_val = 1e53, 0.99
    t_dec_val, t_jet_break_val = 50, 100000
    d_val, theta_val, theta_j_val = 1e9, 0.1, 0.1

    snr_final = calculate_lisa_snr_for_grb(
        E_GRB_val, T_90_val, E_aft_val, beta_val, 
        t_dec_val, t_jet_break_val, d_val, theta_val, theta_j_val
    )