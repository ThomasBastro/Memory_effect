import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import matplotlib.lines as mlines
from scipy.interpolate import interp1d
from matplotlib.colors import LogNorm
from tqdm.auto import tqdm

# Vos imports modules
import utils_grb_aft_new as u_grb 
import utils_phenom as up         
import LISA as li                 # Pour la sensibilité LISA

# --- CONSTANTES ET CHARGEMENT ---
R_REF_PC = 40e6 

def get_detector_sensitivity(detector="et"):
    """Retourne la sensibilité du détecteur choisi."""
    if detector.lower() == "et":
        # Charge ET-D
        data = np.loadtxt("ET_D_sensi.txt")
        return data[:, 0], data[:, 1]
    elif detector.lower() == "lisa":
        # Utilise la classe LISA fournie dans vos fichiers
        lisa = li.LISA()
        f = np.logspace(-5, 0, 2000)
        sn = np.sqrt(f * np.abs(lisa.Sn(f)))
        return f, sn
    else:
        raise ValueError("Détecteur non supporté. Choisissez 'et' ou 'lisa'.")

def load_kilonova_data(file_path="memory_fft_32deg_interp.txt"):
    kilo_freq = np.loadtxt(file_path, skiprows=1)
    return {"f": kilo_freq[:, 0], "hc": kilo_freq[:, 1]}

# --- CALCUL DU SNR ---

def compute_total_snr(f_common, hc_total, f_noise, h_noise, detector="et"):
    """Calcule le SNR selon la plage de fréquence du détecteur."""
    # Définition des masques de fréquence selon le détecteur
    if detector.lower() == "et":
        f_min, f_max = 1.0, 1e4
    else: # lisa
        f_min, f_max = 1e-5, 1e0
        
    mask = (f_common >= f_min) & (f_common <= f_max)
    if not np.any(mask): return 0.0
    
    interp_hn = interp1d(f_noise, h_noise, bounds_error=False, fill_value=np.inf)
    h_n_interp = interp_hn(f_common[mask])
    
    integrand = (hc_total[mask]**2) / (h_n_interp**2)
    return np.sqrt(np.trapz(integrand, x=np.log(f_common[mask])))

# --- FONCTION DE CALCUL DES COMPOSANTES ---

def compute_hc_4_components(f_common, params, kn_data):
    hc_sq = np.zeros_like(f_common)
    r_pc = params["r"]
    scale = R_REF_PC / r_pc
    results = {}

    # 1. GRB + Afterglow (Nouveau modèle)
    t_max = max(params.get('t_jet_break', 86400.0) * 1.2, 1e5)
    t_domain = np.linspace(0, t_max, 5000)
    _, d_h_grb, d_h_aft = u_grb.grb_afterglow_model(
        t_domain, params['E_grb'], params['T_90'], params['E_aft'], 
        params['beta'], params.get('t_dec', 50.0), 
        params.get('t_jet_break', 86400.0), r_pc, params['theta'], params['theta_j']
    )
    h_ft, _, _ = u_grb.fft(f_common, d_h_grb, params['T_90'], d_h_aft, 
                                 params.get('t_dec', 50.0), params.get('t_jet_break', 86400.0))
    hc_grb = 2.0 * f_common * np.abs(h_ft)
    # Nettoyage immédiat
    results['grb'] = np.nan_to_num(hc_grb, nan=0.0, posinf=0.0, neginf=0.0)
    # 2. Ejecta Dynamique
    dh_d = up.delta_h_ejecta(params["M_ej_dyn"], params["v_ej_dyn"], r_pc)
    f_d, fft_d = up.fft_exp_model(dh_d, params.get("tau_dyn", 1e-3))
    
    interp_dyn = interp1d(f_d, 2*f_d*np.abs(fft_d), fill_value=0, bounds_error=False)(f_common)
    # Nettoyage après interpolation
    results['dyn'] = np.nan_to_num(interp_dyn, nan=0.0, posinf=0.0, neginf=0.0)
    # 3. Vent (Wind)
    dh_w = up.delta_h_ejecta(params["M_ej_wind"], params["v_ej_wind"], r_pc)
    f_w, fft_w = up.fft_exp_model(dh_w, params.get("tau_wind", 10.0))
    
    interp_wind = interp1d(f_w, 2*f_w*np.abs(fft_w), fill_value=0, bounds_error=False)(f_common)
    # Nettoyage après interpolation
    results['wind'] = np.nan_to_num(interp_wind, nan=0.0, posinf=0.0, neginf=0.0)
    # 4. Kilonova
    interp_kn = interp1d(kn_data["f"], kn_data["hc"] * scale, fill_value=0, bounds_error=False)(f_common)
    # Nettoyage après interpolation
    results['kn'] = np.nan_to_num(interp_kn, nan=0.0, posinf=0.0, neginf=0.0)
    return results

# --- VISUALISATION GRID AVEC CHOIX DU DÉTECTEUR ---

def plot_hc_grid_4_components(params, name_event, detector="et", outdir="hc_plots_kn", ):
    """
    Génère la grille de subplots et affiche le SNR calculé.
    detector: 'et' ou 'lisa'
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    os.makedirs(outdir, exist_ok=True)
    
    # Configuration spécifique au détecteur
    f_noise, h_noise = get_detector_sensitivity(detector)
    kn_data = load_kilonova_data()
    
    if detector == "et":
        f_common = np.logspace(-0.5, 4, 1500)
        x_lims, y_lims = (1, 3e3), (1e-30, 1e-19)
    else: # lisa
        f_common = np.logspace(-5, 0, 1500)
        x_lims, y_lims = (1e-5, 1), (1e-30, 1e-18)
    
    # Calculs des composantes
    comps_hc = compute_hc_4_components(f_common, params, kn_data)
    
    # Construction du DataFrame et calcul du Total
    rows = []
    hc_total_sq = np.zeros_like(f_common)
    for name, hc_vals in comps_hc.items():
        hc_total_sq += hc_vals**2
        for f, hc in zip(f_common, hc_vals):
            rows.append({'f': f, 'hc': hc, 'component': name})
    
    hc_total = np.sqrt(hc_total_sq)
    for f, hc in zip(f_common, hc_total):
        rows.append({'f': f, 'hc': hc, 'component': 'total'})
        
    df = pd.DataFrame(rows)

    # --- CALCUL ET PRINT DU SNR ---
    total_snr = compute_total_snr(f_common, hc_total, f_noise, h_noise, detector)
    print(f"\n>>> SNR Total calculé pour {detector.upper()} : {total_snr:.4f}")

    # Plotting
    comp_list = ['dyn', 'wind', 'grb', 'kn', 'total']
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    
    for i, name in enumerate(comp_list):
        ax = axes_flat[i]
        # Background
        sns.lineplot(data=df[df['component'] != 'total'], x='f', y='hc', hue='component', 
                     ax=ax, palette=['.7']*4, legend=False, lw=1, alpha=0.3, zorder=1)
        # Main
        main_color = 'k' if name == 'total' else '#8A2BE2'
        sns.lineplot(data=df[df['component'] == name], x='f', y='hc', ax=ax, color=main_color, lw=3, zorder=5)
        # Noise
        ax.plot(f_noise, h_noise, color='crimson', ls=':', lw=2, zorder=10, label=detector.upper())
        
        title = "TOTAL SUM" if name == 'total' else name.upper()
        ax.text(0.95, 0.92, title, transform=ax.transAxes, fontweight='bold', fontsize=18, ha='right')
        ax.set(xscale='log', yscale='log', xlim=x_lims, ylim=y_lims)
        
        if i >= 2: ax.set_xlabel('f [Hz]', fontsize=16)
        if i % 3 == 0: ax.set_ylabel(r'$h_c$', fontsize=16)

    axes_flat[-1].set_visible(False)
    fig.suptitle(f"Memory Breakdown - {name_event} - Detector: {detector.upper()} | Total SNR: {total_snr:.5g}", fontsize=22)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(outdir, f"hc_grid_{name_event}_{detector}.png"), dpi=300)
    plt.show()

# --- EXEMPLE D'UTILISATION ---

if __name__ == "__main__":
    p_study = {
        "E_grb": 3e46, "E_aft": 5e52, "r": 40e6, "T_90": 2.0,
        "theta": np.deg2rad(32), "theta_j": np.deg2rad(5), "beta": 0.99,
        "M_ej_dyn": 0.005, "v_ej_dyn": 0.25, 
        "M_ej_wind": 0.01, "v_ej_wind": 0.05
    }
    p_new ={
        "E_grb": 7.6e51, "E_aft": 5e52, "r": 350e6, "T_90": 51.37,
        "theta": np.deg2rad(30), "theta_j": np.deg2rad(5), "beta": 0.99,
        "M_ej_dyn": 0.02, "v_ej_dyn": 0.3, 
        "M_ej_wind": 0.01, "v_ej_wind": 0.1
    }
    
    # Choix : "et" ou "lisa"
    plot_hc_grid_4_components(p_study, name_event = 'GW 170817', detector="et")
    plot_hc_grid_4_components(p_study, name_event = 'GW 170817', detector="lisa")
    
    plot_hc_grid_4_components(p_new, name_event = 'GW_GRB 211211A', detector="et")
    plot_hc_grid_4_components(p_new, name_event = 'GW_GRB 211211A', detector="lisa")