import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib.patches import Rectangle
import utils_phenom as up
import utils_GRB_afterglow_phenom as ugrb
import os
import tensorflow_probability as tfp
from tqdm.auto import tqdm
import pandas as pd
import time

# Tenter d'importer TensorFlow, sinon passer en mode CPU
try:
    import tensorflow as tf
    print("TensorFlow trouvé. Le mode GPU est disponible.")
    # Limiter l'utilisation de la mémoire GPU si nécessaire
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"{len(gpus)} GPU(s) trouvé(s) et configuré(s).")
        except RuntimeError as e:
            print(e)
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow non installé. Le mode GPU n'est pas disponible.")
    TF_AVAILABLE = False

# ==============================================================================
# Fonctions de chargement et de calcul du signal (inchangées)
# ==============================================================================

def load_etd_sensitivity(file_path="ET_D_sensi.txt"):
    data = np.loadtxt(file_path)
    return data[:, 0], data[:, 1]

def load_memory_inputs():
    time_mem = np.loadtxt("GW170817_memory_time_domain.txt")
    freq_mem = np.loadtxt("GW170817_memory_characteristic_strain.txt")
    kilo_time = np.loadtxt("memory_vs_time_32deg.txt", skiprows=1)
    kilo_freq = np.loadtxt("memory_fft_32deg_interp.txt", skiprows=1)
    nonlinear = {"t": time_mem[:, 0], "h": time_mem[:, 1], "f": freq_mem[:, 0], "hc": freq_mem[:, 1]}
    kilonova = {"t": kilo_time[:, 0] * 24 * 3600, "h": kilo_time[:, 1], "f": kilo_freq[:, 0], "hc": kilo_freq[:, 1]}
    return nonlinear, kilonova

def build_memory_components():
    nonlinear, kilonova = load_memory_inputs()
    r_ref = 40e6
    components = {"nonlinear": nonlinear, "kilonova": kilonova}
    return components, r_ref

def compute_snr_etd(f_signal, hc_signal, f_noise, h_noise):
    f_signal, hc_signal = np.asarray(f_signal), np.asarray(hc_signal)
    mask = (f_signal >= 1.0) & (f_signal <= 1e4)
    f_masked, hc_masked = f_signal[mask], hc_signal[mask]
    if len(f_masked) < 2: return 0.0
    interp_hn = interp1d(f_noise, h_noise, kind="linear", bounds_error=False, fill_value=np.inf)
    h_n_interp = interp_hn(f_masked)
    integrand = (hc_masked**2) / (h_n_interp**2)
    return np.sqrt(np.trapz(integrand, x=np.log(f_masked)))

def scale_with_distance(hc, r_ref_pc, r_pc):
    return hc * (r_ref_pc / r_pc)

def compute_total_hc(f_common, params_dyn, params_wind, params_grb, memory_components, r_ref_pc=40e6, add_dyn=True, add_wind=True, add_grb=True, add_nonlinear=True, add_kilonova=True):
    hc_total = np.zeros_like(f_common)
    r_pc = params_grb.get("r", r_ref_pc)

    with np.errstate(over='ignore', invalid='ignore'):
        if add_dyn:
            delta_h = up.delta_h_ejecta(params_dyn["M_ej"], params_dyn["v_ej"], params_dyn["r"])
            f, fft = up.fft_exp_model(delta_h, params_dyn["tau"])
            hc = 2 * f * np.abs(fft)
            hc_total += interp1d(f, hc, bounds_error=False, fill_value=0.0)(f_common)
        if add_wind:
            delta_h = up.delta_h_ejecta(params_wind["M_ej"], params_wind["v_ej"], params_wind["r"])
            f, fft = up.fft_exp_model(delta_h, params_wind["tau"])
            hc = 2 * f * np.abs(fft)
            hc_total += interp1d(f, hc, bounds_error=False, fill_value=0.0)(f_common)
        if add_grb:
            h_in = ugrb.memory_initial_acceleration_GRB(params_grb["E_grb"], params_grb["theta"], params_grb["phi_ej"], params_grb["r"], params_grb["beta"])
            h_aft = ugrb.memory_afterglow(params_grb["E_aft"], params_grb["theta"], params_grb["phi_ej"], params_grb["r"], params_grb["beta"])
            f, fft = ugrb.total_waveform_fft(h_in, h_aft, params_grb["end_grb"], params_grb["theta"], params_grb["r"], radius=0)
            hc = 2 * f * np.abs(fft)
            hc_total += interp1d(f, hc, bounds_error=False, fill_value=0.0)(f_common)

    if add_nonlinear:
        hc_nl = scale_with_distance(memory_components["nonlinear"]["hc"], r_ref_pc, r_pc)
        hc_total += interp1d(memory_components["nonlinear"]["f"], hc_nl, bounds_error=False, fill_value=0.0)(f_common)
    if add_kilonova:
        hc_kn = scale_with_distance(memory_components["kilonova"]["hc"], r_ref_pc, r_pc)
        hc_total += interp1d(memory_components["kilonova"]["f"], hc_kn, bounds_error=False, fill_value=0.0)(f_common)
    return np.nan_to_num(hc_total)

# ==============================================================================
# Fonctions de formatage pour les plots (inchangées)
# ==============================================================================

def _axis_values_and_label(key, values):
    values = np.asarray(values)
    if key == "r": return values / 1e6, r"$d\,[{\rm Mpc}]$"
    if key in ["theta", "theta_j", "phi_ej"]: return np.rad2deg(values), rf"${_format_key_for_title(key)}\,[^\circ]$"
    if key in ["E_grb", "E_aft"]: return values, rf"${key.replace('_', '_{')}}}\,[{{\rm erg}}]$"
    if key == "M_ej_dyn": return values, r"$M_{\rm ej,dyn}\,[M_\odot]$"
    if key == "M_ej_wind": return values, r"$M_{\rm ej,wind}\,[M_\odot]$"
    if key == 'v_ej_dyn': return values, r"$v_{\rm ej,dyn}\,[c]$"
    if key == "v_ej_wind": return values, r"$v_{\rm ej,wind}\,[c]$"
    if key == "beta": return values, r"$\beta$"
    if key == "end_grb": return values, r"$T_{\rm GRB} [{\rm s}]$"
    return values, key

def _format_key_for_title(key):
    return {"E_grb": r"$E_{\rm GRB}$", "E_aft": r"$E_{\rm aft}$", "theta": r"$\theta_{\rm ej}$", "theta_j": r"$\theta_j$", "phi_ej": r"$\phi_{\rm ej}$", "r": r"$d$", "M_ej_dyn": r"$M_{\rm ej,dyn}$", "M_ej_wind": r"$M_{\rm ej,wind}$", "v_ej_dyn": r"$v_{\rm ej,dyn}$", "v_ej_wind": r"$v_{\rm ej,wind}$", 'end_grb': r"$T_{\rm GRB}$", "beta": r"$\beta$"}.get(key, key)

def _format_value_for_title(key, value):
    v = float(np.asarray(value).squeeze())
    if key in ["theta", "theta_j", "phi_ej"]: return rf"${np.rad2deg(v):.1f}^\circ$"
    if key == "r": return rf"${v / 1e6:.0f}\,\mathrm{{Mpc}}$"
    if key in ["M_ej_dyn", "M_ej_wind"]: return rf"${v:.0e}\,M_\odot$"
    if key in ["v_ej_dyn", "v_ej_wind"]: return rf"${v:.2f}\,c$"
    if key in ["E_grb", "E_aft"]: return rf"${v:.0e}\,\mathrm{{erg}}$"
    if key == "beta": return rf"{v:.2f}"
    if key == "end_grb": return rf"{v:.1f} s"
    return f"{v:.3g}"

def _plot_gw170817_marker(ax, x_key, y_key):
    gw170817 = {"M_ej_dyn": 0.5e-2, "v_ej_dyn": 0.25, "M_ej_wind": 0.01, "v_ej_wind": 0.05, "r": 40e6, "E_grb": 3e46, "E_aft": 1e52, "theta": np.deg2rad(32), "theta_j": np.deg2rad(5), "beta": 0.99, "end_grb": 2.0, "phi_ej": 0.0}
    if x_key not in gw170817 or y_key not in gw170817: return
    x_ref, _ = _axis_values_and_label(x_key, [gw170817[x_key]])
    y_ref, _ = _axis_values_and_label(y_key, [gw170817[y_key]])
    ax.scatter(x_ref, y_ref, marker="*", s=260, c="limegreen", edgecolors="black", linewidths=1.2, zorder=8, label="GW170817")

# ==============================================================================
# SECTION CPU : Calcul de grille et tracé
# ==============================================================================

def calculate_snr_grid(param_dict, fixed_params_base, **kwargs):
    scan_keys = [k for k, v in param_dict.items() if isinstance(v, (list, np.ndarray))]
    if len(scan_keys) != 2: raise ValueError("param_dict doit contenir exactement deux listes/arrays pour le scan.")
    x_key, y_key = scan_keys
    x_values, y_values = np.array(param_dict[x_key]), np.array(param_dict[y_key])
    
    snr_grid = np.zeros((len(y_values), len(x_values)))
    
    memory_components, r_ref_pc = build_memory_components()
    f_noise, h_noise = load_etd_sensitivity()
    
    params_dyn_base = {"M_ej": 0.5e-2, "v_ej": 0.25, "tau": 1e-3, "r": r_ref_pc, "start": -1e-3}
    params_wind_base = {"M_ej": 0.01, "v_ej": 0.05, "tau": 10.0, "r": r_ref_pc, "start": 0.01}
    params_grb_base = {"E_grb": 3e46, "theta": np.deg2rad(32), "phi_ej": 0.0, "r": r_ref_pc, "beta": 0.99, "end_grb": 2.0, "E_aft": 10**(52.2), "theta_j": np.deg2rad(5)}
    
    all_params_base = [params_dyn_base, params_wind_base, params_grb_base]
    current_fixed_params = {k: v for k, v in fixed_params_base.items() if k not in scan_keys}
    for k, v in current_fixed_params.items():
        for p in all_params_base:
            if k in p: p[k] = v
        if k == "r": params_dyn_base["r"] = params_wind_base["r"] = params_grb_base["r"] = v

    f_common = np.logspace(-4, 5, 50000)
    
    for i, y_val in enumerate(tqdm(y_values, desc=f"Calcul grille CPU ({y_key})", miniters=1, mininterval=1.0)):
        for j, x_val in enumerate(x_values):
            p_dyn, pw, pg = params_dyn_base.copy(), params_wind_base.copy(), params_grb_base.copy()
            
            for k, v in [(x_key, x_val), (y_key, y_val)]:
                if k == "M_ej_dyn": p_dyn["M_ej"] = v
                elif k == "v_ej_dyn": p_dyn["v_ej"] = v
                elif k == "M_ej_wind": pw["M_ej"] = v
                elif k == "v_ej_wind": pw["v_ej"] = v
                elif k in p_dyn: p_dyn[k] = v
                elif k in pw: pw[k] = v
                elif k in pg: pg[k] = v
                if k == "r": p_dyn["r"] = pw["r"] = pg["r"] = v

            hc_total = compute_total_hc(f_common, p_dyn, pw, pg, memory_components, r_ref_pc, **kwargs)
            snr_grid[i, j] = compute_snr_etd(f_common, hc_total, f_noise, h_noise)
            
    return snr_grid

def run_and_plot_scenario(scenario_name, param_global, param_zoom, fixed_params, plot_kwargs, component_kwargs):
    """Exécute un scénario complet (calcul et tracé) en mode CPU."""
    print(f"\n--- Démarrage du scénario CPU : {scenario_name} ---")
    
    # Calcul des grilles
    print("Calcul de la grille globale...")
    snr_grid_global = calculate_snr_grid(param_global, fixed_params, **component_kwargs)
    print("Calcul de la grille zoomée...")
    snr_grid_zoom = calculate_snr_grid(param_zoom, fixed_params, **component_kwargs)
    
    # Tracé
    plot_combined_heatmaps(
        snr_grid_global, snr_grid_zoom,
        param_global, param_zoom,
        outdir="final_plots",
        fixed_params=fixed_params,
        filename_tag=scenario_name,
        **plot_kwargs
    )
    
    # Sauvegarde des données
    save_grids_to_csv(snr_grid_global, snr_grid_zoom, param_global, param_zoom, "final_plots", scenario_name)
    print(f"--- Scénario CPU : {scenario_name} terminé ---")

# ==============================================================================
# NOUVELLE SECTION GPU : Logique avec TensorFlow
# ==============================================================================

# REMPLACER CETTE FONCTION dans 170817_snr_condor.py

if TF_AVAILABLE:
    # Outil d'interpolation 1D pour TensorFlow (CORRIGÉ)
    def interp1d_tf(x, xp, fp):
        """
        Interpolation linéaire 1D équivalente à np.interp(x, xp, fp) pour TensorFlow.
        x: Tenseur 1D des points où évaluer.
        xp: Tenseur 1D des coordonnées x des données.
        fp: Tenseur 2D [N_grilles, N_points_données] des coordonnées y des données.
        """
        x = tf.cast(x, tf.float32)
        xp = tf.cast(xp, tf.float32)
        fp = tf.cast(fp, tf.float32)

        # Trouver les indices pour chaque point dans x
        # `idx` aura la même forme que `x`
        idx = tf.searchsorted(xp, x, side='right')
        
        # Gérer les cas limites
        idx = tf.clip_by_value(idx, 1, len(xp) - 1)

        # Coordonnées des points encadrant les points d'évaluation
        # Forme: [len(x)]
        xp_before = tf.gather(xp, idx - 1)
        xp_after = tf.gather(xp, idx)

        # Coordonnées y correspondantes.
        # fp est [N_grilles, N_points_données], on gather sur l'axe 1
        # Forme: [N_grilles, len(x)]
        fp_before = tf.gather(fp, idx - 1, axis=1)
        fp_after = tf.gather(fp, idx, axis=1)

        # Calcul de la pente de l'interpolation
        # Pour éviter la division par zéro si xp_after == xp_before
        slope = (fp_after - fp_before) / tf.maximum(xp_after - xp_before, 1e-9)

        # Calcul de l'interpolation
        # x, xp_before sont [len(x)], on les étend pour le broadcasting avec fp_before [N_grilles, len(x)]
        interpolated = fp_before + slope * (x[tf.newaxis, :] - xp_before[tf.newaxis, :])
        
        return interpolated

    def compute_snr_etd_tf(f_signal, hc_signal, f_noise_tf, h_noise_tf):
        # Assurer que les types sont corrects
        f_signal = tf.cast(f_signal, tf.float32)
        hc_signal = tf.cast(hc_signal, tf.float32)
        
        # Masque de fréquence
        mask = (f_signal >= 1.0) & (f_signal <= 1e4)
        f_masked = tf.boolean_mask(f_signal, mask)
        
        # Appliquer le masque à hc_signal (qui a une dimension de grille)
        hc_masked = tf.boolean_mask(hc_signal, mask, axis=1)
        
        # Interpolation du bruit sur les fréquences du signal
        h_n_interp = interp1d_tf(f_masked, f_noise_tf, h_noise_tf[tf.newaxis, :])
        
        # Calcul de l'intégrande
        integrand = (hc_masked**2) / (h_n_interp**2)
        
        # Utilisation de la fonction trapz de tensorflow_probability
        log_f = tf.math.log(f_masked)
        
        # --- CORRECTION ICI ---
        # Le tenseur log_f (x) doit avoir la MÊME forme que integrand (y).
        # Forme de integrand : (10000, 22222)
        # Forme de log_f : (22222,)
        
        # On ajoute une dimension à log_f pour avoir (1, 22222)
        log_f_reshaped = log_f[tf.newaxis, :]
        
        # On répète (tile) ce tenseur pour qu'il corresponde à la première dimension de l'intégrande.
        log_f_tiled = tf.tile(log_f_reshaped, [tf.shape(integrand)[0], 1])
        

        snr_sq = tfp.math.trapz(y=integrand, x=log_f_tiled, axis=1)


        return tf.sqrt(snr_sq)

    def calculate_snr_grid_tf(param_dict, fixed_params_base, **kwargs):
        """Calcule la grille SNR entièrement sur GPU avec TensorFlow."""
        scan_keys = [k for k, v in param_dict.items() if isinstance(v, (list, np.ndarray))]
        x_key, y_key = scan_keys
        x_values, y_values = np.array(param_dict[x_key]), np.array(param_dict[y_key])

        # Créer la grille de paramètres pour le GPU
        X, Y = np.meshgrid(x_values, y_values)
        grid_x = X.flatten()
        grid_y = Y.flatten()
        
        # Charger les données et les préparer pour TF
        memory_components, r_ref_pc = build_memory_components()
        f_noise, h_noise = load_etd_sensitivity()
        f_noise_tf = tf.constant(f_noise, dtype=tf.float32)
        h_noise_tf = tf.constant(h_noise, dtype=tf.float32)
        
        f_common = tf.constant(np.logspace(-4, 5, 50000), dtype=tf.float32)
        hc_total = tf.zeros([len(grid_x), len(f_common)], dtype=tf.float32)

        # --- Calcul des composantes ---
        # Cette partie reste complexe à vectoriser entièrement à cause des modèles physiques différents.
        # On itère sur la grille pour construire hc_total, mais les calculs internes sont en TF.
        # Une optimisation future pourrait vectoriser aussi cette boucle.
        
        # Pour cet exemple, nous allons utiliser une approche hybride :
        # On calcule hc pour chaque point de la grille, puis on calcule tous les SNR en parallèle.
        # C'est moins efficace que tout vectoriser, mais beaucoup plus simple à implémenter.
        
        all_hc = []
        total_points = len(grid_x)
        for i in tqdm(range(total_points), desc="Pré-calcul GPU des hc", miniters=max(1, total_points // 100), mininterval=1.0):
            x_val, y_val = grid_x[i], grid_y[i]
            
            # Logique de paramètre (identique à la version CPU)
            params_dyn = {"M_ej": 0.5e-2, "v_ej": 0.25, "tau": 1e-3, "r": r_ref_pc, "start": -1e-3}
            params_wind = {"M_ej": 0.01, "v_ej": 0.05, "tau": 10.0, "r": r_ref_pc, "start": 0.01}
            params_grb = {"E_grb": 3e46, "theta": np.deg2rad(32), "phi_ej": 0.0, "r": r_ref_pc, "beta": 0.99, "end_grb": 2.0, "E_aft": 10**(52.2), "theta_j": np.deg2rad(5)}
            
            # Appliquer les paramètres fixes
            all_p = [params_dyn, params_wind, params_grb]
            for k, v in fixed_params_base.items():
                if k not in scan_keys: # Ne pas écraser les variables de la grille
                    for p in all_p:
                        if k in p: p[k] = v
                    if k == "r": 
                        params_dyn["r"] = v
                        params_wind["r"] = v
                        params_grb["r"] = v

            # Appliquer les paramètres de la grille (x_val, y_val)
            grid_params = {x_key: x_val, y_key: y_val}
            for k, v in grid_params.items():
                if k == "M_ej_dyn": params_dyn["M_ej"] = v
                elif k == "v_ej_dyn": params_dyn["v_ej"] = v
                elif k == "M_ej_wind": params_wind["M_ej"] = v
                elif k == "v_ej_wind": params_wind["v_ej"] = v
                elif k in params_grb: params_grb[k] = v
                # Gérer le cas où 'r' est une variable de la grille
                if k == "r":
                    params_dyn["r"] = v
                    params_wind["r"] = v
                    params_grb["r"] = v

            # Calcul de hc pour ce point
            hc = compute_total_hc(f_common.numpy(), params_dyn, params_wind, params_grb, memory_components, r_ref_pc, **kwargs)
            all_hc.append(hc)
            
        hc_total_tf = tf.constant(np.array(all_hc), dtype=tf.float32)
        
        # Calcul de tous les SNR en une seule passe sur le GPU
        print("Calcul des SNR sur GPU...")
        snr_flat = compute_snr_etd_tf(f_common, hc_total_tf, f_noise_tf, h_noise_tf)
        
        # Remettre en forme de grille
        snr_grid = tf.reshape(snr_flat, (len(y_values), len(x_values))).numpy()
        
        return snr_grid

    def run_and_plot_scenario_tf(scenario_name, param_global, param_zoom, fixed_params, plot_kwargs, component_kwargs):
        """Exécute un scénario complet (calcul et tracé) en mode GPU."""
        if not TF_AVAILABLE:
            print(f"TensorFlow n'est pas disponible, passage au mode CPU pour le scénario {scenario_name}.")
            run_and_plot_scenario(scenario_name, param_global, param_zoom, fixed_params, plot_kwargs, component_kwargs)
            return
            
        print(f"\n--- Démarrage du scénario GPU : {scenario_name} ---")
        
        # Calcul des grilles
        print("Calcul de la grille globale (GPU)...")
        snr_grid_global = calculate_snr_grid_tf(param_global, fixed_params, **component_kwargs)
        print("Calcul de la grille zoomée (GPU)...")
        snr_grid_zoom = calculate_snr_grid_tf(param_zoom, fixed_params, **component_kwargs)
        
        # Tracé
        plot_combined_heatmaps(
            snr_grid_global, snr_grid_zoom,
            param_global, param_zoom,
            outdir="final_plots",
            fixed_params=fixed_params,
            filename_tag=scenario_name,
            **plot_kwargs
        )
        
        # Sauvegarde des données
        save_grids_to_csv(snr_grid_global, snr_grid_zoom, param_global, param_zoom, "final_plots", scenario_name)
        print(f"--- Scénario GPU : {scenario_name} terminé ---")

# ==============================================================================
# Fonctions communes de tracé et de sauvegarde
# ==============================================================================

def plot_combined_heatmaps(snr_grid_global, snr_grid_zoom, param_dict_global, param_dict_zoom, outdir, fixed_params={}, **plot_kwargs):
    os.makedirs(outdir, exist_ok=True)
    
    scan_keys = list(param_dict_global.keys())
    x_key, y_key = scan_keys[0], scan_keys[1]
    
    x_values_global, y_values_global = param_dict_global[x_key], param_dict_global[y_key]
    x_values_zoom, y_values_zoom = param_dict_zoom[x_key], param_dict_zoom[y_key]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    
    x_plot_global, x_label = _axis_values_and_label(x_key, x_values_global)
    y_plot_global, y_label = _axis_values_and_label(y_key, y_values_global)
    x_plot_zoom, _ = _axis_values_and_label(x_key, x_values_zoom)
    y_plot_zoom, _ = _axis_values_and_label(y_key, y_values_zoom)

    cmap1 = ListedColormap(plt.get_cmap("twilight")(np.linspace(0, 0.5, 128)))
    cmap2 = ListedColormap(plt.get_cmap("twilight_shifted")(np.linspace(0.5, 1, 128)))

    valid_global = np.isfinite(snr_grid_global) & (snr_grid_global > 0)
    norm1 = LogNorm(vmin=np.min(snr_grid_global[valid_global]) if np.any(valid_global) else 1e-2, vmax=np.max(snr_grid_global[valid_global]) if np.any(valid_global) else 1)
    valid_zoom = np.isfinite(snr_grid_zoom) & (snr_grid_zoom > 0)
    norm2 = LogNorm(vmin=np.min(snr_grid_zoom[valid_zoom]) if np.any(valid_zoom) else 1e-2, vmax=np.max(snr_grid_zoom[valid_zoom]) if np.any(valid_zoom) else 1)

    X1, Y1 = np.meshgrid(x_plot_global, y_plot_global)
    im1 = ax1.pcolormesh(X1, Y1, snr_grid_global, shading='gouraud', cmap=cmap1, norm=norm1)
    fig.colorbar(im1, ax=ax1, label="SNR (Global)", extend='both')
    ax1.set(xscale=plot_kwargs.get('xscale', 'log'), yscale=plot_kwargs.get('yscale', 'log'), xlabel=x_label, ylabel=y_label, title="Vue Globale")
    
    contours1 = ax1.contour(X1, Y1, snr_grid_global, levels=plot_kwargs.get('levels', (1,10,100)), colors="k", linewidths=1.2, linestyles="--")
    ax1.clabel(contours1, inline=True, fontsize=10, fmt="%d")

    zoom_x_min, _ = _axis_values_and_label(x_key, [x_values_zoom[0]])
    zoom_x_max, _ = _axis_values_and_label(x_key, [x_values_zoom[-1]])
    zoom_y_min, _ = _axis_values_and_label(y_key, [y_values_zoom[0]])
    zoom_y_max, _ = _axis_values_and_label(y_key, [y_values_zoom[-1]])
    rect = Rectangle((zoom_x_min[0], zoom_y_min[0]), zoom_x_max[0] - zoom_x_min[0], zoom_y_max[0] - zoom_y_min[0], linewidth=1.5, edgecolor='black', facecolor='none', zorder=10)
    ax1.add_patch(rect)
    if plot_kwargs.get('if_170817_marker_all', False): _plot_gw170817_marker(ax1, x_key, y_key)

    X2, Y2 = np.meshgrid(x_plot_zoom, y_plot_zoom)
    im2 = ax2.pcolormesh(X2, Y2, snr_grid_zoom, shading='gouraud', cmap=cmap2, norm=norm2)
    fig.colorbar(im2, ax=ax2, label="SNR (Zoom)", extend='both')
    ax2.set(xscale=plot_kwargs.get('xscale', 'log'), yscale=plot_kwargs.get('yscale', 'log'), xlabel=x_label, ylabel=y_label, title="Vue Zoomée")
    contours2 = ax2.contour(X2, Y2, snr_grid_zoom, levels=plot_kwargs.get('levels', (1,10,100)), colors="k", linewidths=1.2, linestyles="--")
    ax2.clabel(contours2, inline=True, fontsize=10, fmt="%d")
    if plot_kwargs.get('if_170817_marker_zoom', False): _plot_gw170817_marker(ax2, x_key, y_key)

    all_params = {**param_dict_global, **param_dict_zoom, **fixed_params}
    fixed_for_title = {k: v for k, v in all_params.items() if not isinstance(v, (list, np.ndarray))}
    title_parts = [f"{_format_key_for_title(k)}={_format_value_for_title(k, v)}" for k, v in fixed_for_title.items()]
    n = (len(title_parts) + 1) // 2
    title_str = " | ".join(title_parts[:n]) + ("\n" + " | ".join(title_parts[n:]) if len(title_parts) > n else "")
    fig.suptitle(title_str, fontsize=12)

    base_filename = f"SNR_heatmap_combined_{plot_kwargs.get('filename_tag')}"
    plt.savefig(os.path.join(outdir, f"{base_filename}.png"), dpi=300)
    plt.close()
    print(f"Graphique combiné sauvegardé dans {os.path.join(outdir, f'{base_filename}.png')}")

def save_grids_to_csv(snr_grid_global, snr_grid_zoom, param_global, param_zoom, outdir, scenario_name):
    os.makedirs(outdir, exist_ok=True)
    x_key_g, y_key_g = list(param_global.keys())
    df_global = pd.DataFrame(snr_grid_global, index=param_global[y_key_g], columns=param_global[x_key_g])
    df_global.to_csv(os.path.join(outdir, f"snr_grid_global_{scenario_name}.csv"))

    x_key_z, y_key_z = list(param_zoom.keys())
    df_zoom = pd.DataFrame(snr_grid_zoom, index=param_zoom[y_key_z], columns=param_zoom[x_key_z])
    df_zoom.to_csv(os.path.join(outdir, f"snr_grid_zoom_{scenario_name}.csv"))
    print(f"Grilles de données pour '{scenario_name}' sauvegardées en CSV.")

# ==============================================================================
# POINT D'ENTRÉE PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    # Définition des scénarios de calcul
    SCENARIOS = {
        "Egrb_vs_dist": {
            "param_global": {"E_grb": np.logspace(45, 60, 100), "r": np.logspace(6, 9, 100)},
            "param_zoom": {"E_grb": np.logspace(52, 53, 50), "r": np.logspace(7, 8, 50)},
            "fixed_params": {},
            "plot_kwargs": {"xscale": "log", "yscale": "log", "if_170817_marker_all": True, "if_170817_marker_zoom": False},
            "component_kwargs": {"add_dyn": False, "add_wind": False, "add_nonlinear": False, "add_kilonova": False}
        },
        "dyn_memory": {
            "param_global": {"M_ej_dyn": np.logspace(-5.1, -1, 100), "v_ej_dyn": np.linspace(0.01, 0.99, 100)},
            "param_zoom": {"M_ej_dyn": np.logspace(-4.1, -2, 50), "v_ej_dyn": np.linspace(0.09, 0.4, 50)},
            "fixed_params": {},
            "plot_kwargs": {"xscale": "log", "yscale": "linear", "if_170817_marker_all": True, "if_170817_marker_zoom": True},
            "component_kwargs": {"add_grb": False, "add_wind": False, "add_nonlinear": False, "add_kilonova": False}
        }
    }

    # --- Exécution ---
    # Choisissez le mode d'exécution : 'cpu' ou 'gpu'
    # Le mode 'gpu' ne fonctionnera que si TensorFlow est installé et un GPU est détecté.
    EXECUTION_MODE = 'gpu' 

    start_time = time.time()
    
    # Boucle sur tous les scénarios définis
    for name, config in SCENARIOS.items():
        if EXECUTION_MODE == 'gpu':
            run_and_plot_scenario_tf(
                scenario_name=name,
                param_global=config["param_global"],
                param_zoom=config["param_zoom"],
                fixed_params=config["fixed_params"],
                plot_kwargs=config["plot_kwargs"],
                component_kwargs=config["component_kwargs"]
            )
        else: # Mode CPU par défaut
             run_and_plot_scenario(
                scenario_name=name,
                param_global=config["param_global"],
                param_zoom=config["param_zoom"],
                fixed_params=config["fixed_params"],
                plot_kwargs=config["plot_kwargs"],
                component_kwargs=config["component_kwargs"]
            )

    end_time = time.time()
    print(f"\nTemps d'exécution total : {end_time - start_time:.2f} secondes.")