import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18
import astropy.units as u
import utils_GRB_afterglow_phenom as u_grb
import LISA as li
from scipy.interpolate import interp1d
import os
from matplotlib.colors import LogNorm
import warnings
from tqdm import tqdm
from matplotlib.colors import ListedColormap
import pandas as pd
warnings.filterwarnings("ignore")


def compute_hc_grb_only(params_grb):
    h_in = u_grb.memory_initial_acceleration_GRB(params_grb['E_grb'], params_grb['theta'], params_grb['phi_ej'], params_grb['r'], params_grb['beta'])
    # MODIFICATION: Utilisation de memory_afterglow avec E_aft au lieu de P_in
    h_aft = u_grb.memory_afterglow(params_grb['E_aft'], params_grb['theta'], params_grb['phi_ej'], params_grb['r'], params_grb['beta'])
    # MODIFICATION: Ajout de radius=0 pour être cohérent avec l'absence d'injection
    f_grb, fft_grb_aft = u_grb.total_waveform_fft(h_in, h_aft, params_grb['end_grb'], params_grb['theta'], params_grb['r'], radius=0)
    
    hc_grb = 2.0 * f_grb * np.abs(fft_grb_aft)
    # keep only the frequencies > 1e-4 Hz and < 1e-1 Hz
    mask = (f_grb >= 1e-4) & (f_grb <= 1e-1)
    f_grb = f_grb[mask]
    hc_grb = hc_grb[mask]
    return f_grb, hc_grb


def compute_snr(f_signal, h_c, f_noise, h_n):
    f_signal = np.array(f_signal)
    h_c = np.array(h_c)
    f_noise = np.array(f_noise)
    h_n = np.array(h_n)

    interp_hn = interp1d(f_noise, h_n, kind='linear', bounds_error=False, fill_value=np.inf)
    h_n_interp = interp_hn(f_signal)
    # mask to keep frequencies between 5e-5 and 1e-1 Hz where LISA is sensitive
    mask = (f_signal >= 5e-5) & (f_signal <= 1e-1)
    f_signal = f_signal[mask]
    h_c = h_c[mask]
    h_n_interp = h_n_interp[mask]

    integrand = (h_c**2) / (h_n_interp**2)
    snr_squared = np.trapezoid(integrand, x=np.log(f_signal))
    snr = np.sqrt(snr_squared)
    return snr

def plot_heatmap_snr(param_dict, outdir, xscale="log", yscale="log", grb_parameters_list=None):
    
    x_key, y_key = [k for k, v in param_dict.items() if isinstance(v, (list, np.ndarray))]
    x_values = np.array(param_dict[x_key])
    y_values = np.array(param_dict[y_key])

    param_grb = {
        "E_grb": 1e50, "theta": np.deg2rad(10), "phi_ej": 0.0,
        "r": 1e6, "beta": 0.99, "end_grb": 10, "E_aft": 1e52, "theta_j": np.deg2rad(5)
    }
   
    param_grb.update({k: v for k, v in param_dict.items() if not isinstance(v, (list, np.ndarray))})
    snr_grid = np.zeros((len(y_values), len(x_values)))
    lisa = li.LISA()

    # --- Ajout de la barre de progression TQDM ---
    total_iterations = len(y_values)
    min_iters_for_update = max(1, total_iterations // 100)
    
    progress_bar = tqdm(
        enumerate(y_values), 
        total=total_iterations, 
        desc=f"Calcul Heatmap {y_key} vs {x_key}",
        miniters=min_iters_for_update,
        unit="row"
    )
    
    for i, y in progress_bar: # Utiliser la barre de progression ici

        for j, x in enumerate(x_values):
  
            params = param_grb.copy()
            params[x_key] = x
            params[y_key] = y
            
            # Gérer la conversion de redshift si 'z' est utilisé comme axe
            if x_key == 'z': params['r'] = Planck18.luminosity_distance(x).to_value(u.Mpc)
            if y_key == 'z': params['r'] = Planck18.luminosity_distance(y).to_value(u.Mpc)
            
            try:

                f, hc = compute_hc_grb_only(params)
              
                # Pas besoin d'initialiser LISA dans la boucle
                snr = compute_snr(f, hc, f, np.sqrt(f * np.abs(lisa.Sn(f))))
                snr_grid[i, j] = snr
            except Exception as e:
  
                progress_bar.write(f"Erreur: {e}") # Optionnel pour ne pas polluer la console
                snr_grid[i, j] = np.nan
    
    # Conversion des unités pour les axes
    if x_key == "r":
   
        x_values_plot = x_values 
        x_key_plot = "Distance [Mpc]"
    elif x_key in ["theta", "theta_j", "phi_ej"]:
        x_values_plot = np.rad2deg(x_values)
        x_key_plot = x_key + " [°]"
    else:
        x_values_plot = x_values
        x_key_plot = x_key

    if y_key == "r":
  
        y_values_plot = y_values 
        y_key_plot = "Distance [Mpc]"
    elif y_key in ["theta", "theta_j", "phi_ej"]:
        y_values_plot = np.rad2deg(y_values)
        y_key_plot = y_key + " [°]"
    else:
        y_values_plot = y_values
        y_key_plot = y_key

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    plt.subplots_adjust(top=1.2, bottom=0.1)
    
    valid = np.isfinite(snr_grid)
    pos = valid & (snr_grid > 1e-4) # Seuil pour éviter les valeurs très faibles
    
    snr_grid_plot = snr_grid.copy()
    snr_grid_plot = np.where(pos, snr_grid_plot, np.nan)
    snr_min = np.nanmin(snr_grid_plot)
    snr_max = np.nanmax(snr_grid_plot)

  

    norm = LogNorm(vmin=np.nanmin(snr_grid_plot), vmax=np.nanmax(snr_grid_plot) if np.any(snr_grid_plot) else 1)

    X, Y = np.meshgrid(x_values_plot, y_values_plot)
    
    cmap = ListedColormap(plt.get_cmap('PiYG')(np.linspace(0.5, 1, 128)))
    

    im = ax.pcolormesh(X, Y, snr_grid_plot, shading='gouraud', cmap=cmap, norm=norm)
    
    # Titre avec les paramètres fixes
    fixed_params = {k: v for k, v in param_grb.items() if k not in [x_key, y_key]}
    # remove phi_ej from fixed_params if it's 0.0 to avoid clutter
    if 'phi_ej' in fixed_params and fixed_params['phi_ej'] == 0.0:
        del fixed_params['phi_ej']
    
    param_grb_names_latex = {
        'E_grb': r'$E_{\rm{GRB}}$', 'E_aft': r'$E_{\rm{aft}}$', 'theta': r'$\theta_{\rm ej}$',
        'theta_j': r'$\theta_j$', 'phi_ej': r'$\phi_{\rm{ej}}$', 'beta': r'$\beta$',
        'end_grb': r'$T_{\rm{GRB}}$', 'r': r'$d$'
    }
    param_grb_units = {
        'E_grb': 'erg', 'E_aft': 'erg', 'theta': r'$^\circ$', 'theta_j': r'$^\circ$',
        'phi_ej': r'$^\circ$', 'beta': 'c', 'end_grb': 's', 'r': 'Mpc'
    }
    
    param_strings = []
    for k, v in fixed_params.items():
        name = param_grb_names_latex.get(k, k)
        unit = param_grb_units.get(k, '')
        val_str = ""
        if k in ['theta', 'theta_j', 'phi_ej']:
            val_str = f"{np.rad2deg(v):.1f}"
        elif k == 'r':
            val_str = f"{v/1e6:.1f}"  # Convertir de pc à Mpc pour l'affichage
        elif isinstance(v, float) and (v < 1e-3 or v > 1e3):
            val_str = f"{v:.1e}"
        else:
            val_str = f"{v:.2f}" if isinstance(v, float) else str(v)
        param_strings.append(f"{name}={val_str} {unit}")

    mid = len(param_strings) // 2
    title = " | ".join(param_strings[:mid]) + "\n" + " | ".join(param_strings[mid:])
    ax.set_title(title, fontsize=14, pad=20)

    fig.colorbar(im, ax=ax, label="SNR", extend='both')
    
    x_label = param_grb_names_latex.get(x_key, x_key) + (f" [{param_grb_units[x_key]}]" if x_key in param_grb_units else "")
    y_label = param_grb_names_latex.get(y_key, y_key) + (f" [{param_grb_units[y_key]}]" if y_key in param_grb_units else "")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    
    import math
    exp_min = int(np.floor(np.log10(snr_min)))
    exp_max = int(np.ceil(np.log10(snr_max)))
    
    contour_levels = [10**e for e in range(-4, exp_max+1, 2) if 10**e >= snr_min and 10**e <= snr_max]

    # Trace les contours
    contours = ax.contour(X, Y, snr_grid_plot, levels=contour_levels, colors='k', linewidths=1.2, linestyles='--', zorder=4)
    ax.clabel(contours, inline=True, fontsize=12, fmt=lambda x: f"{x:.2g}")
    
    legend_labels, legend_handles = [], []
    if grb_parameters_list is not None:
        color_dict = {"BOAT": 'crimson', "GRB 250702B": 'royalblue', "GRB 980425": 'darkorange', "GRB 170817A": 'forestgreen'}
        for grb in grb_parameters_list:
            x_val = grb.get(x_key)
            y_val = grb.get(y_key)
            
            if x_key in ["theta", "theta_j", "phi_ej"]: x_val = np.rad2deg(x_val)
            if y_key in ["theta", "theta_j", "phi_ej"]: y_val = np.rad2deg(y_val)

            if x_val is not None and y_val is not None:
                color = color_dict.get(grb['name'], 'black')
                ax.scatter(x_val, y_val, marker='o', s=125, c=[color], edgecolors='black', linewidth=1.5, zorder=5)
                legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, markeredgecolor='black', markeredgewidth=1.5, linestyle='None'))
                legend_labels.append(grb['name'])
    
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(legend_handles), frameon=False, fontsize=13)
    
    filename = f"SNR_heatmap_{x_key}_{y_key}.png"
    plt.savefig(os.path.join(outdir, filename), dpi=300, bbox_inches='tight', pad_inches=0.5) 
    print(f"Saved heatmap to {os.path.join(outdir, filename)}")
    plt.close()
    
    df = pd.DataFrame(snr_grid, index=y_values, columns=x_values)
    csv_filename = os.path.join(outdir, f"SNR_heatmap_{x_key}_{y_key}.csv")
    df.to_csv(csv_filename)
    print(f"SNR grid saved to {csv_filename}")
    
def calculate_and_print_grb_snrs(grb_data):
    """
    Calcule et affiche le SNR pour une liste de configurations de GRB.
    """
    print("\n--- Calcul du SNR pour des GRBs spécifiques ---")
    
    # Initialiser l'objet LISA une seule fois
    lisa = li.LISA()
    
    for name, params in grb_data.items():
        # Préparer le dictionnaire complet de paramètres pour ce GRB
        current_params = {
            "phi_ej": 0.0,
            "beta": 0.99,
        }
        current_params.update(params)
        
        try:
            # Calculer la forme d'onde (hc)
            f, hc = compute_hc_grb_only(current_params)
            
            # Calculer le bruit du détecteur pour les fréquences du signal
            # np.sqrt(f * np.abs(lisa.Sn(f))) est le h_n
            h_n = np.sqrt(f * np.abs(lisa.Sn(f)))
            
            # Calculer le SNR
            snr = compute_snr(f, hc, f, h_n)
            
            print(f"SNR pour {name}: {snr:.4f}")
            
        except Exception as e:
            print(f"Erreur lors du calcul pour {name}: {e}")

    print("--- Fin du calcul des SNRs spécifiques ---\n")
    


def calculate_snr_grid_grb(param_dict, fixed_params={}):
    """
    Calcule la grille de SNR pour les paramètres GRB donnés, avec une barre de progression optimisée.
    """
    scan_keys = [k for k, v in param_dict.items() if isinstance(v, (list, np.ndarray))]
    if len(scan_keys) != 2:
        raise ValueError("param_dict doit contenir exactement 2 axes de scan (listes ou tableaux numpy)")
    x_key, y_key = scan_keys
    x_values, y_values = np.array(param_dict[x_key]), np.array(param_dict[y_key])

    param_grb_base = {
        "E_grb": 1e52, "theta": np.deg2rad(10), "phi_ej": 0.0,
        "r": 100, "beta": 0.99, "end_grb": 10, "E_aft": 1e52, 
        "theta_j": np.deg2rad(5)
    }
    param_grb_base.update({k: v for k, v in param_dict.items() if not isinstance(v, (list, np.ndarray))})
    param_grb_base.update(fixed_params)

    snr_grid = np.zeros((len(y_values), len(x_values)))
    lisa = li.LISA()

    # Calculer le nombre d'itérations pour 1% de la progression
    total_iterations = len(y_values)
    # Mettre à jour au minimum toutes les 1% ou à chaque itération si le total est < 100
    min_iters_for_update = max(1, total_iterations // 100)

    # Utilisation de tqdm avec miniters pour contrôler la fréquence de mise à jour
    progress_bar = tqdm(
        enumerate(y_values), 
        total=total_iterations, 
        desc=f"Calcul grille {y_key} vs {x_key}",
        miniters=min_iters_for_update,
        unit="row" # Ajoute une unité plus descriptive
    )
    
    for i, y_val in progress_bar:
        for j, x_val in enumerate(x_values):
            params = param_grb_base.copy()
            params[x_key] = x_val
            params[y_key] = y_val
            
            if x_key == 'z': params['r'] = Planck18.luminosity_distance(x_val).to_value(u.Mpc)
            if y_key == 'z': params['r'] = Planck18.luminosity_distance(y_val).to_value(u.Mpc)

            try:
                f, hc = compute_hc_grb_only(params)
                h_n = np.sqrt(f * np.abs(lisa.Sn(f)))
                snr = compute_snr(f, hc, f, h_n)
                snr_grid[i, j] = snr
            except Exception as e:
                snr_grid[i, j] = np.nan
                
    return snr_grid, x_key, y_key, x_values, y_values

def plot_combined_heatmap_snr_grb(param_dict_global, param_dict_zoom, outdir, fixed_params={}, xscale="log", yscale="log", levels=(1, 10, 100), grb_parameters_list=None):
    """
    Génère un graphique à 2 panneaux avec une heatmap SNR globale et une région zoomée pour les GRBs.
    """
    os.makedirs(outdir, exist_ok=True)
    from matplotlib.patches import Rectangle
    from matplotlib.colors import LogNorm

    # --- 1. Calculer les grilles de SNR ---
    snr_grid_global, x_key, y_key, x_vals_global, y_vals_global = calculate_snr_grid_grb(param_dict_global, fixed_params)
    snr_grid_zoom, _, _, x_vals_zoom, y_vals_zoom = calculate_snr_grid_grb(param_dict_zoom, fixed_params)

    # --- 2. Préparer la figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
    
    # Créer les objets Colormap correctement
    cmap_global = ListedColormap(plt.get_cmap('PuOr_r')(np.linspace(0, 0.5, 128)))
    cmap_zoom = ListedColormap(plt.get_cmap('PuOr')(np.linspace(0, 0.5, 128)))


    # --- 3. Tracer la Heatmap Globale (gauche) ---
    norm_global = LogNorm(vmin=np.nanmin(snr_grid_global[snr_grid_global > 0]), vmax=np.nanmax(snr_grid_global))
    X1, Y1 = np.meshgrid(x_vals_global, y_vals_global)
    # rdbu mais on ne prend que la moitié et en commançant par 0.5 vers 0
  
    im1 = ax1.pcolormesh(X1, Y1, snr_grid_global, shading='auto', cmap=cmap_global, norm=norm_global)
    fig.colorbar(im1, ax=ax1, label="SNR", extend='both')
    
    contours1 = ax1.contour(X1, Y1, snr_grid_global, levels=levels, colors="k", linewidths=1.2, linestyles="--", zorder=4)
    ax1.clabel(contours1, inline=True, fontsize=10, fmt="%d")
    ax1.set(xscale=xscale, yscale=yscale, xlabel=f"{x_key}", ylabel=f"{y_key}", title="")

    # Dessiner le rectangle de zoom
    rect = Rectangle((x_vals_zoom[0], y_vals_zoom[0]), x_vals_zoom[-1] - x_vals_zoom[0], y_vals_zoom[-1] - y_vals_zoom[0],
                     linewidth=1.5, edgecolor='k', facecolor='none', zorder=1)
    ax1.add_patch(rect)

    # --- 4. Tracer la Heatmap Zoomée (droite) ---
    norm_zoom = LogNorm(vmin=np.nanmin(snr_grid_zoom[snr_grid_zoom > 0]), vmax=np.nanmax(snr_grid_zoom))
    X2, Y2 = np.meshgrid(x_vals_zoom, y_vals_zoom)
    
    im2 = ax2.pcolormesh(X2, Y2, snr_grid_zoom, shading='auto', cmap=cmap_zoom, norm=norm_zoom)
    fig.colorbar(im2, ax=ax2, label="SNR", extend='both')

    contours2 = ax2.contour(X2, Y2, snr_grid_zoom, levels=levels, colors="k", linewidths=1.2, linestyles="--", zorder=4)
    ax2.clabel(contours2, inline=True, fontsize=10, fmt="%d")
    ax2.set(xscale=xscale, yscale=yscale, xlabel=f"{x_key}", title="")
    
    # --- 5. Ajouter les marqueurs GRB ---
    if grb_parameters_list:
        color_dict = {"BOAT": 'crimson', "GRB 250702B": 'royalblue', "GRB 980425": 'darkorange', "GRB 170817A": 'forestgreen'}
        for ax in [ax1, ax2]:
            for grb in grb_parameters_list:
                x_val, y_val = grb.get(x_key), grb.get(y_key)
                if x_val is not None and y_val is not None:
                    color = color_dict.get(grb['name'], 'black')
                    ax.scatter(x_val, y_val, marker='o', s=150, c=[color], edgecolors='black', linewidth=1.5, zorder=5, label=grb['name'])

    # --- 6. Titre et sauvegarde ---
    title_parts = [f"{k}={v:.1e}" if isinstance(v, float) else f"{k}={v}" for k, v in fixed_params.items()]
    fig.suptitle(" | ".join(title_parts), fontsize=14)
    
    # set x/y lim for the 2 regions
    ax1.set_xlim(x_vals_global[0], x_vals_global[-1])
    ax1.set_ylim(y_vals_global[0], y_vals_global[-1])
    ax2.set_xlim(x_vals_zoom[0], x_vals_zoom[-1])
    ax2.set_ylim(y_vals_zoom[0], y_vals_zoom[-1])
    
    filename = f"SNR_heatmap_combined_{x_key}_{y_key}.png"
    plt.savefig(os.path.join(outdir, filename), dpi=300, bbox_inches='tight')
    print(f"Graphique combiné sauvegardé dans {os.path.join(outdir, filename)}")
    plt.close()
    

if __name__ == "__main__":
    outdir = "results_snr_heatmap_grb_final"
    os.makedirs(outdir, exist_ok=True)
        
    # MODIFICATION: Mise à jour des paramètres des GRB avec les infos du tableau
    grb_params_data = {
        "BOAT": {
            "E_grb": 1e55, "E_aft": 3e54, "end_grb": 600, "r": 724.0* 1e6, # Convertir de Mpc à pc
            "theta_j": 0.014, "theta": np.mean([0.00094, 0.0059]) # rad
        },
        "GRB 250702B": {
            "E_grb": 2.2e54, "E_aft": 8e54, "end_grb": 25200, "r": Planck18.luminosity_distance(1.036).to_value(u.pc),
            "theta_j": np.deg2rad(0.5), "theta": np.deg2rad(65) # Assumed nearly face-on
        },
        "GRB 980425": {
            "E_grb": 6e47, "E_aft": 1e49, "end_grb": 30, "r": 36.9* 1e6, # Convertir de Mpc à pc
            "theta_j": np.deg2rad(np.mean([18, 31])), "theta": np.deg2rad(np.mean([24, 35]))
        },
        "GRB 170817A": {
            "E_grb": 3e46, "E_aft": 10**52.2, "end_grb": 2, "r": 40.0* 1e6, # Convertir de Mpc à pc
            "theta_j": 0.3, "theta": np.deg2rad(32) # rad
        }
    }
    print("GRB parameters:", grb_params_data)
    calculate_and_print_grb_snrs(grb_params_data)
    grb_parameters_list = []
    for name, params in grb_params_data.items():
        full_params = {
            "name": name,
            "phi_ej": 0.0,
            "beta": 0.99,
        }
        full_params.update(params)
        grb_parameters_list.append(full_params)
        
    print("Starting SNR heatmap computation...")
    
        # E_grb vs E_aft
    param_dict = {
        'E_grb': np.logspace(45, 60, 250), # erg
        'E_aft': np.logspace(45, 60, 250), # erg
        'end_grb': 10,
        'r': 100*1e6, # Mpc
        'theta': np.deg2rad(10),
        'theta_j': np.deg2rad(5)
    }
    plot_heatmap_snr(param_dict, outdir=outdir, xscale='log', yscale='log', grb_parameters_list=grb_parameters_list)

    # end_grb vs E_aft
    param_dict = {
        'end_grb': np.logspace(0, 5, 250), # s
        'E_aft': np.logspace(48, 55, 250), # erg
        'E_grb': 1e50,
        'r': 100 *1e6
    }
    plot_heatmap_snr(param_dict, outdir=outdir, xscale='log', yscale='log', grb_parameters_list=grb_parameters_list)

    # theta vs theta_j
    param_dict = {
        'theta': np.deg2rad(np.linspace(0.1, 180, 250)),
        'theta_j': np.deg2rad(np.linspace(0.1, 180, 250)),
        'E_aft': 1e52,
        'end_grb': 10,
        'r': 100*1e6, 
        'E_grb': 1e52
    }
    plot_heatmap_snr(param_dict, outdir=outdir, xscale='linear', yscale='linear', grb_parameters_list=grb_parameters_list)


    param_dict = {
    'beta': np.linspace(0.4, 0.9999, 250),
        'theta': np.deg2rad(np.linspace(0.1, 90, 250)),
        'E_aft': 1e54,
        'end_grb': 10,
        'r': 1*1e6, # Mpc
        'E_grb': 1e52,
        'theta_j': np.deg2rad(5)
    }
    plot_heatmap_snr(param_dict, outdir=outdir, xscale='linear', yscale='linear', grb_parameters_list=grb_parameters_list)

    # test r vs E_grb
    param_dict = {
        'r': np.logspace(5, 9, 250), # pc
        'E_grb': np.logspace(48, 55, 250), # erg
        'E_aft': 1e52, 
    }
    plot_heatmap_snr(param_dict, outdir=outdir, xscale='log', yscale='log', grb_parameters_list=grb_parameters_list)