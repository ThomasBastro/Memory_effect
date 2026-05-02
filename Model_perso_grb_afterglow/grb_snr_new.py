import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18
import astropy.units as u
import utils_grb_aft_new as u_grb
import LISA as li
from scipy.interpolate import interp1d
import os
from matplotlib.colors import LogNorm
import warnings
from tqdm import tqdm
from matplotlib.colors import ListedColormap
import pandas as pd
warnings.filterwarnings("ignore")

# --- PARAMÈTRES PAR DÉFAUT POUR LE MODÈLE UNIFIÉ ---
DEFAULT_T90 = 100.0
DEFAULT_TDEC = 50.0
DEFAULT_TJET = 1 * 86400.0  # 3 jours en secondes

def compute_hc_grb_only(params_grb):
    # Récupération des paramètres requis
    E_grb = params_grb['E_grb']
    T_90 = params_grb.get('T_90', DEFAULT_T90)
    E_aft = params_grb['E_aft']
    beta = params_grb['beta']
    t_dec = params_grb.get('t_dec', DEFAULT_TDEC)
    t_jet_break = params_grb.get('t_jet_break', DEFAULT_TJET)
    d = params_grb['r']
    theta = params_grb['theta']
    theta_j = params_grb['theta_j']

    # Domaine temporel pour la mémoire
    t = np.linspace(0, t_jet_break * 1.2, 5000)
    delta_h, delta_h_GRB, delta_h_aft = u_grb.grb_afterglow_model(
        t, E_grb, T_90, E_aft, beta, t_dec, t_jet_break, d, theta, theta_j
    )

    # Domaine fréquentiel
    f = np.logspace(-5, 0, 5000)
    h_ft, h_GRB, h_aft = u_grb.fft(f, delta_h_GRB, T_90, delta_h_aft, t_dec, t_jet_break)
    hc_grb = 2.0 * f * np.abs(h_ft)
    # Filtrage des fréquences utiles
    mask = (f >= 1e-4) & (f <= 1e-1)
    f = f[mask]
    hc_grb = hc_grb[mask]
    return f, hc_grb

def compute_snr(f_signal, h_c, f_noise, h_n):
    f_signal = np.array(f_signal)
    h_c = np.array(h_c)
    f_noise = np.array(f_noise)
    h_n = np.array(h_n)

    interp_hn = interp1d(f_noise, h_n, kind='linear', bounds_error=False, fill_value=np.inf)
    h_n_interp = interp_hn(f_signal)
    mask = (f_signal >= 1e-4) & (f_signal <= 1e-1)
    f_signal = f_signal[mask]
    h_c = h_c[mask]
    h_n_interp = h_n_interp[mask]

    integrand = np.abs((h_c**2) / (h_n_interp**2))
    snr_squared = np.trapezoid(integrand, x=np.log(f_signal))
    snr = np.sqrt(snr_squared)
    return snr

def plot_heatmap_snr(param_dict, outdir, xscale="log", yscale="log", grb_parameters_list=None):
    x_key, y_key = [k for k, v in param_dict.items() if isinstance(v, (list, np.ndarray))]
    x_values = np.array(param_dict[x_key])
    y_values = np.array(param_dict[y_key])

    # Paramètres par défaut
    param_grb = {
        "E_grb": 1e50, "theta": np.deg2rad(10), "phi_ej": 0.0,
        "r": 1e6, "beta": 0.99, "E_aft": 1e52, "theta_j": np.deg2rad(5),
        "T_90": DEFAULT_T90, "t_dec": DEFAULT_TDEC, "t_jet_break": DEFAULT_TJET
    }
    param_grb.update({k: v for k, v in param_dict.items() if not isinstance(v, (list, np.ndarray))})

    snr_grid = np.zeros((len(y_values), len(x_values)))
    lisa = li.LISA()

    total_iterations = len(y_values)
    min_iters_for_update = max(1, total_iterations // 100)
    progress_bar = tqdm(
        enumerate(y_values), 
        total=total_iterations, 
        desc=f"Calcul Heatmap {y_key} vs {x_key}",
        miniters=min_iters_for_update,
        unit="row"
    )
    for i, y in progress_bar:
        for j, x in enumerate(x_values):
            params = param_grb.copy()
            params[x_key] = x
            params[y_key] = y
            if x_key == 'z': params['r'] = Planck18.luminosity_distance(x).to_value(u.Mpc)
            if y_key == 'z': params['r'] = Planck18.luminosity_distance(y).to_value(u.Mpc)
            try:
                f, hc = compute_hc_grb_only(params)
                snr = compute_snr(f, hc, f, np.sqrt(f * np.abs(lisa.Sn(f))))
                snr_grid[i, j] = snr
            except Exception as e:
                progress_bar.write(f"Erreur: {e}")
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
    pos = valid & (snr_grid > 0)
    snr_grid_plot = snr_grid.copy()
    snr_grid_plot = np.where(pos, snr_grid_plot, np.nan)
    snr_min = np.nanmin(snr_grid_plot)
    snr_max = np.nanmax(snr_grid_plot)
    
        # Si le ratio est faible (< 10), force une échelle linéaire ou ajuste vmin/vmax
    if snr_max / snr_min < 10:
        # échelle linéaire
        norm = None
    else:
        # échelle log, mais vmin/vmax robustes
        norm = LogNorm(vmin=snr_min, vmax=snr_max)

    
    snr_masked = np.ma.masked_invalid(snr_grid_plot) # mask nans for contouring
    X, Y = np.meshgrid(x_values_plot, y_values_plot)
    cmap = 'Greens'
    if x_key == 't_dec' and y_key == 't_jet_break':
         # put as nan the points where t_jet_break < t_dec (non-physique)
        snr_masked = np.where(Y < X, np.nan, snr_masked)
    im = ax.pcolormesh(X, Y, snr_masked, shading='gouraud', cmap=cmap, norm=norm)

    # Titre avec les paramètres fixes
    fixed_params = {k: v for k, v in param_grb.items() if k not in [x_key, y_key]}
    if 'phi_ej' in fixed_params and fixed_params['phi_ej'] == 0.0:
        del fixed_params['phi_ej']
    param_grb_names_latex = {
        'E_grb': r'$E_{\rm{GRB}}$', 'E_aft': r'$E_{\rm{aft}}$', 'theta': r'$\theta_{\rm ej}$',
        'theta_j': r'$\theta_j$', 'phi_ej': r'$\phi_{\rm{ej}}$', 'beta': r'$\beta$',
        'end_grb': r'$T_{\rm{GRB}}$', 'r': r'$d$',
        'T_90': r'$T_{90}$', 't_dec': r'$t_{\rm dec}$', 't_jet_break': r'$t_{\rm break}$'
    }
    param_grb_units = {
        'E_grb': 'erg', 'E_aft': 'erg', 'theta': r'$^\circ$', 'theta_j': r'$^\circ$',
        'phi_ej': r'$^\circ$', 'beta': 'c', 'end_grb': 's', 'r': 'Mpc',
        'T_90': 's', 't_dec': 's', 't_jet_break': 's'
    }

    param_strings = []
    for k, v in fixed_params.items():
        name = param_grb_names_latex.get(k, k)
        unit = param_grb_units.get(k, '')
        val_str = ""
        if k in ['theta', 'theta_j', 'phi_ej']:
            val_str = f"{np.rad2deg(v):.1f}"
        elif k == 'r':
            val_str = f"{v/1e6:.1f}"  # pc -> Mpc
        elif isinstance(v, float) and (v < 1e-3 or v > 1e3):
            val_str = f"{v:.1e}"
        elif k== 'beta':
            val_str = f"{v:.2f}"
        else:
            val_str = f"{v:.1f}" if isinstance(v, float) else str(v)
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
    
    contours = ax.contour(X, Y, snr_masked, levels=contour_levels, colors='k', linewidths=1.2, linestyles='-', zorder=4)
    ax.clabel(contours, inline=True, fontsize=10, fmt=lambda x: f"{x:.3g}")
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
    # fit de la "ridge" de haute SNR
    #fit_high_snr_ridge(X, Y, snr_grid, percentile=99.85, logx=(xscale=='log'), logy=(yscale=='log'), ax=ax, color='lime', label='Ridge fit')    
    
    if x_key == 't_dec' and y_key == 't_jet_break':
        plot_tjet_tdec_delta_lines(ax, x_values_plot, y_values_plot, xscale, yscale)
    
    filename = f"SNR_heatmap_{x_key}_{y_key}.png"
    plt.savefig(os.path.join(outdir, filename), dpi=300, bbox_inches='tight', pad_inches=0.5) 
    print(f"Saved heatmap to {os.path.join(outdir, filename)}")
    plt.close()
    # --- Sauvegarde CSV avec tous les paramètres fixes ---
    df = pd.DataFrame(snr_grid, index=y_values, columns=x_values)
    csv_filename = os.path.join(outdir, f"SNR_heatmap_{x_key}_{y_key}.csv")
    df.to_csv(csv_filename)
    print(f"SNR grid saved to {csv_filename}")

def calculate_and_print_grb_snrs(grb_data):
    print("\n--- Calcul du SNR pour des GRBs spécifiques ---")
    lisa = li.LISA()
    for name, params in grb_data.items():
        current_params = {
            "phi_ej": 0.0,
            "beta": 0.99,
            "t_dec": DEFAULT_TDEC,
            "t_jet_break": DEFAULT_TJET
        }
        current_params.update(params)
        try:
            f, hc = compute_hc_grb_only(current_params)
            h_n = np.sqrt(f * np.abs(lisa.Sn(f)))
            snr = compute_snr(f, hc, f, h_n)
            print(f"SNR pour {name}: {snr:.4e}")
        except Exception as e:
            print(f"Erreur lors du calcul pour {name}: {e}")
    print("--- Fin du calcul des SNRs spécifiques ---\n")

def calculate_snr_grid_grb(param_dict, fixed_params={}):
    scan_keys = [k for k, v in param_dict.items() if isinstance(v, (list, np.ndarray))]
    if len(scan_keys) != 2:
        raise ValueError("param_dict doit contenir exactement 2 axes de scan (listes ou tableaux numpy)")
    x_key, y_key = scan_keys
    x_values, y_values = np.array(param_dict[x_key]), np.array(param_dict[y_key])
    param_grb_base = {
        "E_grb": 1e52, "theta": np.deg2rad(10), "phi_ej": 0.0,
        "r": 100, "beta": 0.99, "E_aft": 1e52, "theta_j": np.deg2rad(5),
        "T_90": DEFAULT_T90, "t_dec": DEFAULT_TDEC, "t_jet_break": DEFAULT_TJET
    }
    param_grb_base.update({k: v for k, v in param_dict.items() if not isinstance(v, (list, np.ndarray))})
    param_grb_base.update(fixed_params)
    snr_grid = np.zeros((len(y_values), len(x_values)))
    lisa = li.LISA()
    total_iterations = len(y_values)
    min_iters_for_update = max(1, total_iterations // 100)
    progress_bar = tqdm(
        enumerate(y_values), 
        total=total_iterations, 
        desc=f"Calcul grille {y_key} vs {x_key}",
        miniters=min_iters_for_update,
        unit="row"
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
                progress_bar.write(f"Erreur: {e} | t_dec={params['t_dec']}, t_jet_break={params['t_jet_break']}")
                snr_grid[i, j] = np.nan
    return snr_grid, x_key, y_key, x_values, y_values

from scipy.stats import linregress

def fit_high_snr_ridge(X, Y, snr_grid, percentile=99.5, logx=True, logy=True, ax=None, color='lime', label='Ridge fit'):
    # X, Y: meshgrid (2D arrays), snr_grid: 2D array
    snr_flat = snr_grid.flatten()
    x_flat = X.flatten()
    y_flat = Y.flatten()
    # Isoler les points au-dessus du percentile
    threshold = np.nanpercentile(snr_flat, percentile)
    mask = (snr_flat >= threshold) & np.isfinite(snr_flat) & np.isfinite(x_flat) & np.isfinite(y_flat)
    x_sel = x_flat[mask]
    y_sel = y_flat[mask]
    if logx:
        x_sel = np.log10(x_sel)
    if logy:
        y_sel = np.log10(y_sel)
    # Régression linéaire
    slope, intercept, r_value, p_value, std_err = linregress(x_sel, y_sel)
    print(f"Fit ridge: y = {slope:.3f} * x + {intercept:.3f} (R²={r_value**2:.3f})")
    # Tracé sur la figure
    if ax is not None:
        x_plot = np.linspace(np.nanmin(X), np.nanmax(X), 200)
        if logx:
            x_plot_log = np.log10(x_plot)
        else:
            x_plot_log = x_plot
        y_plot_log = slope * x_plot_log + intercept
        if logy:
            y_plot = 10**y_plot_log
        else:
            y_plot = y_plot_log
        ax.plot(x_plot, y_plot, color=color, lw=2.5, label=label)
        ax.legend()
    return slope, intercept


def plot_tjet_tdec_delta_lines(ax, x_vals, y_vals, xscale='log', yscale='log'):
    """
    Trace sur le heatmap les lignes où t_jet_break - t_dec = const (1s, 1min, 1h, 1j)
    """
    deltas = [1, 60, 3600, 86400, 7*86400]  # en secondes
    labels = ['1 s', '1 min', '1 h', '1 day', '1 week']
    cmap = plt.get_cmap('BuPu')
    cmap = ListedColormap(cmap(np.linspace(0.3, 1, len(deltas))))  
    colors = [cmap(i / len(deltas)) for i in range(len(deltas))]
    x = np.array(x_vals)
    y = np.array(y_vals)

    for delta, label, color in zip(deltas, labels, colors):
        # Pour chaque valeur de x (t_dec), calcule y = x + delta
        y_line = x + delta
        # Ne garde que les points où y_line est dans les bornes de y
        mask = (y_line >= y.min()) & (y_line <= y.max())
        if np.any(mask):
            ax.plot(x[mask], y_line[mask], '--', label=f"$t_{{break}}-t_{{dec}}$={label}", linewidth=1.5, color=color)
        else:
            print(f"Aucune ligne tracée pour delta={delta} car hors des limites du graphique.")
    ax.legend(loc='lower right', fontsize=14)
    
def plot_hc_vs_lisa(grb_dict, outdir="plots_hc", fname="hc_vs_lisa.png"):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(10, 7))

    # 1. Tracé LISA
    lisa = li.LISA()
    f_lisa = np.logspace(-5, 0, 500)
    hn = np.sqrt(f_lisa * np.abs(lisa.Sn(f_lisa)))
    plt.loglog(f_lisa, hn, label="LISA sensitivity", lw=1.5, ls='--', color='black', alpha=0.6)

    # 2. Itération sur ton dictionnaire grb_params_data
    # On utilise .items() pour avoir le nom ("BOAT") et les params ({...})
    for name, data in grb_dict.items():
        # On fait une copie pour ne pas polluer le dictionnaire original
        params = data.copy()
        
        # Ajout des valeurs par défaut
        params.setdefault("phi_ej", 0.0)
        params.setdefault("beta", 0.99)
        params.setdefault("t_dec", DEFAULT_TDEC)
        params.setdefault("t_jet_break", DEFAULT_TJET)
        
        # Calcul
        f, hc = compute_hc_grb_only(params)
        
        # Plot avec le nom du GRB comme label
        plt.loglog(f, hc, label=f"{name}", lw=2)

    # Cosmétique
    plt.xlabel("f [Hz]", fontsize=14)
    plt.ylabel(r"$h_c$", fontsize=14)
    plt.xlim(1e-4, 1e-1)
    plt.ylim(1e-30, 1e-21)
    plt.legend(fontsize=10, loc='best', frameon=True)
    plt.grid(True, which="both", ls=':', alpha=0.5)
    plt.tight_layout()
    
    path = os.path.join(outdir, fname)
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Figure saved to {path}")

if __name__ == "__main__":
    outdir = "results_snr_heatmap_grb_newmodel"
    os.makedirs(outdir, exist_ok=True)
    grb_params_data = {
        "BOAT": {
            "E_grb": 1e55, "E_aft": 3e54, "r": 724.0* 1e6,
            "theta_j": 0.014, "theta": np.mean([0.00094, 0.0059]), 'T_90': 600
        },
        "GRB 250702B": {
            "E_grb": 2.2e54, "E_aft": 8e54, "r": Planck18.luminosity_distance(1.036).to_value(u.pc),
            "theta_j": np.deg2rad(0.5), "theta": np.deg2rad(65), 'T_90': 25000
        },
        "GRB 980425": {
            "E_grb": 6e47, "E_aft": 1e49, "r": 36.9* 1e6,
            "theta_j": np.deg2rad(np.mean([18, 31])), "theta": np.deg2rad(np.mean([24, 35])), 'T_90': 30
        },
        "GRB 170817A": {
            "E_grb": 3e46, "E_aft": 10**52.2, "r": 40.0* 1e6,
            "theta_j": 0.3, "theta": np.deg2rad(32), 'T_90': 2
        },
        "GRB 221009A": {
            "E_grb": 7.6e51, "E_aft": 5e52, 'r': 350.0* 1e6, 'T_90': 51.37, 'theta_j': np.deg2rad(0.04),
            'theta': np.deg2rad(30)
            }
        
    }
    print("GRB parameters:", grb_params_data)
   
  
    params = {
        'd': np.logspace(5, 10.5, 200),
        'E_aft': np.logspace(48, 60, 200),
        "E_grb": 1e52
    }
    plot_heatmap_snr(params, outdir, xscale="log", yscale="log", grb_parameters_list=list(grb_params_data.values()))
   