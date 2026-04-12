import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.colors import LogNorm
import utils_phenom as up
import utils_GRB_afterglow_phenom as ugrb
import os
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib.patches import Rectangle

from tqdm.auto import tqdm
import pandas as pd

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

def compute_grb_component(E_grb, theta, phi_ej, d_pc, beta, E_aft, duration, theta_j):
    h_in = ugrb.memory_initial_acceleration_GRB(E_grb, theta, phi_ej, d_pc, beta)
    h_aft = ugrb.memory_afterglow(E_aft, theta, phi_ej, d_pc, beta)
    t_grb = np.linspace(-5.0, 500.0, 100000)
    h_tot, _ = ugrb.memory_total_waveform(t_grb, h_in, h_aft, duration, theta, d_pc, radius=0)
    f_grb, fft_grb = ugrb.total_waveform_fft(h_in, h_aft, duration, theta, d_pc, radius=0)
    hc_grb = 2.0 * f_grb * np.abs(fft_grb)
    return {"t": t_grb, "h_in": h_in, "h_aft": h_aft, "h": h_tot, "f": f_grb, "hc": hc_grb}

def compute_linear_memory_component(t, M_ej, v_ej, tau, r, start):
    h = up.linear_memory_ejecta_masked(t, M_ej, v_ej, tau, r, start, model="exponential")
    delta_h = up.delta_h_ejecta(M_ej, v_ej, r)
    f_exp, fft_exp = up.fft_exp_model(delta_h, tau)
    return {"t": t, "h": h, "f": f_exp, "hc": 2 * f_exp * np.abs(fft_exp)}

def build_memory_components():
    nonlinear, kilonova = load_memory_inputs()
    r_ref, theta, phi_ej, beta = 40e6, np.deg2rad(32), 0.0, 0.99
    components = {
        "dyn": compute_linear_memory_component(np.linspace(-5.0, 50.0, 100000), M_ej=0.5e-2, v_ej=0.25, tau=1e-3, r=r_ref, start=-1e-3),
        "wind": compute_linear_memory_component(np.linspace(-5.0, 50.0, 100000), M_ej=0.01, v_ej=0.05, tau=10.0, r=r_ref, start=0.01),
        "grb": compute_grb_component(E_grb=3e46, theta=theta, phi_ej=phi_ej, d_pc=r_ref, beta=beta, E_aft=1e52, duration=2.0, theta_j=np.deg2rad(5)),
        "nonlinear": nonlinear, "kilonova": kilonova,
    }
    return components, r_ref

def compute_snr_etd(f_signal, hc_signal, f_noise, h_noise):
    f_signal, hc_signal = np.asarray(f_signal), np.asarray(hc_signal)
    mask = (f_signal >= 1.0) & (f_signal <= 1e4)
    f_masked, hc_masked = f_signal[mask], hc_signal[mask]
    if len(f_masked) < 2: return 0.0
    interp_hn = interp1d(f_noise, h_noise, kind="linear", bounds_error=False, fill_value=np.inf)
    h_n_interp = interp_hn(f_masked)
    integrand = (hc_masked**2) / (h_n_interp**2)
    return np.sqrt(np.trapezoid(integrand, x=np.log(f_masked)))

def scale_with_distance(hc, r_ref_pc, r_pc):
    return hc * (r_ref_pc / r_pc)

def compute_total_hc(f_common, params_dyn, params_wind, params_grb, memory_components, r_ref_pc=40e6, add_dyn=True, add_wind=True, add_grb=True, add_nonlinear=True, add_kilonova=True):
    hc_total = np.zeros_like(f_common)
    r_pc = params_grb["r"]

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

def _axis_values_and_label(key, values):
    values = np.asarray(values)
    if key == "r": return values / 1e6, r"$d\,[{\rm Mpc}]$"
    if key in ["theta", "theta_j", "phi_ej"]: return np.rad2deg(values), rf"${_format_key_for_title(key)}\,[^\circ]$"
    if key == "E_grb": return values, r"$E_{\rm GRB}\,[{\rm erg}]$"
    if key == "E_aft": return values, r"$E_{\rm aft}\,[{\rm erg}]$"
    if key == "M_ej_dyn": return values, r"$M_{\rm ej,dyn}\,[M_\odot]$"
    if key == "M_ej_wind": return values, r"$M_{\rm ej,wind}\,[M_\odot]$"
    if key == 'v_ej_dyn': return values, r"$v_{\rm ej,dyn}\,[c]$"
    if key == "v_ej_wind": return values, r"$v_{\rm ej,wind}\,[c]$"
    if key in ["v_ej_dyn", "v_ej_wind"]: return values, r"$v_{\rm ej}\,[c]$"
    if key == "beta": return values, r"$\beta$"
    if key == "end_grb": return values, r"$T_{\rm GRB} [{\rm s}]$"
    return values, key

def _format_key_for_title(key):
    return {
        "E_grb": r"$E_{\rm GRB}$", "E_aft": r"$E_{\rm aft}$", "theta": r"$\theta_{\rm ej}$",
        "theta_j": r"$\theta_j$", "phi_ej": r"$\phi_{\rm ej}$", "r": r"$d$",
        "M_ej_dyn": r"$M_{\rm ej,dyn}$", "M_ej_wind": r"$M_{\rm ej,wind}$",
        "v_ej_dyn": r"$v_{\rm ej,dyn}$", "v_ej_wind": r"$v_{\rm ej,wind}$",
        'end_grb': r"$T_{\rm GRB}$", "beta": r"$\beta$",
    }.get(key, key)

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
    x_ref, y_ref = _axis_values_and_label(x_key, [gw170817[x_key]])[0], _axis_values_and_label(y_key, [gw170817[y_key]])[0]
    ax.scatter(x_ref, y_ref, marker="*", s=500, c="limegreen", edgecolors="black", linewidths=1.2, zorder=8, label="GW170817")

def plot_heatmap_snr(param_dict, outdir, memory_components, r_ref_pc, f_noise, h_noise, xscale="log", yscale="log", levels=(1, 10, 100), add_dyn=True, add_wind=True, add_grb=True, add_nonlinear=True, add_kilonova=True, filename_tag=None, if_170817_marker=True):
    scan_keys = [k for k, v in param_dict.items() if isinstance(v, (list, np.ndarray))]
    if len(scan_keys) != 2: raise ValueError("param_dict must contain exactly 2 scan axes")
    x_key, y_key = scan_keys
    x_values, y_values = np.array(param_dict[x_key]), np.array(param_dict[y_key])

    params_dyn = {"M_ej": 0.5e-2, "v_ej": 0.25, "tau": 1e-3, "r": r_ref_pc, "start": -1e-3}
    params_wind = {"M_ej": 0.01, "v_ej": 0.05, "tau": 10.0, "r": r_ref_pc, "start": 0.01}
    params_grb = {"E_grb": 3e46, "theta": np.deg2rad(32), "phi_ej": 0.0, "r": r_ref_pc, "beta": 0.99, "end_grb": 2.0, "E_aft": 10**(52.2), "theta_j": np.deg2rad(5)}
    
    all_params = [params_dyn, params_wind, params_grb]
    fixed_params = {k: v for k, v in param_dict.items() if k not in scan_keys}
    for k, v in fixed_params.items():
        for p in all_params:
            if k in p: p[k] = v
        if k == "r": params_dyn["r"] = params_wind["r"] = params_grb["r"] = v

    snr_grid = np.zeros((len(y_values), len(x_values)))
    for i, y_val in tqdm(enumerate(y_values), total=len(y_values), desc=f"Grid {y_key}"):
        for j, x_val in enumerate(x_values):
            p_dyn, pw, pg = params_dyn.copy(), params_wind.copy(), params_grb.copy()
            
            for k, v in [(x_key, x_val), (y_key, y_val)]:
                # Gérer les cas spécifiques pour dyn et wind
                if k == "M_ej_dyn":
                    p_dyn["M_ej"] = v
                elif k == "v_ej_dyn":
                    p_dyn["v_ej"] = v
                elif k == "M_ej_wind":
                    pw["M_ej"] = v
                elif k == "v_ej_wind":
                    pw["v_ej"] = v
                # Gérer les autres paramètres
                elif k in p_dyn:
                    p_dyn[k] = v
                elif k in pw:
                    pw[k] = v
                elif k in pg:
                    pg[k] = v
                # Gérer le cas spécial de la distance 'r'
                if k == "r":
                    p_dyn["r"] = v
                    pw["r"] = v
                    pg["r"] = v
   
            f_common = np.logspace(-4, 5, 50000)
            hc_total = compute_total_hc(f_common, p_dyn, pw, pg, memory_components, r_ref_pc, add_dyn, add_wind, add_grb, add_nonlinear, add_kilonova)
            snr_grid[i, j] = compute_snr_etd(f_common, hc_total, f_noise, h_noise)
    x_plot, x_label = _axis_values_and_label(x_key, x_values)
    y_plot, y_label = _axis_values_and_label(y_key, y_values)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    
    valid = np.isfinite(snr_grid) & (snr_grid > 0)
    norm = LogNorm(vmin=np.min(snr_grid[valid]) if np.any(valid) else 1e-2, vmax=np.max(snr_grid[valid]) if np.any(valid) else 1)
    
    original_cmap = plt.get_cmap("twilight")
  
    subset_colors = original_cmap(np.linspace(0, 0.5, 128))
  
    custom_cmap = ListedColormap(subset_colors)
    X, Y = np.meshgrid(x_plot, y_plot)
    im = ax.pcolormesh(X, Y, snr_grid, shading='gouraud', cmap=custom_cmap, norm=norm)
    fig.colorbar(im, ax=ax, label="SNR")
    
    ax.set(xscale=xscale, yscale=yscale, xlabel=x_label, ylabel=y_label)
    contours = ax.contour(X, Y, snr_grid, levels=levels, colors="k", linewidths=1.2, linestyles="--")
    ax.clabel(contours, inline=True, fontsize=10, fmt="%d")
    if if_170817_marker:
        _plot_gw170817_marker(ax, x_key, y_key)

        # Dictionnaire complet des paramètres fixes
    fixed_values_all = {
        "M_ej_dyn": params_dyn["M_ej"],
        "v_ej_dyn": params_dyn["v_ej"],
        "M_ej_wind": params_wind["M_ej"],
        "v_ej_wind": params_wind["v_ej"],
        "E_grb": params_grb["E_grb"],
        "E_aft": params_grb["E_aft"],
        "end_grb": params_grb["end_grb"],
        "theta": params_grb["theta"],
        "r": params_grb["r"],
    }

    # Filtrer pour ne garder que les paramètres qui ne sont pas scannés
    fixed_for_title = {k: v for k, v in fixed_values_all.items() if k not in scan_keys}
    
    # Construire les éléments du titre
    title_parts = [f"{_format_key_for_title(k)}={_format_value_for_title(k, v)}" for k, v in fixed_for_title.items()]
    
    # Diviser le titre en deux lignes pour la lisibilité
    n = (len(title_parts) + 1) // 2
    title_str = " | ".join(title_parts[:n])
    if len(title_parts) > n:
        title_str += "\n" + " | ".join(title_parts[n:])
    print(f"Title for plot: {title_str}")    
    ax.set_title(title_str, fontsize=10)
    
    base_filename = f"SNR_heatmap_{x_key}_{y_key}"
    if filename_tag: base_filename += f"_{filename_tag}"
    active = [name for name, flag in zip(["dyn", "wind", "grb", "nl", "kn"], [add_dyn, add_wind, add_grb, add_nonlinear, add_kilonova]) if flag]
    if len(active) < 5: base_filename += f"_{'-'.join(active)}"
    
    plt.savefig(os.path.join(outdir, f"{base_filename}.png"), dpi=300)
    pd.DataFrame(snr_grid, index=y_values, columns=x_values).to_csv(os.path.join(outdir, f"{base_filename}.csv"))
    plt.close()
    print(f"Saved files with base name {base_filename} to {outdir}")
    
def plot_combined_heatmap_snr(
    param_dict_global, param_dict_zoom, outdir, memory_components, r_ref_pc, f_noise, h_noise,
    fixed_params={}, xscale="log", yscale="log", levels=(1, 10, 100),
    add_dyn=True, add_wind=True, add_grb=True, add_nonlinear=True, add_kilonova=True,
    filename_tag=None, if_170817_marker_all=True, if_170817_marker_zoom=True, zoom_components=None
):
    """
    Génère un plot à 3 panneaux :
    - 1 : heatmap globale (tous composants)
    - 2 : zoom classique (tous composants)
    - 3 : zoom avec uniquement les composants spécifiés dans zoom_components
    """
    os.makedirs(outdir, exist_ok=True)

    # 1. Grille globale (inchangée)
    snr_grid_global, x_key, y_key, x_values_global, y_values_global = calculate_snr_grid(
        param_dict_global, fixed_params, memory_components, r_ref_pc, f_noise, h_noise,
        add_dyn, add_wind, add_grb, add_nonlinear, add_kilonova
    )

    # 2. Grille zoom classique (tous composants comme la globale)
    snr_grid_zoom, _, _, x_values_zoom, y_values_zoom = calculate_snr_grid(
        param_dict_zoom, fixed_params, memory_components, r_ref_pc, f_noise, h_noise,
        add_dyn, add_wind, add_grb, add_nonlinear, add_kilonova
    )

    # 3. Grille zoom avec sélection des composants
    if zoom_components is not None:
        z_add_dyn = zoom_components.get("dyn", False)
        z_add_wind = zoom_components.get("wind", False)
        z_add_grb = zoom_components.get("grb", False)
        z_add_nonlinear = zoom_components.get("nonlinear", False)
        z_add_kilonova = zoom_components.get("kilonova", False)
    else:
        z_add_dyn = add_dyn
        z_add_wind = add_wind
        z_add_grb = add_grb
        z_add_nonlinear = add_nonlinear
        z_add_kilonova = add_kilonova

    snr_grid_zoom_selected, _, _, _, _ = calculate_snr_grid(
        param_dict_zoom, fixed_params, memory_components, r_ref_pc, f_noise, h_noise,
        z_add_dyn, z_add_wind, z_add_grb, z_add_nonlinear, z_add_kilonova
    )

    # 4. Préparation des axes
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 8), constrained_layout=True)
    x_plot_global, x_label = _axis_values_and_label(x_key, x_values_global)
    y_plot_global, y_label = _axis_values_and_label(y_key, y_values_global)
    x_plot_zoom, _ = _axis_values_and_label(x_key, x_values_zoom)
    y_plot_zoom, _ = _axis_values_and_label(y_key, y_values_zoom)

    # Colormaps et normalisation
    twilight_map = plt.get_cmap("twilight")
    twilight_map_rev = plt.get_cmap("twilight_shifted")
    Purples = plt.get_cmap("Purples")
    cmap1 = ListedColormap(twilight_map(np.linspace(0, 0.5, 128)))
    cmap2 = ListedColormap(twilight_map_rev(np.linspace(0.5, 1, 128)))
    cmap3 = Purples

    valid_global = np.isfinite(snr_grid_global) & (snr_grid_global > 0)
    norm1 = LogNorm(vmin=np.min(snr_grid_global[valid_global]) if np.any(valid_global) else 1e-2,
                    vmax=np.max(snr_grid_global[valid_global]) if np.any(valid_global) else 1)
    valid_zoom = np.isfinite(snr_grid_zoom) & (snr_grid_zoom > 0)
    norm2 = LogNorm(vmin=np.min(snr_grid_zoom[valid_zoom]) if np.any(valid_zoom) else 1e-2,
                    vmax=np.max(snr_grid_zoom[valid_zoom]) if np.any(valid_zoom) else 1)
    valid_zoom_sel = np.isfinite(snr_grid_zoom_selected) & (snr_grid_zoom_selected > 0)
    norm3 = LogNorm(vmin=np.min(snr_grid_zoom_selected[valid_zoom_sel]) if np.any(valid_zoom_sel) else 1e-2,
                    vmax=np.max(snr_grid_zoom_selected[valid_zoom_sel]) if np.any(valid_zoom_sel) else 1)

    # 5. Plot global
    X1, Y1 = np.meshgrid(x_plot_global, y_plot_global)
    im1 = ax1.pcolormesh(X1, Y1, snr_grid_global, shading='gouraud', cmap=cmap1, norm=norm1)

    cbar1 = fig.colorbar(im1, ax=ax1, label="SNR", extend='both')
    cbar1.ax.tick_params(labelsize=22)
    cbar1.set_label("SNR", fontsize=24)
    ax1.set(xscale=xscale, yscale=yscale, xlabel=x_label, ylabel=y_label, title="")
    contours1 = ax1.contour(X1, Y1, snr_grid_global, levels=levels, colors="k", linewidths=1.2, linestyles="--")
    ax1.clabel(contours1, inline=True, fontsize=18, fmt=lambda x: f"{x:.2g}")

    # Draw zoom box sur le global
    zoom_x_min, zoom_x_max = _axis_values_and_label(x_key, [x_values_zoom[0], x_values_zoom[-1]])[0]
    zoom_y_min, zoom_y_max = _axis_values_and_label(y_key, [y_values_zoom[0], y_values_zoom[-1]])[0]
    rect = Rectangle((zoom_x_min, zoom_y_min), zoom_x_max - zoom_x_min, zoom_y_max - zoom_y_min,
                     linewidth=2.5, edgecolor='black', facecolor='none', zorder=10)
    ax1.add_patch(rect)

    if if_170817_marker_all:
        _plot_gw170817_marker(ax1, x_key, y_key)

    # 6. Plot zoom classique
    X2, Y2 = np.meshgrid(x_plot_zoom, y_plot_zoom)
    im2 = ax2.pcolormesh(X2, Y2, snr_grid_zoom, shading='gouraud', cmap=cmap2, norm=norm2)

    cbar2 = fig.colorbar(im2, ax=ax2, label="SNR", extend='both')
    cbar2.ax.tick_params(labelsize=22)
    cbar2.set_label("SNR", fontsize=24)
    ax2.set(xscale=xscale, yscale=yscale, xlabel=x_label, ylabel='', title="")
    contours2 = ax2.contour(X2, Y2, snr_grid_zoom, levels=levels, colors="k", linewidths=1.2, linestyles="--")
    ax2.clabel(contours2, inline=True, fontsize=18,fmt=lambda x: f"{x:.2g}")
    if if_170817_marker_zoom:
        _plot_gw170817_marker(ax2, x_key, y_key)

    # 7. Plot zoom sélectionné
    im3 = ax3.pcolormesh(X2, Y2, snr_grid_zoom_selected, shading='gouraud', cmap=cmap3, norm=norm3)

    cbar3 = fig.colorbar(im3, ax=ax3, label="SNR", extend='both')
    cbar3.ax.tick_params(labelsize=22)
    cbar3.set_label("SNR", fontsize=24)
    ax3.set(xscale=xscale, yscale=yscale, xlabel=x_label, ylabel='', title="")
    contours3 = ax3.contour(X2, Y2, snr_grid_zoom_selected, levels=levels, colors="k", linewidths=1.2, linestyles="--")
    ax3.clabel(contours3, inline=True, fontsize=18, fmt=lambda x: f"{x:.2g}")
    
    if if_170817_marker_zoom:
        _plot_gw170817_marker(ax3, x_key, y_key)
    # augmenter la taille des labels et des ticks pour tous les axes
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='both', which='major', labelsize=22)
        ax.tick_params(axis='both', which='minor', labelsize=22)
        ax.xaxis.label.set_size(24)
        ax.yaxis.label.set_size(24)   

    # --- 8. Titre général et sauvegarde ---
    params_dyn_base = {"M_ej": 0.5e-2, "v_ej": 0.25, "tau": 1e-3, "r": r_ref_pc, "start": -1e-3}
    params_wind_base = {"M_ej": 0.01, "v_ej": 0.05, "tau": 10.0, "r": r_ref_pc, "start": 0.01}
    params_grb_base = {"E_grb": 3e46, "theta": np.deg2rad(32), "phi_ej": 0.0, "r": r_ref_pc, "beta": 0.99, "end_grb": 2.0, "E_aft": 10**(52.2), "theta_j": np.deg2rad(5)}
    for p in [params_dyn_base, params_wind_base, params_grb_base]:
        p.update(fixed_params)
    if "r" in fixed_params:
        params_dyn_base["r"] = params_wind_base["r"] = params_grb_base["r"] = fixed_params["r"]
    fixed_values_for_title = {
        "M_ej_dyn": params_dyn_base["M_ej"],
        "v_ej_dyn": params_dyn_base["v_ej"],
        "M_ej_wind": params_wind_base["M_ej"],
        "v_ej_wind": params_wind_base["v_ej"],
        "E_grb": params_grb_base["E_grb"],
        "E_aft": params_grb_base["E_aft"],
        "end_grb": params_grb_base["end_grb"],
        "theta": params_grb_base["theta"],
        "r": params_grb_base["r"],
    }
    title_parts = [f"{_format_key_for_title(k)}={_format_value_for_title(k, v)}" for k, v in fixed_values_for_title.items() if k not in [x_key, y_key]]
    
    print(f"Title for combined plot: {title_parts}")
    fig.suptitle(" | ".join(title_parts), fontsize=28)

    # --- 9. Save files ---
    base_filename = f"SNR_heatmap_combined_{x_key}_{y_key}_3panels"
    if filename_tag: base_filename += f"_{filename_tag}"
    plt.savefig(os.path.join(outdir, f"{base_filename}.png"), dpi=300)
    pd.DataFrame(snr_grid_global, index=y_values_global, columns=x_values_global).to_csv(os.path.join(outdir, f"{base_filename}_global.csv"))
    pd.DataFrame(snr_grid_zoom, index=y_values_zoom, columns=x_values_zoom).to_csv(os.path.join(outdir, f"{base_filename}_zoom.csv"))
    pd.DataFrame(snr_grid_zoom_selected, index=y_values_zoom, columns=x_values_zoom).to_csv(os.path.join(outdir, f"{base_filename}_zoom_selected.csv"))
    plt.close()
    print(f"Saved combined plot and data with base name {base_filename} to {outdir}")


def calculate_snr_grid(param_dict, fixed_params_base, memory_components, r_ref_pc, f_noise, h_noise, add_dyn=True, add_wind=True, add_grb=True, add_nonlinear=True, add_kilonova=True):
    """
    Calculates the SNR grid for a given set of parameters.
    This function is extracted from plot_heatmap_snr to be reusable.
    """
    scan_keys = [k for k, v in param_dict.items() if isinstance(v, (list, np.ndarray))]
    if len(scan_keys) != 2:
        raise ValueError("param_dict must contain exactly 2 scan axes (lists or numpy arrays)")
    x_key, y_key = scan_keys
    x_values, y_values = np.array(param_dict[x_key]), np.array(param_dict[y_key])

    # Base parameters that can be updated
    params_dyn_base = {"M_ej": 0.5e-2, "v_ej": 0.25, "tau": 1e-3, "r": r_ref_pc, "start": -1e-3}
    params_wind_base = {"M_ej": 0.01, "v_ej": 0.05, "tau": 10.0, "r": r_ref_pc, "start": 0.01}
    params_grb_base = {"E_grb": 3e46, "theta": np.deg2rad(32), "phi_ej": 0.0, "r": r_ref_pc, "beta": 0.99, "end_grb": 2.0, "E_aft": 10**(52.2), "theta_j": np.deg2rad(5)}

    all_params_base = [params_dyn_base, params_wind_base, params_grb_base]
    
    # Update base parameters with any fixed values provided
    fixed_params = {k: v for k, v in fixed_params_base.items() if k not in scan_keys}
    for k, v in fixed_params.items():
        for p in all_params_base:
            if k in p: p[k] = v
        if k == "r": params_dyn_base["r"] = params_wind_base["r"] = params_grb_base["r"] = v

    snr_grid = np.zeros((len(y_values), len(x_values)))
    f_common = np.logspace(-4, 5, 50000)

    for i, y_val in tqdm(enumerate(y_values), total=len(y_values), desc=f"Grid {y_key}"):
        for j, x_val in enumerate(x_values):
            p_dyn, pw, pg = params_dyn_base.copy(), params_wind_base.copy(), params_grb_base.copy()
            
            current_params = {x_key: x_val, y_key: y_val}
            for k, v in current_params.items():
                if k == "M_ej_dyn": p_dyn["M_ej"] = v
                elif k == "v_ej_dyn": p_dyn["v_ej"] = v
                elif k == "M_ej_wind": pw["M_ej"] = v
                elif k == "v_ej_wind": pw["v_ej"] = v
                elif k in p_dyn: p_dyn[k] = v
                elif k in pw: pw[k] = v
                elif k in pg: pg[k] = v
                
                if k == "r":
                    p_dyn["r"] = pw["r"] = pg["r"] = v

            hc_total = compute_total_hc(f_common, p_dyn, pw, pg, memory_components, r_ref_pc, add_dyn, add_wind, add_grb, add_nonlinear, add_kilonova)
            snr_grid[i, j] = compute_snr_etd(f_common, hc_total, f_noise, h_noise)
            
    return snr_grid, x_key, y_key, x_values, y_values

    
    
    
    

if __name__ == "__main__":
    outdir = "results_snr_heatmap_170817"
    os.makedirs(outdir, exist_ok=True)

    memory_components, r_ref_pc = build_memory_components()
    f_noise, h_noise = load_etd_sensitivity()
    print("Memory components and ET-D sensitivity loaded. Starting SNR computations...")
    
  
    
    # --- New combined plot execution ---
    outdir_combined = "combined_snr_heatmap_170817_v2"
    print(f"\nStarting combined plot computation, saving to {outdir_combined}...")

    # Define global and zoom parameter ranges
    param_global = {
        "r": np.logspace(6, 9, 125),
        "E_grb": np.logspace(45, 60, 125)
    
    }
    param_zoom = { 
        "r": np.logspace(7, 8.5, 125), 
        "E_grb": np.logspace(46, 53, 125)
    }
    

    plot_combined_heatmap_snr(
        param_dict_global=param_global,
        param_dict_zoom=param_zoom,
        outdir=outdir_combined,
        memory_components=memory_components,
        r_ref_pc=r_ref_pc,
        f_noise=f_noise,
        h_noise=h_noise,
        xscale="log",
        yscale="log",
        levels=(0.1,1, 10, 100),
        filename_tag="Egrb_vs_dist",
        if_170817_marker_all=True,
        if_170817_marker_zoom=True, 
        zoom_components={"grb": True, "dyn": False, "wind": False, "nonlinear": False, "kilonova": False}
    )
    
    # Same with the dyn component
    param_global_dyn = {
        "M_ej_dyn": np.logspace(-5.1, -1, 125),
        "v_ej_dyn": np.linspace(0.01, 0.99, 125)
    }
    param_zoom_dyn = {
        "M_ej_dyn": np.logspace(-4.1, -2, 125),
        "v_ej_dyn": np.linspace(0.09, 0.4, 125)
    }
    plot_combined_heatmap_snr(
        param_dict_global=param_global_dyn,
        param_dict_zoom=param_zoom_dyn,
        outdir=os.path.join(outdir, "combined_dyn"),
        memory_components=memory_components,
        r_ref_pc=r_ref_pc,
        f_noise=f_noise,
        h_noise=h_noise,
        xscale="log",
        yscale="linear",
        levels=(0.1,1, 10, 100),
        filename_tag="dyn_memory",
        if_170817_marker_all=True,
        if_170817_marker_zoom=True,
        zoom_components={"dyn": True, "grb": False, "wind": False, "nonlinear": False, "kilonova": False}
    )
    # E_grb and E_aft
    param_global_grb = {
        "E_grb": np.logspace(44.9, 60, 125),
        "E_aft": np.logspace(44.9, 60, 125)
    }
    param_zoom_grb = {
        "E_grb": np.logspace(45.9, 53, 125),
        "E_aft": np.logspace(49.9, 56, 125)
    }
    plot_combined_heatmap_snr(
        param_dict_global=param_global_grb,
        param_dict_zoom=param_zoom_grb,
        outdir=os.path.join(outdir, "combined_grb"),
        memory_components=memory_components,
        r_ref_pc=r_ref_pc,
        f_noise=f_noise,
        h_noise=h_noise,
        xscale="log",
        yscale="log",
        levels=(0.1, 1, 10, 100),
        filename_tag="grb_afterglow",
        if_170817_marker_all=True,
        if_170817_marker_zoom=True,
        zoom_components={"grb": True, "dyn": False, "wind": False, "nonlinear": False, "kilonova": False}
    )
    # Duration of the GRB vs E_aft
    param_global_grb_dur = {
        "end_grb": np.logspace(0., 5, 125),
        "E_aft": np.logspace(44.9, 60, 250)
    }
    param_zoom_grb_dur = {
        "end_grb": np.logspace(0., 2.4, 125),  # Correspond à 1 s à 250 s, couvrant la durée typique des GRBs courts et longs
        "E_aft": np.logspace(45.9, 56, 125)
    }
    plot_combined_heatmap_snr(
        param_dict_global=param_global_grb_dur,
        param_dict_zoom=param_zoom_grb_dur,
        outdir=os.path.join(outdir, "combined_grb_duration"),
        memory_components=memory_components,
        r_ref_pc=r_ref_pc,
        f_noise=f_noise,
        h_noise=h_noise,
        xscale="log",
        yscale="log",
        levels=(0.1,1, 10, 100),
        filename_tag="grb_duration_vs_afterglow",
        if_170817_marker_all=True,
        if_170817_marker_zoom=True,
        zoom_components={"grb": True, "dyn": False, "wind": False, "nonlinear": False, "kilonova": False}
    )
    # wind parameters
    param_global_wind = {
        "M_ej_wind": np.logspace(-4.1, -1, 125),
        "v_ej_wind": np.linspace(0.01, 0.9, 125)
    }
    param_zoom_wind = {
        "M_ej_wind": np.logspace(-3.1, -2, 125),
        "v_ej_wind": np.linspace(0.9, 2.1, 125)
    }
    plot_combined_heatmap_snr(
        param_dict_global=param_global_wind,
        param_dict_zoom=param_zoom_wind,
        outdir=os.path.join(outdir, "combined_wind"),
        memory_components=memory_components,
        r_ref_pc=r_ref_pc,
        f_noise=f_noise,
        h_noise=h_noise,
        xscale="log",
        yscale="linear",
        levels=(0.1,1, 10, 100),
        filename_tag="wind_memory",
        if_170817_marker_all=True,
        if_170817_marker_zoom=True,
        zoom_components={"dyn": False, "grb": False, "wind": True, "nonlinear": False, "kilonova": False}
    )

    print("All computations finished.")