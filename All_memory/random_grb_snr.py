import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18
import astropy.units as u
import utils_signal as usig
import utils_GRB_afterglow_phenom as u_grb
import LISA as li
from scipy.interpolate import interp1d
import os
import warnings
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings("ignore")

from scipy.stats import qmc

def random_sample_params_qmc(n_samples, sampler_type="sobol", seed=42):
    # 8 dimensions: [log10E_grb, theta, phi_ej, logr, beta, log10end_grb, log10E_aft, theta_j]
    d = 8

    if sampler_type == "sobol":
        # Sobol fonctionne idéalement avec 2^m points
        m = int(np.ceil(np.log2(n_samples)))
        sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
        u = sampler.random_base2(m=m)[:n_samples]
    elif sampler_type == "halton":
        sampler = qmc.Halton(d=d, scramble=True, seed=seed)
        u = sampler.random(n=n_samples)
    else:
        raise ValueError("sampler_type doit être 'sobol' ou 'halton'")

    # u in [0,1]^8 -> variables "scaled" in their target ranges
    # set the lower and upper bounds for each parameter
    l_bounds = [40, 0, 0, 7, 0.1, 0, 40, 0] 
    u_bounds = [58, np.pi, 2*np.pi, 9, 0.9999, 6, 58, np.pi/2]

    scaled = qmc.scale(u, l_bounds, u_bounds)

    logE_grb  = scaled[:, 0]
    theta     = scaled[:, 1]
    phi_ej    = scaled[:, 2]
    logr      = scaled[:, 3]
    beta      = scaled[:, 4]
    logend_grb= scaled[:, 5]
    logE_aft  = scaled[:, 6]
    theta_j   = scaled[:, 7]

    E_grb   = 10**logE_grb
    r       = 10**logr
    end_grb = 10**logend_grb
    E_aft   = 10**logE_aft

    params_list = []
    for i in range(n_samples):
        params_list.append({
            "E_grb": E_grb[i],
            "theta": theta[i],
            "phi_ej": phi_ej[i],
            "r": r[i],
            "beta": beta[i],
            "end_grb": end_grb[i],
            "E_aft": E_aft[i],
            "theta_j": theta_j[i],
        })
    return params_list

def plot_hexbin_parameter_space(params_list, outdir, gridsize=30, cmap="viridis"):
    os.makedirs(outdir, exist_ok=True)

    # DataFrame + variables en log demandées
    df = pd.DataFrame(params_list).copy()
    df["log10E_grb"] = np.log10(df["E_grb"])
    df["logr"] = np.log10(df["r"])
    df["log10end_grb"] = np.log10(df["end_grb"])
    df["log10E_aft"] = np.log10(df["E_aft"])

    # Paramètres demandés
    cols = ["log10E_grb", "theta", "phi_ej", "logr", "beta", "log10end_grb", "log10E_aft"]
    n = len(cols)

    fig, axes = plt.subplots(n, n, figsize=(2.6 * n, 2.4 * n))
    hb_for_cbar = None

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]

            if i == j:
                # Diagonale: marginales 1D
                ax.hist(
                    df[cols[i]].values,
                    bins="auto",
                    density=True,
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=0.5,
                    color="steelblue"
                )
            elif i > j:
                # Triangle inférieur: hexbin 2D
                hb = ax.hexbin(
                    df[cols[j]].values,
                    df[cols[i]].values,
                    gridsize=gridsize,
                    mincnt=1,
                    cmap=cmap
                )
                hb_for_cbar = hb
            else:
                # Triangle supérieur: vide pour éviter surcharge visuelle
                ax.axis("off")
                continue

            # Labels seulement en bordure pour lisibilité
            if i == n - 1:
                ax.set_xlabel(cols[j])
            else:
                ax.set_xticklabels([])

            if j == 0:
                ax.set_ylabel(cols[i])
            else:
                ax.set_yticklabels([])

    # Colorbar globale (densité hexbin)
    if hb_for_cbar is not None:
        cbar = fig.colorbar(hb_for_cbar, ax=axes, fraction=0.02, pad=0.01)
        cbar.set_label("Counts per hexagon")

    fig.suptitle("Hexbin 2D parameter-space coverage (QMC sampling)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "sampling_hexbin_matrix_7params.png"), dpi=300, bbox_inches="tight")
    plt.show()

def compute_hc_grb_only(params_grb):
    h_in = u_grb.memory_initial_acceleration_GRB(params_grb['E_grb'], params_grb['theta'], params_grb['phi_ej'], params_grb['r'], params_grb['beta'])
    # Use afterglow memory directly
    h_aft = u_grb.memory_afterglow(params_grb['E_aft'], params_grb['theta'], params_grb['phi_ej'], params_grb['r'], params_grb['beta'])
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

    interp_hn = interp1d(f_noise, h_n, kind='linear', bounds_error=False)
    h_n_interp = interp_hn(f_signal)

    integrand = (h_c**2) / (h_n_interp**2)
    snr_squared = np.trapezoid(integrand, x=np.log(f_signal))
    snr = np.sqrt(snr_squared)
    return snr

def random_sample_params(n_samples):
    # Define ranges based on realistic GRB properties
    # E_grb:
    E_grb = 10 ** np.random.uniform(40, 58, n_samples)
    # theta: 0 to 180 deg
    theta = np.random.uniform(0, np.pi, n_samples)
    # phi_ej: 0 to 2pi
    phi_ej = np.random.uniform(0, 2*np.pi, n_samples)
    # r: distance in pc, from 1e7 to 1e9 pc (10 Mpc to 1 Gpc)
    r = 10 ** np.random.uniform(7, 9, n_samples)
    # beta: 0.1 to 0.9999
    beta = np.random.uniform(0.1, 0.9999, n_samples)
    # end_grb: duration 1 to 1e6 s
    end_grb = 10 ** np.random.uniform(0, 6, n_samples)
    # E_aft: total energy in afterglow
    E_aft = 10 ** np.random.uniform(40, 58, n_samples)
    # theta_j: 0 to 90 deg (though not used now) - change formula ? 
    theta_j = np.random.uniform(0, np.pi/2, n_samples)
    
    params_list = []
    for i in range(n_samples):
        params = {
            'E_grb': E_grb[i],
            'theta': theta[i],
            'phi_ej': phi_ej[i],
            'r': r[i],
            'beta': beta[i],
            'end_grb': end_grb[i],
            'E_aft': E_aft[i],
            'theta_j': theta_j[i]
        }
        params_list.append(params)
    return params_list



def main(n_samples):
    params_list = random_sample_params_qmc(n_samples, sampler_type="sobol", seed=42)
    

    snr_list = []
    valid_params = []

    lisa = li.LISA()

    total = len(params_list)
    last_percent = -1

    with tqdm(total=100, desc="Computing SNR", bar_format="{desc}: {n:3.0f}%") as pbar:
        for i, params in enumerate(params_list, start=1):
            try:
                f, hc = compute_hc_grb_only(params)
                snr = compute_snr(f, hc, f, np.sqrt(f * np.abs(lisa.Sn(f))))
                if np.isfinite(snr):
                    params_copy = params.copy()
                    params_copy["snr"] = snr
                    snr_list.append(snr)
                    valid_params.append(params_copy)
            except:
                pass

            percent = (i * 100) // total
            if percent > last_percent:
                pbar.update(percent - last_percent)
                last_percent = percent
    pbar.close()
    
    snr_array = np.array(snr_list)
    
        # Save directory
    outdir = 'results_random_snr'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # save in CSV param + SNR for all valid configurations
    df_valid = pd.DataFrame(valid_params)
    df_valid.to_csv(os.path.join(outdir, 'valid_configs_with_snr.csv'), index=False)

    threshold = 10.0

    # Find max SNR
    max_snr = np.max(snr_array) if len(snr_array) > 0 else np.nan
    print(f"Max SNR: {max_snr}")
    print(f"Threshold (10 SNR): {threshold}")

    # Physical realism criteria
    realistic_E = (1e47, 1e55)         # min = GRB 170817A, max = BOAT
    realistic_r_mpc = (10, 10000)      # 10 Mpc to 10 Gpc
    realistic_beta = (0.9, 0.9999)
    realistic_end_grb = (1, 1000)
    realistic_E_aft = (1e50, 1e56)

    # Build realistic mask on all valid configurations (not only high SNR)
    realistic_mask = np.array([
        (realistic_E[0] <= p['E_grb'] <= realistic_E[1]) and
        (realistic_r_mpc[0] * 1e6 <= p['r'] <= realistic_r_mpc[1] * 1e6) and
        (realistic_beta[0] <= p['beta'] <= realistic_beta[1]) and
        (realistic_end_grb[0] <= p['end_grb'] <= realistic_end_grb[1]) and
        (realistic_E_aft[0] <= p['E_aft'] <= realistic_E_aft[1])
        for p in valid_params
    ], dtype=bool)

    snr_realistic = snr_array[realistic_mask]
    snr_non_physical = snr_array[~realistic_mask]

    print(f"Number of realistic configurations (all): {realistic_mask.sum()}")
    print(f"Number of non-physical configurations (all): {(~realistic_mask).sum()}")

    # Find configurations above threshold
    above_threshold = snr_array > threshold
    print(f"Number of configurations above threshold: {above_threshold.sum()}")

    # Realistic + high SNR configurations for export
    realistic_high_snr_mask = realistic_mask & above_threshold
    realistic_high_snr_params = [valid_params[i] for i in range(len(valid_params)) if realistic_high_snr_mask[i]]

    # Common bins for comparable overlays
    bins = 'auto'

    # Plot 1: realistic configurations only
    plt.figure(figsize=(8, 6))
    if len(snr_realistic) > 0:
        plt.hist(
            snr_realistic,
            bins=bins,
            density=False,
            alpha=0.95,
            edgecolor='white',
            color='forestgreen',
        )
    plt.axvline(x=threshold, color='k', linestyle='--')
    plt.xlabel('SNR')
    plt.ylabel('Counts')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'snr_distribution_realistic_only.png'), dpi=300)
    plt.show()

    # Plot 2: all configurations with realistic vs non-physical
    plt.figure(figsize=(8, 6))

    plot_max = np.percentile(snr_array, 95)

    snr_non_plot = snr_non_physical[(snr_non_physical >= 0) & (snr_non_physical <= plot_max)]
    snr_real_plot = snr_realistic[(snr_realistic >= 0) & (snr_realistic <= plot_max)]

    # Réduire drastiquement les bins
    bins = np.linspace(0, plot_max, 20)

    plt.hist(
        [snr_non_plot, snr_real_plot],
        bins=bins,
        density=False,  
        alpha=0.8,
        edgecolor="white",
        linewidth=0.3,
        color=["lightcoral", "forestgreen"],
        label=["Non-physical", "Realistic"],
        histtype="bar",
        stacked=True # pour empiler les histogrammes et mieux visualiser les proportions
    )

    plt.axvline(x=threshold, color="k", linestyle="--", linewidth=1.5)
    plt.xlim(0, plot_max)
    plt.xlabel("SNR", fontsize=12)
    plt.ylabel("Count", fontsize=12)  # <-- CHANGE: pas "Density"
    plt.legend(loc="upper right", fontsize=12, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "snr_distribution_all_realistic_vs_nonphysical.png"), dpi=300)
    plt.show()

    # Save realistic + high SNR configs
    if len(realistic_high_snr_params) > 0:
        print("Yes, there are realistic configurations with high SNR.")
        df_realistic = pd.DataFrame(realistic_high_snr_params)
        print(df_realistic)
        df_realistic.to_csv(os.path.join(outdir, 'realistic_high_snr_configs.csv'), index=False)
    else:
        print("No realistic configurations found above the threshold.")
    
    
    

if __name__ == "__main__":
    n_samples = 200_000 
    main(n_samples)