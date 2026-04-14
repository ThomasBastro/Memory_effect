import os
import warnings

import corner
import emcee
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import joblib

import LISA as li
import utils_grb_aft_new as u_grb

warnings.filterwarnings("ignore")

# -- Compute Min LISA sensitivity for Gaussian prior on t_dec and t_jet_break ---
def compute_lisa_min_sensitivity():
    lisa = li.LISA()
    f = np.logspace(-5, 0, 10000)
    Sn = lisa.Sn(f)
    min_index = np.argmin(Sn)
    return f[min_index]

# --- KDE Models ---
KDE_MODEL_DIR = '/home/stu_brabant/Memory_effect/test_pop_grb/'
try:
    kde_Eiso = joblib.load(os.path.join(KDE_MODEL_DIR, 'kde_Eiso.joblib'))
    kde_DL = joblib.load(os.path.join(KDE_MODEL_DIR, 'kde_D_L.joblib'))
    kde_t90 = joblib.load(os.path.join(KDE_MODEL_DIR, 'kde_t90.joblib'))
    KDE_MODELS = {
        "logE_grb": kde_Eiso,
        "logr": kde_DL,
        "logT90": kde_t90,
    }
    print("Modèles KDE chargés avec succès.")
except FileNotFoundError as e:
    print(f"Erreur: Fichier de modèle KDE non trouvé. Vérifiez le chemin dans KDE_MODEL_DIR.")
    print(f"Détail de l'erreur: {e}")
    KDE_MODELS = {}

# --- PARAMS ---
PARAM_BOUNDS = {
    "logE_grb": (40.0, 60.0),
    "logE_aft": (40.0, 60.0),
    "logr": (2.0, 9.5),
    "logT90": (-2.0, 6.0),
    "theta": (0.0, np.pi / 2.0),
    "logt_dec": (0.0, 4.0),           # t_dec in s
    "logt_jet_break": (0.0, 2.5),     # t_jet_break in days
}
PARAM_ORDER = ["logE_grb", "logE_aft", "logr", "logT90", "theta", "logt_dec", "logt_jet_break"]

PHYSICAL_MEAN = {
    "logE_aft": 52.0,
    "theta": np.deg2rad(10.0),
}
PHYSICAL_SIGMA = {
    "logE_aft": 2.5,
    "theta": 0.6,
}

def params_from_vector(x):
    logE_grb, logE_aft, logr, logT90, theta, logt_dec, logt_jet_break = x
    return {
        "E_grb": 10**logE_grb,
        "E_aft": 10**logE_aft,
        "r": 10**logr,
        "T_90": 10**logT90,
        "theta": theta,
        "t_dec": 10**logt_dec,
        "t_jet_break": 10**logt_jet_break * 86400,  # jours -> secondes
        "theta_j": np.deg2rad(5.0),
        "phi_ej": 0.0,
        "beta": 0.99,
    }

def compute_hc_grb_only(params_grb):
    t = np.linspace(0, params_grb["t_jet_break"] * 1.2, 1000)
    delta_h, delta_h_GRB, delta_h_aft = u_grb.grb_afterglow_model(
        t,
        params_grb["E_grb"],
        params_grb["T_90"],
        params_grb["E_aft"],
        params_grb["beta"],
        params_grb["t_dec"],
        params_grb["t_jet_break"],
        params_grb["r"],
        params_grb["theta"],
        params_grb["theta_j"],
    )
    f = np.logspace(-5, 0, 50000)
    h_ft, _, _ = u_grb.fft(
        f,
        delta_h_GRB,
        params_grb["T_90"],
        delta_h_aft,
        params_grb["t_dec"],
        params_grb["t_jet_break"],
    )
    hc_grb = 2.0 * f * np.abs(h_ft)
    mask = (f >= 1e-4) & (f <= 1e-1)
    return f[mask], hc_grb[mask]

def compute_snr(f_signal, h_c, f_noise, h_n):
    f_signal = np.asarray(f_signal)
    h_c = np.asarray(h_c)
    f_noise = np.asarray(f_noise)
    h_n = np.asarray(h_n)
    interp_hn = interp1d(
        f_noise,
        h_n,
        kind="linear",
        bounds_error=False,
        fill_value=np.inf,
    )
    h_n_interp = interp_hn(f_signal)
    integrand = (h_c**2) / (h_n_interp**2)
    snr_squared = np.trapezoid(integrand, x=np.log(f_signal))
    return np.sqrt(snr_squared)

# Detection settings
SNR_THRESHOLD = 10.0
SNR_SIGMA = 5.0

def log_prior(x):
    for i, param_name in enumerate(PARAM_ORDER):
        if not (PARAM_BOUNDS[param_name][0] <= x[i] <= PARAM_BOUNDS[param_name][1]):
            return -np.inf
    return 0.0

def log_physical_kde_and_gaussian(x):
    if not KDE_MODELS:
        print("Avertissement: Les modèles KDE n'ont pas été chargés. La vraisemblance physique ne sera pas calculée.")
        return -np.inf
    logE_grb, logE_aft, logr, logT90, theta, logt_dec, logt_jet_break = x
    ll_e_grb = KDE_MODELS["logE_grb"].score_samples(np.array([[logE_grb]]))[0]
    ll_r = KDE_MODELS["logr"].score_samples(np.array([[logr]]))[0]
    ll_T90 = KDE_MODELS["logT90"].score_samples(np.array([[logT90]]))[0]
    ll_e_aft = -0.5 * ((logE_aft - PHYSICAL_MEAN["logE_aft"]) / PHYSICAL_SIGMA["logE_aft"]) ** 2
    # uniforme sur E_aft entre 1e50 et 1e56 erg
    if 50.0 <= logE_aft <= 56.0:
        ll_e_aft = 0.0
    else:
        ll_e_aft = -np.inf
  
    # On a donc : kde sur E_grb r, T90 + unif sur E_aft
    return ll_e_grb + ll_r + ll_T90 + ll_e_aft 

def log_likelihood_snr_only(x, lisa, threshold=10.0, width=2.0, alpha=0.2):
    params = params_from_vector(x)
    try:
        f, hc = compute_hc_grb_only(params)
        noise = np.sqrt(f * np.abs(lisa.Sn(f)))
        snr = compute_snr(f, hc, f, noise)
    except Exception:
        return -np.inf
    if not np.isfinite(snr):
        return -np.inf
    # softplus(x) = log(1 + exp(x))
    softplus = np.log(1 + np.exp((snr - threshold) / width))
    return alpha * softplus

def log_likelihood_snr_and_physical(x, lisa, threshold=10.0, width=2.0, alpha=0.2):
    ll_detect = log_likelihood_snr_only(x, lisa, threshold, width, alpha)
    if not np.isfinite(ll_detect):
        return -np.inf
    ll_physical = log_physical_kde_and_gaussian(x)
    return ll_detect + ll_physical

def log_probability(x, lisa, likelihood_func):
    lp = log_prior(x)
    if not np.isfinite(lp):
        return -np.inf
    ll = likelihood_func(x, lisa)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

def initialize_walkers(nwalkers, seed=42):
    rng = np.random.default_rng(seed)
    center = np.array([
        52.0, 52.0, 8.0, 2.0, np.deg2rad(10.0), 2.0, 1.0
    ])
    scale = np.array([0.7, 0.7, 0.4, 0.4, 0.2, 0.5, 0.5])
    p0 = center + scale * rng.normal(size=(nwalkers, len(center)))
    for i, param_name in enumerate(PARAM_ORDER):
        p0[:, i] = np.clip(p0[:, i], *PARAM_BOUNDS[param_name])
    return p0

def run_emcee_sampler(likelihood_func, nwalkers=30, nsteps=800, burnin=300, seed=42):
    lisa = li.LISA()
    ndim = len(PARAM_ORDER)
    p0 = initialize_walkers(nwalkers, seed=seed)
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(lisa, likelihood_func)
    )
    print(f"Running for likelihood: {likelihood_func.__name__}")
    print("Burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, burnin, progress=True)
    sampler.reset()
    print("Production...")
    sampler.run_mcmc(p0, nsteps, progress=True)
    return sampler

def process_and_save_results(sampler, outdir, case_name):
    print(f"Processing results for case: {case_name}")
    case_outdir = os.path.join(outdir, case_name)
    os.makedirs(case_outdir, exist_ok=True)
    flat_samples = sampler.get_chain(flat=True)
    flat_log_prob = sampler.get_log_prob(flat=True)
    df_post = pd.DataFrame(flat_samples, columns=PARAM_ORDER)
    df_post["log_prob"] = flat_log_prob
    df_post["E_iso_grb"] = 10**df_post["logE_grb"]
    df_post["E_iso_aft"] = 10**df_post["logE_aft"]
    df_post["r_pc"] = 10**df_post["logr"]
    df_post["T90"] = 10**df_post["logT90"]
    df_post["t_dec"] = 10**df_post["logt_dec"]
    df_post["t_jet_break_days"] = 10**df_post["logt_jet_break"]
    lisa = li.LISA()
    snr_list = []
    for _, row in df_post.iterrows():
        x_row = row[PARAM_ORDER].values
        params = params_from_vector(x_row)
        try:
            f, hc = compute_hc_grb_only(params)
            noise = np.sqrt(f * np.abs(lisa.Sn(f)))
            snr = compute_snr(f, hc, f, noise)
        except Exception:
            snr = np.nan
        snr_list.append(snr)
    df_post["snr"] = snr_list
    df_post = df_post[np.isfinite(df_post["snr"]) & np.isfinite(df_post["log_prob"])].copy()
    df_post.to_csv(os.path.join(case_outdir, "posterior_samples.csv"), index=False)
    print(f"Results for '{case_name}' saved in {case_outdir}")
    return df_post, case_outdir

def plot_prior_vs_posterior(df_post, outdir, corner_cols, corner_labels):
    print(f"Generating prior vs posterior plots in {outdir}...")
    for i, col_name in enumerate(corner_cols):
        fig, ax = plt.subplots(figsize=(8, 6))
        posterior_data = df_post[col_name].to_numpy()
        ax.hist(posterior_data, bins='auto', density=True, alpha=0.7, color='cornflowerblue', label='Posterior')
        x_grid = np.linspace(posterior_data.min(), posterior_data.max(), 500)
        if col_name in KDE_MODELS:
            kde_prior = KDE_MODELS[col_name]
            log_density = kde_prior.score_samples(x_grid.reshape(-1, 1))
            density = np.exp(log_density)
            ax.plot(x_grid, density, color='darkorange', lw=2.5, label='Prior (KDE)')
        elif col_name in PHYSICAL_MEAN:
            mean = PHYSICAL_MEAN[col_name]
            sigma = PHYSICAL_SIGMA[col_name]
            density = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_grid - mean) / sigma)**2)
            ax.plot(x_grid, density, color='darkorange', lw=2.5, label='Prior (Gaussian)')
        ax.set_xlabel(corner_labels[i])
        ax.set_ylabel("Density")
        ax.set_title(f"Prior vs Posterior for {col_name}")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        fig.savefig(os.path.join(outdir, f"prior_vs_posterior_{col_name}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
    print("Prior vs posterior plots saved.")

if __name__ == "__main__":
    base_outdir = "results_bayesian_emcee"
    os.makedirs(base_outdir, exist_ok=True)
    likelihood_cases = {
        "snr_only": {
            "func": log_likelihood_snr_only,
            "color": "lightcoral",
            "label": "SNR Threshold Only"
        },
        "snr_and_physical": {
            "func": log_likelihood_snr_and_physical,
            "color": "forestgreen",
            "label": "SNR Threshold + Physical KDE"
        },
    }
    results_data = {}
    for case_name, case_info in likelihood_cases.items():
        sampler = run_emcee_sampler(case_info["func"])
        df_post, case_outdir = process_and_save_results(sampler, base_outdir, case_name)
        results_data[case_name] = {"df": df_post, "outdir": case_outdir}
    # --- Plotting ---
    corner_cols = ["logE_grb", "logE_aft", "logr", "logT90", "theta", "logt_dec", "logt_jet_break"]
    corner_labels = [
        r"$\log_{10}(E_{\rm GRB})$", r"$\log_{10}(E_{\rm aft})$",
        r"$\log_{10}(D_L / \rm{pc})$", r"$\log_{10}(T_{90} / s)$",
        r"$\theta$", r"$\log_{10}(t_{\rm dec}/s)$", r"$\log_{10}(t_{\rm jet}/\mathrm{days})$"
    ]
    print("\nGenerating individual plots for each case...")
    for case_name, data in results_data.items():
        df_post = data["df"]
        case_outdir = data["outdir"]
        color = likelihood_cases[case_name]["color"]
        corner_samples = df_post[corner_cols].to_numpy()
        fig_corner_ind = corner.corner(
            corner_samples, labels=corner_labels, bins=35, color=color,
            show_titles=True, title_fmt=".3f", quantiles=[0.16, 0.50, 0.84],
            smooth=1.0, plot_datapoints=False, fill_contours=True
        )
        fig_corner_ind.savefig(os.path.join(case_outdir, "corner_posterior.png"), dpi=300, bbox_inches="tight")
        plt.close(fig_corner_ind)
        if case_name == "snr_and_physical":
            plot_prior_vs_posterior(df_post, case_outdir, corner_cols, corner_labels)
    print("Individual plots saved.")
    print("\nGenerating combined plots...")
    fig_corner_comb = None
    for i, (case_name, data) in enumerate(results_data.items()):
        df_post = data["df"]
        color = likelihood_cases[case_name]["color"]
        corner_samples = df_post[corner_cols].to_numpy()
        if fig_corner_comb is None:
            fig_corner_comb = corner.corner(
                corner_samples, labels=corner_labels, bins=35, color=color,
                show_titles=False,
                quantiles=[0.16, 0.50, 0.84],
                smooth=1.0, plot_datapoints=False, fill_contours=True
            )
        else:
            corner.corner(
                corner_samples, fig=fig_corner_comb, bins=35, color=color,
                smooth=1.0, plot_datapoints=False, fill_contours=True
            )
    legend_elements = [mpatches.Patch(facecolor=case_info["color"], label=case_info["label"])
                       for case_info in likelihood_cases.values()]
    fig_corner_comb.legend(handles=legend_elements, loc='upper right', fontsize=12)
    fig_corner_comb.savefig(os.path.join(base_outdir, "corner_posterior_combined.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_corner_comb)
    print(f"All combined plots have been saved in: {base_outdir}")