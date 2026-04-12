import os
import warnings

import corner
import emcee
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import joblib  # Ajout de joblib pour charger les modèles KDE

import LISA as li
import utils_GRB_afterglow_phenom as u_grb

warnings.filterwarnings("ignore")

# --- NOUVEAU: Configuration des modèles et de la physique ---
# Chemin vers le dossier contenant les modèles KDE sauvegardés
KDE_MODEL_DIR = '/home/stu_brabant/Memory_effect/test_pop_grb/'

# Angle d'ouverture du jet en radians (5 degrés)
THETA_J_RAD = np.deg2rad(5.0)

# Facteur de correction pour l'énergie (beaming factor)
F_BEAM = 1.0 - np.cos(THETA_J_RAD)

# --- NOUVEAU: Chargement des modèles KDE ---
try:
    kde_Eiso = joblib.load(os.path.join(KDE_MODEL_DIR, 'kde_Eiso.joblib'))
    kde_DL = joblib.load(os.path.join(KDE_MODEL_DIR, 'kde_D_L.joblib'))
    kde_t90 = joblib.load(os.path.join(KDE_MODEL_DIR, 'kde_t90.joblib'))
    KDE_MODELS = {
        "logE_grb": kde_Eiso,
        "logr": kde_DL,
        "logend_grb": kde_t90,
    }
    print("Modèles KDE chargés avec succès.")
except FileNotFoundError as e:
    print(f"Erreur: Fichier de modèle KDE non trouvé. Vérifiez le chemin dans KDE_MODEL_DIR.")
    print(f"Détail de l'erreur: {e}")
    KDE_MODELS = {} # Garde le dictionnaire vide pour que le script ne plante pas plus tard


def compute_hc_grb_only(params_grb):
    h_in = u_grb.memory_initial_acceleration_GRB(
        params_grb["E_grb"],
        params_grb["theta"],
        params_grb["phi_ej"],
        params_grb["r"],
        params_grb["beta"],
    )

    h_aft = u_grb.memory_afterglow(
        params_grb["E_aft"],
        params_grb["theta"],
        params_grb["phi_ej"],
        params_grb["r"],
        params_grb["beta"],
    )

    f_grb, fft_grb_aft = u_grb.total_waveform_fft(
        h_in,
        h_aft,
        params_grb["end_grb"],
        params_grb["theta"],
        params_grb["r"],
        radius=0,
    )

    hc_grb = 2.0 * f_grb * np.abs(fft_grb_aft)
    mask = (f_grb >= 1e-4) & (f_grb <= 1e-1)
    return f_grb[mask], hc_grb[mask]


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

# Very broad uniform prior bounds
PARAM_BOUNDS = {
    "logE_grb": (40.0, 60.0),
    "logE_aft": (40.0, 60.0),
    "logr": (2.0, 9.5), # Ajusté pour correspondre aux données du KDE
    "logend_grb": (-2.0, 6.0), # Ajusté pour correspondre aux données du KDE
    "beta": (0.0, 0.999999),
    "theta": (0.0, np.pi / 2.0), # Limité à pi/2 car le modèle est symétrique
}
PARAM_ORDER = ["logE_grb", "logE_aft", "logr", "logend_grb", "beta", "theta"]

# Gaussian physical preference (pour les paramètres SANS KDE)
PHYSICAL_MEAN = {
    "logE_aft": 52.0,
    "beta": 0.99,
    "theta": np.deg2rad(10.0),
}
PHYSICAL_SIGMA = {
    "logE_aft": 2.5,
    "beta": 0.15,
    "theta": 0.6,
}

# MODIFIÉ: La fonction utilise maintenant le facteur de correction F_BEAM
def params_from_vector(x):
    logE_grb, logE_aft, logr, logend_grb, beta, theta = x
    return {
        # E_grb et E_aft sont les énergies *collimatées* utilisées pour la physique
        "E_grb": (10**logE_grb) * F_BEAM,
        "E_aft": (10**logE_aft) * F_BEAM,
        "r": 10**logr,
        "end_grb": 10**logend_grb,
        "beta": beta,
        "theta": theta,
        "phi_ej": 0.0,
    }


def log_prior(x):
    logE_grb, logE_aft, logr, logend_grb, beta, theta = x
    # Utilise les bornes de PARAM_BOUNDS pour vérifier si les paramètres sont valides
    for i, param_name in enumerate(PARAM_ORDER):
        if not (PARAM_BOUNDS[param_name][0] <= x[i] <= PARAM_BOUNDS[param_name][1]):
            return -np.inf
    return 0.0

# NOUVEAU: Fonction de vraisemblance physique hybride (KDE + Gaussienne)
def log_physical_kde_and_gaussian(x):
    if not KDE_MODELS: # Si les modèles n'ont pas été chargés
        print("Avertissement: Les modèles KDE n'ont pas été chargés. La vraisemblance physique ne sera pas calculée.")
        return -np.inf

    logE_grb, logE_aft, logr, logend_grb, beta, theta = x
    
    # 1. Termes calculés avec les KDEs
    # Note: les KDEs attendent une entrée de type [[valeur]], d'où le .reshape(1, -1)
    ll_e_grb = KDE_MODELS["logE_grb"].score_samples(np.array([[logE_grb]]))[0]
    ll_r = KDE_MODELS["logr"].score_samples(np.array([[logr]]))[0]
    ll_end_grb = KDE_MODELS["logend_grb"].score_samples(np.array([[logend_grb]]))[0]

    # 2. Termes calculés avec les priors Gaussiens restants
    ll_e_aft = -0.5 * ((logE_aft - PHYSICAL_MEAN["logE_aft"]) / PHYSICAL_SIGMA["logE_aft"]) ** 2
    ll_beta = -0.5 * ((beta - PHYSICAL_MEAN["beta"]) / PHYSICAL_SIGMA["beta"]) ** 2
    ll_theta = -0.5 * ((theta - PHYSICAL_MEAN["theta"]) / PHYSICAL_SIGMA["theta"]) ** 2
    
    return ll_e_grb + ll_r + ll_end_grb + ll_e_aft + ll_beta + ll_theta


def log_likelihood_snr_only(x, lisa):
    # Les paramètres isotropes sont passés à params_from_vector, qui les corrige
    params = params_from_vector(x)
    try:
        f, hc = compute_hc_grb_only(params)
        noise = np.sqrt(f * np.abs(lisa.Sn(f)))
        snr = compute_snr(f, hc, f, noise)
    except Exception:
        return -np.inf

    if not np.isfinite(snr):
        return -np.inf

    delta = max(0.0, SNR_THRESHOLD - snr)
    ll_detect = -0.5 * (delta / SNR_SIGMA) ** 2
    return ll_detect

# MODIFIÉ: Utilise la nouvelle fonction de vraisemblance physique
def log_likelihood_snr_and_physical(x, lisa):
    ll_detect = log_likelihood_snr_only(x, lisa)
    if not np.isfinite(ll_detect):
        return -np.inf
    
    # Appelle la nouvelle fonction hybride
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
    # Le centre est basé sur les moyennes des priors Gaussiens
    # Pour les KDEs, on pourrait utiliser la médiane des données d'entraînement, mais la moyenne gaussienne reste un bon point de départ.
    center = np.array([
        52.0, PHYSICAL_MEAN["logE_aft"], 
        8.0, 2.0, 
        PHYSICAL_MEAN["beta"], PHYSICAL_MEAN["theta"]
    ])
    scale = np.array([0.7, 0.7, 0.4, 0.4, 0.03, 0.2])
    p0 = center + scale * rng.normal(size=(nwalkers, len(center)))

    # Assure que les marcheurs démarrent dans les bornes
    for i, param_name in enumerate(PARAM_ORDER):
        p0[:, i] = np.clip(p0[:, i], *PARAM_BOUNDS[param_name])
        
    return p0


def run_emcee_sampler(likelihood_func, nwalkers=30, nsteps=500, burnin=200, seed=42):
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

    # Calcule les énergies isotropes pour la sauvegarde
    df_post["E_iso_grb"] = 10**df_post["logE_grb"]
    df_post["E_iso_aft"] = 10**df_post["logE_aft"]
    df_post["r_mpc"] = 10**df_post["logr"]
    df_post["t90"] = 10**df_post["logend_grb"]

    lisa = li.LISA()
    snr_list = []
    for _, row in df_post.iterrows():
        # Recrée le vecteur de paramètres pour le calcul du SNR
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

# Fonction pour tracer les distributions a priori vs a posteriori
def plot_prior_vs_posterior(df_post, outdir, corner_cols, corner_labels):
    print(f"Generating prior vs posterior plots in {outdir}...")
    
    for i, col_name in enumerate(corner_cols):
        fig, ax = plt.subplots(figsize=(8, 6))

        # 1. Trace l'histogramme de la distribution a posteriori
        posterior_data = df_post[col_name].to_numpy()
        ax.hist(posterior_data, bins='auto', density=True, alpha=0.7, color='cornflowerblue', label='Posterior')

        # 2. Trace la distribution a priori par-dessus
        x_grid = np.linspace(posterior_data.min(), posterior_data.max(), 500)
        
        if col_name in KDE_MODELS:
            # C'est un prior KDE
            kde_prior = KDE_MODELS[col_name]
            log_density = kde_prior.score_samples(x_grid.reshape(-1, 1))
            density = np.exp(log_density)
            ax.plot(x_grid, density, color='darkorange', lw=2.5, label='Prior (KDE)')
        elif col_name in PHYSICAL_MEAN:
            # C'est un prior Gaussien
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

    # MODIFIÉ: La configuration utilise la nouvelle fonction de vraisemblance
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
    corner_cols = ["logE_grb", "logE_aft", "logr", "logend_grb", "beta", "theta"]
    corner_labels = [
        r"$\log_{10}(E_{\rm iso, GRB})$", r"$\log_{10}(E_{\rm iso, aft})$",
        r"$\log_{10}(D_L / \rm{Mpc})$", r"$\log_{10}(T_{90} / s)$",
        r"$\beta$", r"$\theta$",
    ]

    # --- Individual Plots ---
    print("\nGenerating individual plots for each case...")
    for case_name, data in results_data.items():
        df_post = data["df"]
        case_outdir = data["outdir"]
        color = likelihood_cases[case_name]["color"]
        
        # Individual Corner Plot
        corner_samples = df_post[corner_cols].to_numpy()
        fig_corner_ind = corner.corner(
            corner_samples, labels=corner_labels, bins=35, color=color,
            show_titles=True, title_fmt=".3f", quantiles=[0.16, 0.50, 0.84],
            smooth=1.0, plot_datapoints=False, fill_contours=True
        )
        fig_corner_ind.savefig(os.path.join(case_outdir, "corner_posterior.png"), dpi=300, bbox_inches="tight")
        plt.close(fig_corner_ind)

        # NOUVEAU: Appel de la fonction de tracé Prior vs Posterior
        if case_name == "snr_and_physical":
            plot_prior_vs_posterior(df_post, case_outdir, corner_cols, corner_labels)

    print("Individual plots saved.")

    # --- Combined Plots ---
    print("\nGenerating combined plots...")
    fig_corner_comb = None
    for i, (case_name, data) in enumerate(results_data.items()):
        df_post = data["df"]
        color = likelihood_cases[case_name]["color"]
        corner_samples = df_post[corner_cols].to_numpy()
        
        if fig_corner_comb is None:
            fig_corner_comb = corner.corner(
                corner_samples, labels=corner_labels, bins=35, color=color,
                show_titles=False, # Les titres sont moins utiles en comparaison
                quantiles=[0.16, 0.50, 0.84],
                smooth=1.0, plot_datapoints=False, fill_contours=True
            )
        else:
            corner.corner(
                corner_samples, fig=fig_corner_comb, bins='auto', color=color,
                smooth=1.0, plot_datapoints=False, fill_contours=True
            )

    legend_elements = [mpatches.Patch(facecolor=case_info["color"], label=case_info["label"])
                       for case_info in likelihood_cases.values()]
    fig_corner_comb.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    fig_corner_comb.savefig(os.path.join(base_outdir, "corner_posterior_combined.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_corner_comb)

    print(f"All combined plots have been saved in: {base_outdir}")
