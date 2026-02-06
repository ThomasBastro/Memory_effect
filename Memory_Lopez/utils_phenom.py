import numpy as np
import matplotlib.pyplot as plt
from lal import MSUN_SI, PC_SI, C_SI, G_SI
from matplotlib.ticker import FixedLocator, LogLocator
outdir = "./figures/"

def model_exponential(t, delta_h, tau):
    return delta_h / (1 + np.exp(-t / tau))
def model_tanh(t, delta_h, tau):
    return delta_h * 0.5 * np.tanh(t/tau) + delta_h * 0.5

def phenom_memory_models(tau, M_ej_dyn, v_ej_dyn, r, t, model, plot):
    """
    Compare exponential (Lopez et al.) and tanh (Favata) models for dynamical ejecta memory.

    Parameters:
    -----------
    tau : float
        Characteristic timescale (s)
    M_ej_dyn : float
        Ejecta mass in solar mass
    v_ej_dyn : float
        Ejecta velocity (m/s)
    r : float
        Distance to source (parsecs)
    t : array
        Time array (s)
    G_SI, C_SI, MSUN_SI, PC_SI : float
        Physical constants
    model : str
        Model type ('exponential' or 'tanh')
    plot : bool
        Whether to plot the results
    Returns:
    --------
    t : array
        Time array (s)
    h : array
        Memory strain array of the chossen model
    """
    r = r * PC_SI  # Convert distance from parsecs to meters
    M_ej_dyn = M_ej_dyn * MSUN_SI  # Convert mass from solar masses to kg
    delta_h_dyn = 2 * G_SI / (C_SI**4 * r) * M_ej_dyn * v_ej_dyn**2
    if model == 'exponential':
        h = model_exponential(t, delta_h_dyn, tau)
    if model == 'tanh':
        h = model_tanh(t, delta_h_dyn, tau)
    if plot:
        plt.figure(figsize=(8,5))
        plt.plot(t, h)
        plt.xlabel("t(s)")
        plt.ylabel(r"$h(t)$")
        plt.title(f"Comparison of linear memory models for dynamical ejecta \n $\\tau$ = {tau:.0e}s")
        plt.legend(loc = 'lower right')
        plt.tight_layout()
        #Save the figure
        plt.savefig(outdir + f"phenom_memory_models_tau{tau:.0e}_Mej{M_ej_dyn/MSUN_SI:.2f}_vej{v_ej_dyn/C_SI:.2f}_{model}.png")
        plt.show()
    return t, h

def linear_memory_ejecta_masked(t, M_ej, v_ej, tau, r, start_ms, end_ms, model='exponential'):
    delta_h = 2 * G_SI / (C_SI**4 * r) * M_ej * v_ej**2
    if model == 'exponential':
        h_t = model_exponential(t, delta_h, tau)
    elif model == 'tanh':
        h_t = model_tanh(t, delta_h, tau)
    else:
        raise ValueError("Model must be 'exponential' or 'tanh'")
    h_masked = np.zeros_like(t)
    mask_active = (t >= start_ms) & (t <= end_ms)
    mask_after = t > end_ms
    h_masked[mask_active] = h_t[mask_active]
    if np.any(mask_active):
        last_val = h_t[mask_active][-1]
        h_masked[mask_after] = last_val
    return h_masked

from matplotlib.ticker import FixedLocator, LogLocator

def phenom_memory_ejecta_components(
    t,
    M_ej_dyn, v_ej_dyn, tau_dyn, start_dyn, end_dyn,
    M_ej_wind, v_ej_wind, tau_wind, start_wind, end_wind,
    r,
    model='exponential',
    plot = True,
    delta_h_typical= 3.8e-25
):
    """
    Plot linear memory from two ejecta components (dynamical + wind) with masks and custom parameters.

    Parameters
    ----------
    t : array
        Time array (ms)
    M_ej_dyn, v_ej_dyn, tau_dyn : float
        Mass (kg), velocity (m/s), timescale (ms) for dynamical ejecta
    start_dyn, end_dyn : float
        Start/end time (ms) for dynamical ejecta
    M_ej_wind, v_ej_wind, tau_wind : float
        Mass (kg), velocity (m/s), timescale (ms) for wind ejecta
    start_wind, end_wind : float
        Start/end time (ms) for wind ejecta
    r : float
        Distance (meters)
    model : str
        'exponential' or 'tanh'
    delta_h_typical : float
        Typical value for reference line (default: 3.8e-25)
    """
    # Compute memory for both components (dynamical and wind) with masking 
    h_dyn = linear_memory_ejecta_masked(t, M_ej_dyn, v_ej_dyn, tau_dyn, r, start_dyn, end_dyn, model=model)
    h_wind = linear_memory_ejecta_masked(t, M_ej_wind, v_ej_wind, tau_wind, r, start_wind, end_wind, model=model)
    h_tot = h_dyn + h_wind
    if plot : 
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        # Dynamical ejecta plot
        axes[0].plot(t, h_dyn, color='#D7263D', linewidth=2.5)
        axes[0].fill_between(t, h_dyn, 0, color='#D7263D', alpha=0.2)
        axes[0].set_title(f"Dynamical ejecta ({start_dyn:.0f}–{end_dyn:.0f} ms)", fontsize=15, fontweight='bold')
        axes[0].set_xlabel("t(ms)", fontsize=13)
        axes[0].set_xlim(-1, end_dyn + 1)
        axes[0].set_yscale('linear')
        axes[0].tick_params(axis='both', labelsize=12)
        axes[0].axvline(end_dyn, color='#D7263D', linestyle='--', alpha=0.7)
        axes[0].axhline(delta_h_typical, color='black', linestyle='dotted', linewidth=2, alpha=0.7, label=r'Typical $\Delta h$ (Lopez et al.)')
        axes[0].text(end_dyn, delta_h_typical*1.05, r"$3.8 \times 10^{-25}$ (Lopez et.al)", color='black', fontsize=13, va='bottom', ha='right')
        # Wind ejecta plot
        axes[1].plot(t, h_wind, color='#1B998B', linewidth=2.5)
        axes[1].fill_between(t, h_wind, 0, color='#1B998B', alpha=0.2)
        axes[1].set_title(f"Wind ejecta ({start_wind:.0f}–{end_wind:.0f} ms)", fontsize=15, fontweight='bold')
        axes[1].set_xlabel("t(ms)", fontsize=13)
        axes[1].set_xlim(0, end_wind + 5000)
        axes[1].set_xscale('symlog', linthresh=10, linscale=1.0)
        axes[1].tick_params(axis='both', labelsize=12)
        axes[1].axvline(end_wind, color='#1B998B', linestyle='--', alpha=0.7)
        # Total memory plot = Dynamical + Wind
        axes[2].plot(t, h_tot, color='k', linewidth=2.5, alpha=0.85)
        axes[2].fill_between(t, h_tot, 0, color='#2E294E', alpha=0.13)
        axes[2].set_title("Total ejecta memory", fontsize=15, fontweight='bold')
        axes[2].set_xlabel("t(ms)", fontsize=13)
        axes[2].set_xscale('symlog', linthresh=10, linscale=1.0)
        axes[2].axvline(end_dyn, color='#D7263D', linestyle='--', alpha=0.7)
        axes[2].axvline(end_wind, color='#1B998B', linestyle='--', alpha=0.7)
        axes[2].set_xlim(-1, end_wind + 5000)
        axes[2].axvline(end_dyn, color='#D7263D', linestyle='--', alpha=0.7)
        axes[2].axvline(end_wind, color='#1B998B', linestyle='--', alpha=0.7)
        axes[2].text(end_dyn-1, axes[2].get_ylim()[1]*0.45, 'dynamical', color='#D7263D',
                    fontsize=13, fontweight='bold', rotation=0, va='top', ha='right')
        axes[2].text(end_dyn + 50, axes[2].get_ylim()[1]*0.45, '+ wind', color='#1B998B',
                    fontsize=13, fontweight='bold', rotation=0, va='top', ha='right')

        for ax in axes:
            major_ticks = [0,10, 100, 1000, 10000]
            ax.xaxis.set_major_locator(FixedLocator(major_ticks))
            ax.xaxis.set_minor_locator(LogLocator(base=10, subs='auto'))
            ax.tick_params(axis='x', which='major', labelsize=12, direction='in', width=1.5, length=6)
            ax.tick_params(axis='x', which='minor', direction='in', width=1.0, length=3, labelbottom=False)
            ax.set_ylabel(r"$h(t)$", fontsize=15, fontweight='bold')

        fig.suptitle(rf"Linear memory from ejecta components of a similar case to GW170817" + "\n" +
             rf"Phenomenological model for $\tau_{{dyn}} = {tau_dyn}$ ms and $\tau_{{wind}} = {tau_wind}$ ms", 
             fontsize=18, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(outdir + f"linear_memory_ejecta_components_{model}_Mejdyn{M_ej_dyn/MSUN_SI:.2f}_vejdyn{v_ej_dyn/C_SI:.2f}_tau{tau_dyn:.0e}_Mejwind{M_ej_wind/MSUN_SI:.2f}_vejwind{v_ej_wind/C_SI:.2f}_tau{tau_wind:.0e}.png")
        plt.show()


def phenom_memory_GRB(
    t, E_j, beta, theta, r, tau_GRB, start_GRB, end_GRB, model='exponential', plot=False):
    """
    Calcule et retourne la mémoire GRB masquée (active entre start_GRB et end_GRB, puis constante).

    Paramètres :
    - t : array, temps (ms)
    - E_j : énergie du jet (Joules)
    - beta : vitesse du jet (en unités de c)
    - theta : angle du jet (radians)
    - r : distance à la source (mètres)
    - tau_GRB : temps caractéristique (ms)
    - start_GRB, end_GRB : début/fin de l'émission mémoire (ms)
    - model : 'exponential' ou 'tanh'
    - plot : bool, si True, affiche le signal mémoire GRB
    Retour :
    - h_GRB : array, signal mémoire masqué
    """
    delta_h_GRB = (G_SI/C_SI**4) * (2 * E_j * beta**2 / r) * (np.sin(theta)**2 / (1 - beta * np.cos(theta)))
    if model == 'exponential':
        h_GRB_raw = model_exponential(t, delta_h_GRB, tau_GRB)
    elif model == 'tanh':
        h_GRB_raw = model_tanh(t, delta_h_GRB, tau_GRB)
    else:
        raise ValueError("Model must be 'exponential' or 'tanh'")
    h_GRB = np.zeros_like(t)
    mask_GRB_active = (t >= start_GRB) & (t <= end_GRB)
    mask_GRB_after = t > end_GRB
    h_GRB[mask_GRB_active] = h_GRB_raw[mask_GRB_active]
    if np.any(mask_GRB_active):
        last_val_GRB = h_GRB_raw[mask_GRB_active][-1]
        h_GRB[mask_GRB_after] = last_val_GRB
    if plot: 
        plt.figure(figsize=(8, 5))
        plt.plot(t, h_GRB, label=f'GRB Memory ({model} model)', color='orange', linewidth=1.5)
        plt.fill_between(t, h_GRB, 0, color='orange', alpha=0.3)
        plt.xlabel("t (s)")
        plt.ylabel(r"$h(t)$")
        plt.title(f"Linear memory from a GRB \n $\\tau_{{GRB}}$ = {tau_GRB:.0f}s" + "\n" + rf"$E_{{j}}$ = {E_j*1e7:.0e} erg, $\beta$ = {beta:.2f}, $\theta$ = {theta:.2f} rad", fontsize=14)
        plt.legend(loc='lower right')
        plt.xscale('symlog', linthresh=10, linscale=1.0)
        plt.tight_layout()
        plt.savefig(outdir + f"phenom_memory_GRB_Ej{E_j:.2e}_beta{beta:.2f}_theta{theta:.2f}_tau{tau_GRB:.0e}_{model}_masked.png")
        plt.xlim(-3, end_GRB)
        plt.tick_params(axis='both', labelsize=12)
        ax = plt.gca()
        ax.xaxis.set_major_locator(FixedLocator([0,10,100,1000,10000]))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs='auto'))
        plt.show()
    return h_GRB

def phenom_memory_ejecta_components_GRB(
    t,
    r,
    M_ej_dyn, v_ej_dyn, tau_dyn, start_dyn, end_dyn,
    M_ej_wind, v_ej_wind, tau_wind, start_wind, end_wind,
    E_j, beta, theta, tau_GRB, start_GRB, end_GRB,
    model='exponential',
    plot=True,
    delta_h_typical=3.8e-25
):
    """
    Plot linear memory from two ejecta components (dynamical + wind) and GRB jet


    Parameters
    ----------
    t : array
        Time array (ms)
    M_ej_dyn, v_ej_dyn, tau_dyn : float
        Mass (kg), velocity (m/s), timescale (ms) for dynamical ejecta
    start_dyn, end_dyn : float
        Start/end time (ms) for dynamical ejecta
    M_ej_wind, v_ej_wind, tau_wind : float
        Mass (kg), velocity (m/s), timescale (ms) for wind ejecta
    start_wind, end_wind : float
        Start/end time (ms) for wind ejecta
    r : float
        Distance (meters)
    E_j : float
        Jet energy (Joules)
    beta : float
        Jet velocity (in units of c)
    theta : float
        Jet angle (radians)
    tau_GRB : float
        GRB timescale (ms)
    start_GRB, end_GRB : float
        Start/end time (ms) for GRB memory
    model : str
        'exponential' or 'tanh'
    delta_h_typical : float
        Typical value for reference line (default: 3.8e-25)
    outdir : str
        Output directory for figures
    """
    # Compute memory for both ejecta components
    h_dyn = linear_memory_ejecta_masked(t, M_ej_dyn, v_ej_dyn, tau_dyn, r, start_dyn, end_dyn, model=model)
    h_wind = linear_memory_ejecta_masked(t, M_ej_wind, v_ej_wind, tau_wind, r, start_wind, end_wind, model=model)
    # Compute GRB memory (masked)
    h_GRB = phenom_memory_GRB(t, E_j, beta, theta, r, tau_GRB, start_GRB, end_GRB, model=model, plot=False)
    h_tot = h_dyn + h_wind + h_GRB

    if plot:


        # Data for each subplot
        curves = [
            {"data": h_GRB, "color": "orange", "label": "GRB", "title": f"GRB memory ({start_GRB:.0f}–{end_GRB:.0f} ms)", "vline": end_GRB},
            {"data": h_dyn, "color": "#D7263D", "label": "Dynamical", "title": f"Dynamical ejecta ({start_dyn:.0f}–{end_dyn:.0f} ms)", "vline": end_dyn},
            {"data": h_wind, "color": "#1B998B", "label": "Wind", "title": f"Wind ejecta ({start_wind:.0f}–{end_wind:.0f} ms)", "vline": end_wind},
            {"data": h_tot, "color": "k", "label": "Total", "title": "Total memory", "vline": end_GRB}
        ]
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        for i, ax in enumerate(axes):
            ax.plot(t, curves[i]["data"], color=curves[i]["color"], linewidth=2.5)
            ax.fill_between(t, curves[i]["data"], 0, color=curves[i]["color"], alpha=0.2)
            ax.set_title(curves[i]["title"], fontsize=15, fontweight='bold')
            ax.set_xlabel("t(ms)", fontsize=13)
            
            if i == 0:
                ax.axvline(start_GRB, color='orange', linestyle='--', alpha=0.7)
                ax.axvline(end_GRB, color='orange', linestyle='--', alpha=0.7)
                ax.set_ylabel(r"$h(t)$", fontsize=15, fontweight='bold')
                ax.set_xlim(-1, end_wind + 5000)
                ax.set_xscale('symlog', linthresh=10, linscale=1.0)
  
            if i == 1:
                ax.axvline(start_dyn, color='#D7263D', linestyle='--', alpha=0.7)
                ax.axvline(end_dyn, color='#D7263D', linestyle='--', alpha=0.7)
                ax.axhline(delta_h_typical, color='black', linestyle='dotted', linewidth=2, alpha=0.7)
                ax.text(4.5, delta_h_typical*1.05, r"$3.8 \times 10^{-25}$ (Lopez et al.)", color='black', fontsize=13, va='bottom', ha='left')
                ax.set_xscale('linear')
                ax.set_xlim(-1, end_dyn + 1)
            if i == 2:
                ax.axvline(start_wind, color='#1B998B', linestyle='--', alpha=0.7)
                ax.axvline(end_wind, color='#1B998B', linestyle='--', alpha=0.7)
                ax.set_xlim(0, end_wind + 5000)
                ax.set_xscale('symlog', linthresh=10, linscale=1.0)
            if i == 3:
                ax.axvline(start_dyn, color='#D7263D', linestyle='--', alpha=0.7)
                ax.axvline(end_dyn, color='#D7263D', linestyle='--', alpha=0.7)
                ax.axvline(start_wind, color='#1B998B', linestyle='--', alpha=0.7)
                ax.axvline(end_wind, color='#1B998B', linestyle='--', alpha=0.7)
                ax.axvline(start_GRB, color='orange', linestyle='--', alpha=0.7)
                ax.axvline(end_GRB, color='orange', linestyle='--', alpha=0.7)
                ax.set_ylabel(r"$h(t)$", fontsize=15, fontweight='bold')
                ax.set_xlim(-1, end_wind + 5000)
                ax.set_xscale('symlog', linthresh=10, linscale=1.0)
                
            ax.tick_params(axis='both', labelsize=12)
            ax.xaxis.set_major_locator(FixedLocator([0,10,100,1000,10000]))
            ax.xaxis.set_minor_locator(LogLocator(base=10, subs='auto'))

        fig.suptitle(
            rf"Linear memory from ejecta components and GRB (GW170817-like)"+'\n'
            rf"Phenomenological model for $\tau_{{dyn}} = {tau_dyn}$ ms, $\tau_{{wind}} = {tau_wind}$ ms, $\tau_{{GRB}} = {tau_GRB}$ ms",
            fontsize=18, fontweight='bold', y=1.05
        )
        plt.tight_layout()
        plt.savefig(
            outdir + f"linear_memory_ejecta_components_GRB_{model}_Mejdyn{M_ej_dyn/MSUN_SI:.2f}_vejdyn{v_ej_dyn/C_SI:.2f}_tau{tau_dyn:.0e}_"
            f"Mejwind{M_ej_wind/MSUN_SI:.2f}_vejwind{v_ej_wind/C_SI:.2f}_tau{tau_wind:.0e}_Ej{E_j:.2e}_beta{beta:.2f}_theta{theta:.2f}_tauGRB{tau_GRB:.0e}.png"
        )
        plt.show()


def phenom_memory_total_plot(
    t,
    r,
    M_ej_dyn, v_ej_dyn, tau_dyn, start_dyn, end_dyn,
    M_ej_wind, v_ej_wind, tau_wind, start_wind, end_wind,
    E_j, beta, theta, tau_GRB, start_GRB, end_GRB,
    model='exponential',
    delta_h_typical=3.8e-25,

):
    """
    Plot only the total memory (dynamical + wind + GRB) with vertical lines for each component.

    Parameters are the same as for phenom_memory_ejecta_components_GRB.
    """
    h_dyn = linear_memory_ejecta_masked(t, M_ej_dyn, v_ej_dyn, tau_dyn, r, start_dyn, end_dyn, model=model)
    h_wind = linear_memory_ejecta_masked(t, M_ej_wind, v_ej_wind, tau_wind, r, start_wind, end_wind, model=model)
    delta_h_GRB = (G_SI/C_SI**4) * (2 * E_j * beta**2 / r) * (np.sin(theta)**2 / (1 - beta * np.cos(theta)))
    if model == 'exponential':
        h_GRB_raw = model_exponential(t, delta_h_GRB, tau_GRB)
    elif model == 'tanh':
        h_GRB_raw = model_tanh(t, delta_h_GRB, tau_GRB)
    else:
        raise ValueError("Model must be 'exponential' or 'tanh'")
    h_GRB = np.zeros_like(t)
    mask_GRB_active = (t >= start_GRB) & (t <= end_GRB)
    mask_GRB_after = t > end_GRB
    h_GRB[mask_GRB_active] = h_GRB_raw[mask_GRB_active]
    if np.any(mask_GRB_active):
        last_val_GRB = h_GRB_raw[mask_GRB_active][-1]
        h_GRB[mask_GRB_after] = last_val_GRB
    h_tot = h_dyn + h_wind + h_GRB

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, LogLocator

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(t, h_tot, color='k', linewidth=1.5, alpha=0.85)
    # Remplissage gradué : chaque composant s'ajoute au-dessus du précédent
    # 1. GRB (de 0 à h_GRB)
    ax.fill_between(t, h_GRB, 0, color='orange', alpha=0.23, label='GRB')

    # 2. Dynamical (de h_GRB à h_GRB + h_dyn)
    ax.fill_between(
        t, h_GRB + h_dyn, h_GRB, 
        where=(t >= start_dyn), 
        color='#D7263D', alpha=0.23, label='Dynamical'
    )

    # 3. Wind (de h_GRB + h_dyn à h_GRB + h_dyn + h_wind)
    ax.fill_between(
        t, h_GRB + h_dyn + h_wind, h_GRB + h_dyn, 
        where=(t >= start_wind), 
        color='#1B998B', alpha=0.23, label='Wind'
    )
    ax.set_title("Total linear memory (Dynamical + Wind + GRB)", fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right', bbox_to_anchor=(1.0, 0.5))
    ax.set_xlabel("t (ms)", fontsize=14)
    ax.set_ylabel(r"$h(t)$", fontsize=15, fontweight='bold')
    ax.set_xscale('symlog', linthresh=10, linscale=1.5)
    ax.set_xlim(-0.5, end_wind + 5000)

    vlines_config = [
        (start_dyn, '#D7263D'),
        #(end_dyn, '#D7263D'),
        (start_wind, '#1B998B'),
        #(end_wind, '#1B998B'),
        (start_GRB, 'orange'),
        #(end_GRB, 'orange')
    ]
    for end_time, color in vlines_config:
        ax.axvline(end_time, color=color, linestyle='--', alpha=0.7, linewidth=1.5)

    major_ticks = [0, 10, 100, 1000, 10000]
    ax.xaxis.set_major_locator(FixedLocator(major_ticks))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs='auto'))
    ax.tick_params(axis='x', which='major', labelsize=12, direction='in', width=1.5, length=6)
    ax.tick_params(axis='x', which='minor', direction='in', width=1.0, length=3, labelbottom=False)
    ax.tick_params(axis='y', labelsize=12)

    fig.suptitle(
        rf"Total linear memory from ejecta and GRB"+ '\n'+
        rf"$\tau_{{dyn}} = {tau_dyn}$ ms, $\tau_{{wind}} = {tau_wind}$ ms, $\tau_{{GRB}} = {tau_GRB}$ ms",
        fontsize=15, fontweight='bold', y=0.98
    )
    plt.tight_layout()
    plt.savefig(outdir + f"total_memory_{model}_tau{tau_dyn:.0e}_{tau_wind:.0e}_{tau_GRB:.0e}.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    return h_tot
def memory_fft_phenom(t, h_list, plot=True, save=False, show=False, add_info_title=None, labels=None, outdir="./figures/", outdir_memory_fft="memory_fft"):
    """
    Computes and plots the FFT of one or several memory signals (t and h provided).
    Parameters:
    -----------
    t : array-like
        Time array (seconds)
    h_list : list of array-like
        List of memory signals as a function of time (strain)
    plot : bool
        Whether to generate a plot of the FFT results
    save : bool
        Whether to save the FFT plot to disk
    show : bool
        Whether to display the FFT plot immediately
    add_info_title : str or None
        Additional info for the plot title and filename
    outdir : str
        Output directory
    outdir_memory_fft : str
        Subdirectory for FFT plots
    Returns:
    --------
    frequencies : array-like
        Frequencies corresponding to the FFT result (for the last h in h_list)
    fft_h : array-like
        FFT amplitude of the memory signal (for the last h in h_list)
    """
    import numpy as np
    import os
    from scipy.fft import fft, fftfreq
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    results = []
    colors = ['r', 'b', 'g', 'm', 'c', 'y', 'k']
    for i, h in enumerate(h_list):
        # Filter finite values
        valid_mask = np.isfinite(h) & np.isfinite(t)
        t_valid = t[valid_mask]
        h_valid = h[valid_mask]

        # Interpolate onto a regular grid if needed
        t_regular = np.linspace(t_valid.min(), t_valid.max(), len(t_valid))
        h_interp = np.interp(t_regular, t_valid, h_valid)
        t_seconds = t_regular
        h_valid = h_interp
        dt_mean = np.mean(np.diff(t_seconds))

        N = len(h_valid)
        fft_h = fft(h_valid)
        frequencies = fftfreq(N, dt_mean)[:N//2]
        fft_h = 2.0/N * np.abs(fft_h[:N//2])

        pos_mask = frequencies > 0
        frequencies_plot = frequencies[pos_mask]
        fft_h_plot = fft_h[pos_mask]
        results.append((frequencies_plot, fft_h_plot))

        if plot:
            color = colors[i % len(colors)]
            plt.loglog(frequencies_plot, fft_h_plot, '-', linewidth=2, color=color, label=labels[i] if labels else None)

    if plot:
        plt.tick_params(top=True, right=True, axis='both', which='major', labelsize=12, direction='in', length=6, width=1.2)
        plt.xlabel('f [Hz]', fontsize=14)
        plt.ylabel('Amplitude (strain)', fontsize=14)
        if add_info_title:
            plt.title('FFT memory signal\n' + f'{add_info_title}', fontsize=15)
        else:
            plt.title('FFT memory signal for ejecta', fontsize=15)
        plt.xlim(1e-6, 200)
        plt.grid(True, alpha=0.3)
        if len(h_list) > 1:
            plt.legend()
        plt.tight_layout()
    if save:
        os.makedirs(os.path.join(outdir, outdir_memory_fft), exist_ok=True)
        if add_info_title:
            add_info_save = '_' + add_info_title
            plt.savefig(os.path.join(outdir, outdir_memory_fft, f"memory_fft_ejecta{add_info_save}.png"), dpi=150, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(outdir, outdir_memory_fft, "memory_fft_ejecta.png"), dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
