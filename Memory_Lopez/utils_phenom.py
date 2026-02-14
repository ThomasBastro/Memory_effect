import numpy as np
import matplotlib.pyplot as plt
import os
from lal import MSUN_SI, PC_SI, C_SI, G_SI
from matplotlib.ticker import FixedLocator, LogLocator

import LISA as li
lisa = li.LISA() # initialize LISA object
f_lisa = np.logspace(np.log10(1.0e-5), np.log10(1.0e0), 1000)
Sn_lisa = lisa.Sn(f_lisa)

outdir = "./figures/"

def model_exponential(t, delta_h, tau):
    return delta_h / (1 + np.exp(-t / tau))
def model_tanh(t, delta_h, tau):
    return delta_h * 0.5 * np.tanh(t/tau) + delta_h * 0.5

def delta_h_ejecta(M_ej, v_ej, r):
    """
    Compute the memory amplitude from ejecta mass and velocity.
    The mass is in unit of solar mass, velocity in unit of c, distance in parsecs.

    Parameters:
    -----------
    M_ej : float
        Ejecta mass in solar mass
    v_ej : float
        Ejecta velocity (m/s)
    r : float
        Distance to source (parsecs)
    Returns:
    --------
    delta_h : float
        Memory amplitude (strain)
    """
    r = r * PC_SI  # Convert distance from parsecs to meters
    M_ej = M_ej * MSUN_SI  # Convert mass from solar masses to kg
    v_ej = v_ej * C_SI  # Convert velocity from c to m/s
    delta_h = 2 * G_SI / (C_SI**4 * r) * M_ej * v_ej**2
    return delta_h
def delta_h_GRB(E_j, beta, theta, r):
    """
    Compute the memory amplitude from GRB jet parameters.
    The energy is in erg , velocity in units of c, distance in parsecs.

    Parameters:
    -----------
    E_j : float
        Energy of the GRB jet (erg)
    beta : float
        Velocity of the jet (in units of c)
    theta : float
        Angle of the jet (radians)
    r : float
        Distance to source (parsecs)
    Returns:
    --------
    delta_h_GRB : float
        Memory amplitude from the GRB jet (strain)
    """
    E_j = E_j * 1e-7  # Convert energy from erg to Joules
    r = r * PC_SI  # Convert distance from parsecs to meters
    delta_h_GRB = (G_SI/C_SI**4) * (2 * E_j * beta**2 / r) * (np.sin(theta)**2 / (1 - beta * np.cos(theta)))
    return delta_h_GRB

def phenom_memory_models(tau, M_ej_dyn, v_ej_dyn, r, t, model, plot):
    """
    Compare exponential (Lopez et al.) and tanh (Favata) models for dynamical ejecta memory.
    The mass is in unit of solar mass, velocity in unit of c, distance in parsecs, time in seconds, and tau in seconds.

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
    v_ej_dyn = v_ej_dyn * C_SI  # Convert velocity from c to m/s
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
        plt.savefig(outdir + f"phenom_memory_models_tau{tau:.0e}_Mej{M_ej_dyn:.2f}_vej{v_ej_dyn/C_SI:.2f}_{model}.png")
        plt.show()
    return t, h

def linear_memory_ejecta_masked(t, M_ej, v_ej, tau, r, start_ms, end_ms, model='exponential'):
    """
    Compute linear memory from ejecta with masking for active and after phases.
    The mass is in unit of solar mass, velocity in unit of c, distance in parsecs, time in seconds, and tau in seconds.
    Parameters:
    -----------
    t : array
        Time array (s)
    M_ej : float
        Ejecta mass in solar mass
    v_ej : float        Ejecta velocity (m/s)
    tau : float
        Characteristic timescale (s)
    r : float
        Distance to source (parsecs)
    start_ms : float
        Start time of active phase (ms)
    end_ms : float
        End time of active phase (ms)
    model : str
        Model type ('exponential' or 'tanh')
    Returns:
    --------    
    h_masked : array
        Memory strain array with masking applied
    """
    # Convert parameters to SI units
    r = r * PC_SI  # Convert distance from parsecs to meters
    M_ej = M_ej * MSUN_SI  # Convert mass from solar masses to kg
    v_ej = v_ej * C_SI  # Convert velocity from c to m/s
    
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
    delta_h_typical= True,
    save = False
):
    """
    Plot linear memory from two ejecta components (dynamical + wind) with masks and custom parameters.
    The mass is in unit of solar mass, velocity in unit of c, distance in parsecs, time in seconds, and tau in seconds.

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
    delta_h_typical : bool
        Whether to plot a reference line for typical memory amplitude from Lopez et al. (default: True)
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
        if delta_h_typical :
            delta_h_typical=3.8e-25
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
        if save:
            plt.savefig(outdir + f"linear_memory_ejecta_components_{model}_Mejdyn{M_ej_dyn:.2f}_vejdyn{v_ej_dyn/C_SI:.2f}_tau{tau_dyn:.0e}_Mejwind{M_ej_wind:.2f}_vejwind{v_ej_wind/C_SI:.2f}_tau{tau_wind:.0e}.png")
        plt.show()


def phenom_memory_GRB(
    t, E_j, beta, theta, r, tau_GRB, start_GRB, end_GRB, model='exponential', plot=False):
    """
    Compute and plot the linear memory from a GRB jet with masking.
    The energy is in erg , velocity in units of c, distance in parsecs, time in seconds, and tau in seconds.

    Parameters   
    ----------
    t : array
        Time array (s)
    E_j : float
        Energy of the GRB jet (erg)
    beta : float
        Velocity of the jet (in units of c)
    theta : float
        Angle of the jet (radians)
    r : float
        Distance to source (parsecs)
    tau_GRB : float
        Characteristic timescale for GRB memory (s)
    start_GRB : float
        Start time of GRB memory active phase (s)
    end_GRB : float
        End time of GRB memory active phase (s)
    model : str
        Model type for GRB memory evolution ('exponential' or 'tanh')
    plot : bool
        Whether to plot the results
    Returns
    -------
    h_GRB : array
        Memory strain array for the GRB jet with masking applied
    """
    E_j = E_j * 1e-7  # Convert energy from erg to Joules
    r = r * PC_SI  # Convert distance from parsecs to meters
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
    save=False,
    delta_h_typical=True
):
    """
    Plot linear memory from two ejecta components (dynamical + wind) and GRB jet
    The mass is in unit of solar mass, velocity in unit of c, distance in parsecs, time in seconds, and tau in seconds.

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
    delta_h_typical : bool
        Whether to plot a reference line for typical memory amplitude from Lopez et al. (default: True)
    outdir : str
        Output directory for figures
    plot : bool
        Whether to plot the results
    save : bool
        Whether to save the figure (default: False)
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
                if delta_h_typical :
                    delta_h_typical=3.8e-25
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
        if save:
            plt.savefig(
            outdir + f"linear_memory_ejecta_components_GRB_{model}_Mejdyn{M_ej_dyn:.2f}_vejdyn{v_ej_dyn/C_SI:.2f}_tau{tau_dyn:.0e}_"
            f"Mejwind{M_ej_wind:.2f}_vejwind{v_ej_wind:.2f}_tau{tau_wind:.0e}_Ej{E_j:.2e}_beta{beta:.2f}_theta{theta:.2f}_tauGRB{tau_GRB:.0e}.png"
        )
        plt.show()


def phenom_memory_total_plot(
    t,
    r,
    M_ej_dyn, v_ej_dyn, tau_dyn, start_dyn, end_dyn,
    M_ej_wind, v_ej_wind, tau_wind, start_wind, end_wind,
    E_j, beta, theta, tau_GRB, start_GRB, end_GRB,
    model='exponential',
    save=False

):
    """
    Plot only the total memory (dynamical + wind + GRB) with vertical lines for each component.
    The mass is in unit of solar mass, velocity in unit of c, distance in parsecs, time in seconds, and tau in seconds.

    Parameters are the same as for phenom_memory_ejecta_components_GRB.
    """
    # Compute memory for both ejecta components and GRB
    h_dyn = linear_memory_ejecta_masked(t, M_ej_dyn, v_ej_dyn, tau_dyn, r, start_dyn, end_dyn, model=model)
    h_wind = linear_memory_ejecta_masked(t, M_ej_wind, v_ej_wind, tau_wind, r, start_wind, end_wind, model=model)
    h_GRB = phenom_memory_GRB(t, E_j, beta, theta, r, tau_GRB, start_GRB, end_GRB, model=model, plot=False)
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
    if save:
        plt.savefig(outdir + f"total_memory_{model}_tau{tau_dyn:.0e}_{tau_wind:.0e}_{tau_GRB:.0e}.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    return h_tot

def memory_fft_phenom(
    t, h_list, plot=True, save=False, show=False, add_info_title=None, labels=None,
    outdir="./figures/", outdir_memory_fft="memory_fft", return_fig=False, return_fft=False, LISA=False
):
    """
    Computes and plots the FFT of one or several memory signals (t and h provided).
    Returns a list of (frequencies, fft) for each signal.
    """
    from scipy.fft import fft, fftfreq
    plt.figure(figsize=(8, 8))
    results = []
    colors = ['r', 'b', 'g', 'm', 'c', 'y', 'k']

    for i, h in enumerate(h_list):
        valid_mask = np.isfinite(h) & np.isfinite(t)
        t_valid = t[valid_mask]
        h_valid = h[valid_mask]

        # Interpolate onto a regular grid if needed
        t_regular = np.linspace(t_valid.min(), t_valid.max(), len(t_valid))
        h_interp = np.interp(t_regular, t_valid, h_valid)
        dt_mean = np.mean(np.diff(t_regular))

        N = len(h_interp)
        fft_h = fft(h_interp)
        frequencies = fftfreq(N, dt_mean)
        # Only positive frequencies
        pos_mask = frequencies > 0
        frequencies_plot = frequencies[pos_mask]
        fft_h_plot = 2.0/N * np.abs(fft_h[pos_mask])
        # Plot characteristic strain (h_c = 2f|h(f)|) to compare with sensitivity curves
        results.append((frequencies_plot, 2*frequencies_plot*fft_h_plot))

        if plot:
            color = colors[i % len(colors)]
            plt.loglog(frequencies_plot, fft_h_plot, '-', linewidth=2, color=color, label=labels[i] if labels else f"Signal {i+1}")

    if plot:
        plt.tick_params(top=True, right=True, axis='both', which='major', labelsize=12, direction='in', length=6, width=1.2)
        plt.xlabel('f [Hz]', fontsize=14)
        plt.ylabel(r'Caracteristic strain $h_c$', fontsize=14)
        if LISA:
            plt.loglog(f_lisa, np.sqrt(f_lisa*Sn_lisa), label='LISA', color='k', linestyle='-.')
        if add_info_title:
            plt.title('FFT memory signal\n' + f'{add_info_title}', fontsize=15)
        else:
            plt.title('FFT memory signal for ejecta', fontsize=15)
        plt.xlim(1e-6, 200)
        if labels is not None:
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
    if return_fig:
        return plt.gcf(), plt.gca()
    if return_fft:
        return results

def memory_fft_formula(delta_h, tau, plot=True, save=False, show=False, add_info_title=None, outdir="./figures/", outdir_memory_fft="memory_fft", return_fig=False, LISA=False):
    """
    Computes and plots the FFT of a memory signal (t and h provided) based on formula (4.2) provided by https://arxiv.org/pdf/gr-qc/0405067
    The characteristic strain is rather plot than the FFT amplitude, to be able to compare with sensitivity curves
    Parameters:
    -----------
    delta_h : float
        Memory amplitude (strain)
    tau : float
        Characteristic timescale (seconds)
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
        Frequencies corresponding to the FFT result
    fft_h : array-like
        FFT amplitude of the memory signal
    """
    f = np.logspace(-6, 4, int(1e5))  # Frequencies from 1e-6 to 10000 Hz with 10,000 points
    fourier_h_square = delta_h**2 * (1-np.cos(2 * np.pi * f * tau)) /(8 * np.pi**4 * f**4 * tau**2) 
    caracteristic_strain = 2*f* np.sqrt(fourier_h_square)
    if plot:
        plt.figure(figsize=(8, 8))
        plt.loglog(f,caracteristic_strain , '-', linewidth=2)
        plt.xlabel('f [Hz]', fontsize=14)
        plt.ylabel(r'Caracteristic strain $h_c$[f]', fontsize=14)
        plt.xlim(1e-5, 1e2)
        if add_info_title:
            plt.title('FFT memory signal from formula\n' + f'{add_info_title}', fontsize=15)
        else:
            plt.title('FFT memory signal from formula', fontsize=15)
        plt.xlim(1e-6, 200)
        plt.tight_layout()
    if LISA:
        plt.loglog(f_lisa, np.sqrt(f_lisa*Sn_lisa), label='LISA', color='k', linestyle='-.')
    if save:
        os.makedirs(os.path.join(outdir, outdir_memory_fft), exist_ok=True)
        if add_info_title:
            add_info_save = '_' + add_info_title
            plt.savefig(os.path.join(outdir, outdir_memory_fft, f"memory_fft_formula{add_info_save}.png"), dpi=150, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(outdir, outdir_memory_fft, "memory_fft_formula.png"), dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    if return_fig:
        return plt.gcf(), plt.gca()
    
    return f, caracteristic_strain

#--- GRB + afterglow modesl from https://arxiv.org/pdf/2301.12590

def memory_initial_acceleration_GRB(Ej, theta_ej, phi_ej, d, beta=0.99):
    """
    Initial memory from the jet acceleration phase (Eq. 11) : h_in
    The jets are assumed to be instantaneously accelerated.
    Paramètres:
    -----------
    Ej : float
        Total kinetic energy of the jet [erg]
    theta_ej : float
        Viewing angle (angle between jet axis and line of sight) [rad]
    phi_ej : float
        Azimuthal angle of the jet in the plane of the sky [rad]
    d : float
        Distance to the source [pc]
    beta : float
        Normalized velocity of the jet (v/c), typically close to 1 for (ultra)relativistic jets
    
    Retourne:
    ---------
    h_in : float
        Amplitude GW mémoire de la phase initiale
    """
 
    # Angular factor
    angular_factor = beta**2 * np.sin(theta_ej)**2 * np.cos(2 * phi_ej) / (1 - beta* np.cos(theta_ej))  
    # Convert energy from erg to Joules
    Ej *= 1e-7
    # Convert distance from parsec to meters
    d_m = d * PC_SI
    # Calculate the initial memory amplitude
    h_in = (2 * G_SI / C_SI**4) * (Ej / d_m) * angular_factor
    return h_in


def memory_afterglow_injection(Pin, T_end, theta_ej, theta_j, d, beta=0.99):
    """
    Additional memory from the afterglow injection phase (Eq. 12) : h_m 
    Arise from the continuous energy injection into the external medium by the jet after the initial acceleration phase.
    The GW signals in GRB afterglows originate from the shock-accelerated ISM and the synchrotrons emission.
    Parameters:
-----------
    Pin : float
        Power of the energy injection [erg/s] ~ 10^48- 10^50 erg/s
    T_end : float
        Duration of the energy injection phase [s] ~ 10^2 - 10^3 s =? Duration of the burst in the observer frame
    theta_ej : float
        Viewing angle (angle between jet axis and line of sight) [rad]
    theta_j : float
        Jet opening angle [rad]
        theta_j = theta_ej / 2
    d : float
        Distance to the source [pc]
    beta : float
        Normalized velocity of the jet (v/c), typically close to 1 for (ultra)relativistic jets
    """
    # Angular factor (same as initial memory)
    angular_factor = beta**2 * np.sin(theta_ej)**2 * (1- np.cos(theta_j)) / (1 - beta* np.cos(theta_ej))   
    # Convert power from erg/s to W
    Pin *=  1e-7
    # Convert Pin to total injected energy over T_end
    E_injected_J = Pin * T_end
    # Convert distance from parsec to meters
    d_m = d * PC_SI
    # Calculate the additional memory amplitude from the afterglow injection
    h_m = (G_SI / C_SI**4) * (E_injected_J / d_m) * angular_factor
    
    return h_m


def memory_total_waveform(t_obs, h_in, h_m, t_end_injection, theta_ej, radius =0.01):
    """
    Total memory waveform combining the initial acceleration and afterglow injection phases
    Parameters:
    -----------
    t_obs : array-like
        Time array for the observed memory signal [s]
    h_in : float
        Initial memory amplitude from the jet acceleration phase
    h_m : float
        Additional memory amplitude from the afterglow injection phase
    t_end_injection : float
        Characteristic timescale for the afterglow injection phase (duration over which h_m is accumulated) [s]
    theta_ej : float
        Viewing angle (angle between jet axis and line of sight) [rad]
    radius : float [pc]
        Characteristic radius for the afterglow shock (in parsecs) used to estimate the timescale t_m for the memory to reach its maximum value. Default is 0.01 [pc]
        
    """
    h_total = np.zeros_like(t_obs) # Zero array to hold the total memory signal before the injection starts
    
    # In this model, we have to compute t_m the timescale for the memory to reach its maximum value (h_in + h_m).
    # Normally it is defined as the end time of the energy injection phase + (distance of the jet to the source at T_end) / c 
    # Here, for simplicity, we assume that R_end ~0.01 pc - typicall value
    t_m = t_end_injection + ( (radius * PC_SI) * (1-np.cos(theta_ej)) )/ C_SI # Neglect redshift (otherwise add a factor (1+z) in the numerator)
 
    # Mask for the rising phase of the memory (from 0 to t_m)
    mask_rise = (t_obs > 0) & (t_obs <= t_m)
    h_total[mask_rise] = h_in + h_m * (t_obs[mask_rise] / t_m)
    
    # Mask for the plateau phase of the memory (after t_m), where the memory has reached its maximum value (h_in + h_m)
    mask_plateau = t_obs > t_m
    h_total[mask_plateau] = h_in + h_m
    
    return h_total, t_m

def new_fft(h_in, h_m, t_end_injection, theta_ej, radius=0.01):
    """
    Compute the FFT as mentionned in (17)
    
    Parameters:
    -----------
    h_in : float
        Initial memory amplitude from the jet acceleration phase
    h_m : float
        Additional memory amplitude from the afterglow injection phase
    t_end_injection : float
        Characteristic timescale for the afterglow injection phase (duration over which h_m is accumulated) [s]
    theta_ej : float
        Viewing angle (angle between jet axis and line of sight) [rad]
    radius : float [pc]
        Characteristic radius for the afterglow shock (in parsecs) used to estimate the timescale t_m for the memory to reach its maximum value. Default is 0.01 [pc]
    """
    f = np.logspace(-6, 4, int(1e5))  # Frequencies from 1e-6 to 10000 Hz with 10,000 points
    t_m = t_end_injection + (radius* PC_SI) * (1-np.cos(theta_ej))/ C_SI
    
    a = h_m/(4*np.pi**2 * f**2 * t_m)
    b = h_in/(2*np.pi* f)
    
    fourier_h_square = 4*a**2 * (np.sin(np.pi * f * t_m))**4 +(a*np.sin(2*np.pi * f * t_m) + b)**2
    
    return f, np.sqrt(fourier_h_square)