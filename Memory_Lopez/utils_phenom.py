import numpy as np
import matplotlib.pyplot as plt
from lal import MSUN_SI, PC_SI, C_SI, G_SI

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
        plt.legend()
        plt.tight_layout()
        #Save the figure
        plt.savefig(outdir + f"phenom_memory_models_tau{tau:.0e}_Mej{M_ej_dyn/MSUN_SI:.2f}_vej{v_ej_dyn/C_SI:.2f}_{model}.png")
        plt.show()
    return t, h

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
        # Save the figure

def phenom_memory_GRB(t, E_j, beta, theta, r, tau_GRB, model='exponential', plot=False):
    """
    Compute the linear memory from a GRB jet using a phenomenological model.
    
    Parameters:
    - t: time array 
    - E_j: energy released in the jet (Joules)
    - beta: velocity of the jet in units of c
    - theta: angle between the jet axis and the line of sight (radians)
    - r: distance to the source (meters)
    - tau_GRB: characteristic timescale of the GRB memory signal 
    - model: 'exponential' or 'tanh' for the time evolution of the memory signal
    - plot: whether to plot the memory signal
    
    Returns:
    - t: time array 
    - h_GRB: memory signal as a function of time
    """
    # Calculate the amplitude of the memory signal
    delta_h = (G_SI/C_SI**4) * (2 * E_j * beta**2 / r) * (np.sin(theta)**2 / (1 - beta * np.cos(theta)))
    
    # Generate the time evolution of the memory signal
    if model == 'exponential':
        h_GRB = model_exponential(t, delta_h, tau_GRB)
    elif model == 'tanh':
        h_GRB = model_tanh(t, delta_h, tau_GRB)
    else:
        raise ValueError("Model must be 'exponential' or 'tanh'")
    
    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(t, h_GRB, label=f'GRB Memory ({model} model)', color='orange')
        plt.xlabel("t (s)")
        plt.ylabel(r"$h(t)$")
        plt.title(f"Linear memory from a GRB \n $\\tau_{{GRB}}$ = {tau_GRB:.0f}s")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir + f"phenom_memory_GRB_Ej{E_j:.2e}_beta{beta:.2f}_theta{theta:.2f}_tau{tau_GRB:.0e}_{model}.png")
        plt.show()
        
        
    
    return t, h_GRB