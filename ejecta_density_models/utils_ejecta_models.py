import numpy as np
import matplotlib.pyplot as plt
from lal import C_SI, G_SI, PC_SI

from matplotlib import animation
from matplotlib.animation import PillowWriter
from IPython.display import HTML
import os
# Parameters
c = C_SI
outdir='ejecta_density_plot'
f_d_pol = 0.01  # CORRECTED : density contrast for polar region = 0.01 as seen in the reference paper for various models
# This value should be determined based on the specific simulation/model being used !

# Angular function (eq 4 - eq 5)
def Theta(theta):
    """Transition function between polar and equatorial regions"""
    return 1.0 / (1.0 + np.exp(-10 * (theta - np.pi/4)))

def eta(theta, f_d_pol=f_d_pol):
    """Angular density profile for dynamical ejecta"""
    return (1 - Theta(theta)) * f_d_pol + Theta(theta)

# Modified density profile function
def rho_profile_model1(r, t, theta, v_pm_min, v_pm_max, v_d_min, v_max, f_d_pol=0.01):
    """
    Compute the density profile for a simple ejecta model with dynamical and post-merger components
    Assume that the post-merger ejecta won't reach the highest velocities of the dynamical ejecta
    
    Parameters:
        r: radius (m)
        t: time (s)
        theta: polar angle (radians)
        v_pm_min: minimum velocity for post-merger ejecta (in units of c)
        v_pm_max: maximum velocity for post-merger ejecta (in units of c)
        v_d_min: minimum velocity for dynamical ejecta (in units of c)
        v_max: maximum velocity for dynamical ejecta (in units of c) = 0.9 c
    Returns:
        rho_dyn: density of dynamical ejecta
        rho_pm: density of post-merger ejecta
        rho_tot: total density (sum of both)
    """
    v = r / t / c
    eta_theta = eta(theta, f_d_pol)
    t_day = t / 86400  # t in days
    rho_floor = 1e-20 * t_day**-3  
    if v_pm_max > v_d_min:
        raise ValueError("In this model, v_pm_max must be less than v_d_min !")
    # Dynamical ejecta
    if v_d_min <= v < 0.4:
        rho_dyn = eta_theta * r**(-4) * t**(-3)
    elif 0.4 <= v < v_max:
        rho_dyn = eta_theta * r**(-8) * t**(-3)
    else:
        rho_dyn = 0.0

    # Post-merger ejecta
    if v_pm_min <= v < v_pm_max:
        rho_pm = r**-3 * t**-3
    else:
        rho_pm = 0.0
        
    # Floor (if outside both regions)
    if v < v_pm_min or (v_pm_max <= v < v_d_min):
        rho_floor_val = rho_floor
    else:
        rho_floor_val = 0.0

    rho_tot = rho_dyn + rho_pm + rho_floor_val
    return rho_dyn, rho_pm, rho_tot

# Second model
def rho_profile_model2(r, t, theta, v_pm_min, v_pm_max, v_d_min, v_max, f_d_pol=0.01):
    """
    Second density profile model 
    - Allows v_pm_max > v_d_min.
    - In the overlap region (v_d_min < v < v_pm_max), for theta > pi/4, only dynamical ejecta is present.
    Parameters:
        r: radius (m)
        t: time (s)
        theta: polar angle (radians)
        v_pm_min: minimum velocity for post-merger ejecta (in units of c)
        v_pm_max: maximum velocity for post-merger ejecta (in units of c)
        v_d_min: minimum velocity for dynamical ejecta (in units of c)
        v_max: maximum velocity for dynamical ejecta (in units of c) =
    
    Returns:
        rho_dyn: density of dynamical ejecta
        rho_pm: density of post-merger ejecta
        rho_tot: total density (sum of both)
    """
    if v_pm_max > v_d_min:
        v = r / t / c
        eta_theta = eta(theta, f_d_pol)
        t_day = t / 86400
        rho_floor = 1e-20 * t_day**-3  # g/cm^3

        # Dynamical ejecta
        if v_d_min <= v < 0.4:
            rho_dyn = eta_theta * r**-4 * t**-3
        elif 0.4 <= v < v_max:
            rho_dyn = eta_theta * r**-8 * t**-3
        else:
            rho_dyn = 0.0

        # Post-merger ejecta (with truncation in overlap for theta > pi/4)
        if v_pm_min <= v < v_pm_max:
            if (v > v_d_min) and (theta > np.pi/4):
                rho_pm = 0.0  
            else:
                rho_pm = r**-3 * t**-3
        else:
            rho_pm = 0.0

        # Floor (if outside both regions)
        if v < v_pm_min or (v_pm_max <= v < v_d_min):
            rho_floor_val = rho_floor
        else:
            rho_floor_val = 0.0

        rho_tot = rho_dyn + rho_pm + rho_floor_val
        return rho_dyn, rho_pm, rho_tot
    else:
        # First model if v_pm_max <= v_d_min (same case)
        return rho_profile_model1(r, t, theta, v_pm_min, v_pm_max, v_d_min, v_max, f_d_pol)
    
# fonction qui trace la densité de l'ej dynamique - post-merger - totale pour un modèle et ses params
def plot_ejecta_density_grid(model, model_name, t=0.01, v_pm_min=0.025, v_pm_max=0.15, v_d_min=0.15, v_max=0.9, n=200, f_d_pol=0.01):
    """
    Plot the density distribution of dynamical ejecta, post-merger ejecta, and total density in the (v_x, v_z) plane for a given model and parameters.
    Parameters:
    - model: function that computes the density profile (e.g., rho_profile_model1 or rho_profile_model2)
    - model_name: string name of the model (for plot title)
    - t: time in seconds
    - v_pm_min: minimum velocity for post-merger ejecta (in units of c)
    - v_pm_max: maximum velocity for post-merger ejecta (in units of c)
    - v_d_min: minimum velocity for dynamical ejecta (in units of c)
    - v_max: maximum velocity for dynamical ejecta (in units of c)
    - n: number of grid points in each velocity dimension
    - f_d_pol: density contrast for polar region (default 0.01)
    """

    vx_grid = np.linspace(0, v_max, n)
    vz_grid = np.linspace(0, v_max, n)
    VX, VZ = np.meshgrid(vx_grid, vz_grid)
    THETA = np.arctan2(VX, VZ)

    rho_dyn = np.zeros_like(VX)
    rho_pm = np.zeros_like(VX)
    rho_tot = np.zeros_like(VX)
    for i in range(n):
        for j in range(n):
            v_tot = np.sqrt(VX[i, j]**2 + VZ[i, j]**2)
            r = v_tot * c * t
            theta_val = THETA[i, j] 
            dyn, pm, tot = model(r, t, theta_val, v_pm_min, v_pm_max, v_d_min, v_max, f_d_pol)
            rho_dyn[i, j] = dyn
            rho_pm[i, j] = pm
            rho_tot[i, j] = tot

    levels = np.linspace(-20, -9, 12)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    titles = ["Dynamical ejecta", "Post-merger ejecta", "Total density"]
    datas = [rho_dyn, rho_pm, rho_tot]
    cmaps = ['Reds', 'Blues', 'Greens']

    for ax, data, title, cmap in zip(axes, datas, titles, cmaps):
        im = ax.contourf(VX, VZ, np.log10(data + 1e-30), levels = levels, cmap=cmap)
        ax.set_xlabel(r'$v_x/c$')
        ax.set_title(title)
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, 0.5)
        fig.colorbar(im, ax=ax, orientation='vertical', label=r'$\log_{10}(\rho)$ [g/cm$^3$]')
    axes[0].set_ylabel(r'$v_z/c$')
    plt.suptitle(f"Ejecta density at t={t:.3f} s for model {model_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    
def make_ejecta_density_gif(
    model,
    model_name,
    t_values,
    v_pm_min,
    v_pm_max,
    v_d_min,
    v_max,
    n,
    f_d_pol,
    xlim=0.5,
    ylim=0.5,
    levels=np.linspace(-20, -9, 12),
    cmaps=['Reds', 'Blues', 'Greens'],
    titles=["Dynamical ejecta", "Post-merger ejecta", "Total density"],
    fps=8,
    dpi=200
):
    vx_grid = np.linspace(0, v_max, n)
    vz_grid = np.linspace(0, v_max, n)
    VX, VZ = np.meshgrid(vx_grid, vz_grid)
    THETA = np.arctan2(VX, VZ)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

    def compute_densities(t):
        rho_dyn = np.zeros_like(VX)
        rho_pm = np.zeros_like(VX)
        rho_tot = np.zeros_like(VX)
        for i in range(n):
            for j in range(n):
                v_tot = np.sqrt(VX[i, j]**2 + VZ[i, j]**2)
                r = v_tot * c * t
                theta_val = THETA[i, j]
                dyn, pm, tot = model(r, t, theta_val, v_pm_min, v_pm_max, v_d_min, v_max, f_d_pol)
                rho_dyn[i, j] = dyn
                rho_pm[i, j] = pm
                rho_tot[i, j] = tot
        return rho_dyn, rho_pm, rho_tot

    def init():
        for ax in axes:
            ax.clear()
        for ax, title in zip(axes, titles):
            ax.set_xlabel(r'$v_x/c$')
            ax.set_xlim(0, xlim)
            ax.set_ylim(0, ylim)
            ax.set_title(title)
        axes[0].set_ylabel(r'$v_z/c$')
        return axes

    def animate(frame):
        t = t_values[frame]
        rho_dyn, rho_pm, rho_tot = compute_densities(t)
        datas = [rho_dyn, rho_pm, rho_tot]
        for ax, data, cmap in zip(axes, datas, cmaps):
            ax.clear()
            ax.contourf(VX, VZ, np.log10(data + 1e-30), levels=levels, cmap=cmap)
            ax.set_xlabel(r'$v_x/c$')
            ax.set_xlim(0, xlim)
            ax.set_ylim(0, ylim)
        axes[0].set_ylabel(r'$v_z/c$')
        axes[0].set_title("Dynamical ejecta")
        axes[1].set_title("Post-merger ejecta")
        axes[2].set_title("Total density")
        fig.suptitle(f"Ejecta density at t={t:.3f} s for {model_name}", fontsize=16)
        return axes

    os.makedirs(outdir, exist_ok=True)
    ani = animation.FuncAnimation(fig, animate, frames=len(t_values), init_func=init, blit=False, interval=250)
    writer = PillowWriter(fps=fps)
    gif_path = f'{outdir}/ejecta_density_{model_name.replace(" ", "_")}.gif'
    ani.save(gif_path, writer=writer, dpi=dpi)
    plt.close(fig)
    HTML(ani.to_jshtml())
    return gif_path, ani

def XZ_dyn(theta, XZ):
    """Mass fraction of element Z for dynamical ejecta, angular dependence."""
    return (1 - Theta(theta)) * XZ + Theta(theta) * XZ

def XZ_pm(Ye_range):
    """Mass fraction for post-merger ejecta, fixed by Ye range."""
    # Example values for lanthanides
    if Ye_range == "0.3-0.4":
        return 1e-3
    elif Ye_range == "0.2-0.4":
        return 0.025
    elif Ye_range == "0.1-0.3":
        return 0.14
    else:
        return 0.0

def compute_ejecta_mass(
    rho_profile_func,
    t,
    theta,
    v_pm_min, v_pm_max, v_d_min, v_max,
    f_d_pol=0.01,
    r_min=1e7, r_max=1e10, n_r=200
):
    """
    Calcule la masse totale de l'éjecta à un angle theta donné et à un instant t,
    en intégrant rho(r, t, theta) * dV sur r.

    Paramètres :
    ------------
    rho_profile_func : fonction
        Fonction de profil de densité (ex: rho_profile_model1 ou model2)
    t : float
        Temps (s)
    theta : float
        Angle polaire (rad)
    v_pm_min, v_pm_max, v_d_min, v_max : float
        Paramètres de vitesses (en unités de c)
    f_d_pol : float
        Contraste polaire
    r_min, r_max : float
        Bornes d'intégration en rayon (m)
    n_r : int
        Nombre de points pour l'intégration radiale

    Retour :
    --------
    mass_theta : float
        Masse totale de l'éjecta à cet angle (en g)
    """
    r_grid = np.linspace(r_min, r_max, n_r)
    dr = np.gradient(r_grid)
    mass_theta_dyn = 0.0
    mass_theta_pm = 0.0
    mass_theta = 0.0
    for ir, r in enumerate(r_grid):
        rho_dyn, rho_pm, rho_tot = rho_profile_func(
            r, t, theta, v_pm_min, v_pm_max, v_d_min, v_max, f_d_pol
        )
        dV = 2 * np.pi * r**2 * np.sin(theta) * dr[ir]  # dV sphérique, symétrie axiale
        mass_theta_dyn += rho_dyn * dV
        mass_theta_pm += rho_pm * dV
        mass_theta += rho_tot * dV
        
    return mass_theta_dyn, mass_theta_pm, mass_theta



def compute_memory_from_density_profile(
    rho_profile_func,
    t_arr,
    theta,
    v_pm_min, v_pm_max, v_d_min, v_max,
    f_d_pol=0.01,
    r_min=1e7, r_max=1e10, n_r=200,
    distance_pc=40e6,
    tau=1e-3,
    model='exponential'
):
    """
    Calcule la mémoire linéaire phénoménologique pour un angle donné à partir d'un profil de densité.

    Paramètres :
    ------------
    rho_profile_func : fonction
        Fonction de profil de densité (ex: rho_profile_model1 ou model2)
    t_arr : array
        Tableau des temps (s)
    theta : float
        Angle polaire (rad)
    v_pm_min, v_pm_max, v_d_min, v_max : float
        Paramètres de vitesses (en unités de c)
    f_d_pol : float
        Contraste polaire
    r_min, r_max : float
        Bornes d'intégration en rayon (m)
    n_r : int
        Nombre de points pour l'intégration radiale
    distance_pc : float
        Distance à la source (parsec)
    tau : float
        Temps caractéristique pour la montée de la mémoire (s)
    model : str
        'exponential' ou 'tanh'
    Retour :
    --------
    h_t : array (len(t_arr),)
        Mémoire linéaire pour chaque temps à l'angle donné
    """
    r_grid = np.linspace(r_min, r_max, n_r)
    dr = np.gradient(r_grid)
    distance_m = distance_pc * PC_SI

    h_t = np.zeros(len(t_arr))

    for it, t in enumerate(t_arr):
        # Intégration de la densité sur r pour obtenir la masse éjectée à cet angle et à cet instant
        mass_theta = 0.0
        v_theta = 0.0
        mass_tot = 0.0
        for ir, r in enumerate(r_grid):
            rho_dyn, rho_pm, rho_tot = rho_profile_func(
                r, t, theta, v_pm_min, v_pm_max, v_d_min, v_max, f_d_pol
            )
            dV = 2 * np.pi * r**2 * np.sin(theta) * dr[ir]  # dV sphérique, symétrie axiale
            mass_theta += rho_tot * dV
            v_loc = r / (t * c) if t > 0 else 0
            v_theta += v_loc * rho_tot * dV
            mass_tot += rho_tot * dV
        # Masse totale à cet angle
        M_ej_theta = mass_theta  # en g
        if M_ej_theta == 0:
            h_t[it] = 0
            continue
        # Vitesse caractéristique
        v_char = v_theta / mass_tot if mass_tot > 0 else 0
        # Conversion masse en kg
        M_ej_theta_kg = M_ej_theta * 1e-3
        # Mémoire linéaire à cet angle (amplitude finale)
        delta_h_theta = 2 * G_SI / (C_SI**4 * distance_m) * M_ej_theta_kg * (v_char * c)**2
        # Modèle phénoménologique pour la montée
        if model == 'exponential':
            h_t[it] = delta_h_theta / (1 + np.exp(-t / tau))
        elif model == 'tanh':
            h_t[it] = delta_h_theta * 0.5 * np.tanh(t / tau) + delta_h_theta * 0.5
        else:
            raise ValueError("model must be 'exponential' or 'tanh'")

    return h_t

def plot_memory_vs_theta(
    rho_profile_func,
    t_arr,
    theta_arr,
    v_pm_min, v_pm_max, v_d_min, v_max,
    f_d_pol=0.01,
    r_min=1e7, r_max=1e10, n_r=2000,
    distance_pc=40e6,
    tau=1e-3,
    model='exponential'
):
    """
    Calcule et affiche la mémoire linéaire phénoménologique pour plusieurs angles theta.
    Affiche une heatmap (t, theta) de la mémoire.
    """
    memory_map = []
    for theta in theta_arr:
        h_t = compute_memory_from_density_profile(
            rho_profile_func, t_arr, theta,
            v_pm_min, v_pm_max, v_d_min, v_max,
            f_d_pol, r_min, r_max, n_r, distance_pc, tau, model
        )
        memory_map.append(h_t)
    memory_map = np.array(memory_map)  # shape (len(theta_arr), len(t_arr))

    plt.figure(figsize=(8, 5))
    extent = [t_arr[0], t_arr[-1], theta_arr[0], theta_arr[-1]]
    plt.imshow(memory_map, aspect='auto', origin='lower', extent=extent, cmap='viridis')
    plt.colorbar(label='Memory $h(t,\\theta)$')
    plt.xlabel('Time (s)')
    plt.ylabel('Theta (rad)')
    plt.title('Phenomenological memory vs time and angle')
    plt.tight_layout()
    plt.show()
    return memory_map

#--- GRB + afterglow modesl from https://arxiv.org/pdf/2301.12590

def memory_initial_acceleration(Ej, theta_ej, phi_ej, d, beta=0.99):
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
    angular_factor = beta**2 * np.sin(theta_ej)**2 / (1 - beta* np.cos(theta_ej))  / np.cos(2 * phi_ej)
    # Convert energy from erg to Joules
    Ej_J = Ej * 1e-7
    # Convert distance from parsec to meters
    d_m = d * PC_SI
    # Calculate the initial memory amplitude
    h_in = (2 * G_SI / c**4) * (Ej_J / d_m) * angular_factor
    return h_in


def memory_afterglow_injection(Pin, T_end, theta_ej, phi_ej, d, beta=0.99):
    """
    Additional memory from the afterglow injection phase (Eq. 12) : h_m 
    Arise from the continuous energy injection into the external medium by the jet after the initial acceleration phase.
    The GW signals in GRB afterglows originate from the shock-accelerated ISM and the synchrotrons emission.
    Parameters:
-----------
    Pin : float
        Power of the energy injection [erg/s] ~ 10^48- 10^50 erg/s
    T_end : float
        Duration of the energy injection phase [s] ~ 10^2 - 10^3 s
    theta_ej : float
        Viewing angle (angle between jet axis and line of sight) [rad]
    phi_ej : float
        Azimuthal angle of the jet in the plane of the sky [rad]
    d : float
        Distance to the source [pc]
    beta : float
        Normalized velocity of the jet (v/c), typically close to 1 for (ultra)relativistic jets
    """
    # Angular factor (same as initial memory)
    angular_factor = beta**2 * np.sin(theta_ej)**2 / (1 - beta* np.cos(theta_ej))  / np.cos(2 * phi_ej)
    # Convert power from erg/s to W
    Pin *=  1e-7
    # Convert Pin to total injected energy over T_end
    E_injected_J = Pin * T_end
    # Convert distance from parsec to meters
    d_m = d * PC_SI
    # Calculate the additional memory amplitude from the afterglow injection
    h_m = (2 * G_SI / c**4) * (E_injected_J / d_m) * angular_factor
    
    return h_m


def memory_total_waveform(t_obs, h_in, h_m, t_m):
    """
    Forme d'onde complète de la mémoire GW (Eq. 17)
    
    Paramètres:
    -----------
    t_obs : array
        Temps d'observation [s]
    h_in : float
        Mémoire de la phase initiale
    h_m : float
        Mémoire additionnelle due à l'injection
    t_m : float
        Temps caractéristique = T_end + R_end*(1-cos(theta_ej))*(1+z)/c
    
    Retourne:
    ---------
    h_total : array
        Amplitude GW mémoire totale
    """
    h_total = np.zeros_like(t_obs)
    
    # Phase de montée linéaire
    mask_rise = (t_obs > 0) & (t_obs <= t_m)
    h_total[mask_rise] = h_in + h_m * (t_obs[mask_rise] / t_m)
    
    # Phase plateau (mémoire permanente)
    mask_plateau = t_obs > t_m
    h_total[mask_plateau] = h_in + h_m
    
    return h_total  
    
  