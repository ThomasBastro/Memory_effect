import numpy as np
import matplotlib.pyplot as plt
from lal import C_SI, G_SI, PC_SI, MSUN_SI

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
    v = r / (t*C_SI)  # velocity in units of c
    eta_theta = eta(theta, f_d_pol)
    t_day = t / 86400  # t in days
    rho_floor = 1e-17 * t_day**-3  # 1e-20g/cm^3 = 1e-17 kg/m^3
    if v_pm_max > v_d_min:
        raise ValueError("In this model, v_pm_max must be less than v_d_min !")
    # Dynamical ejecta
    if v_d_min <= v < 0.4:
        rho_dyn = eta_theta * r**(-4) * t**(-3)
    elif 0.4 <= v < v_max:
        # Compute the normalization factor at the transition velocity (v = 0.4 c)
        r_transition = 0.4 * t * C_SI
        norm = r_transition**(-4) * t**(-3) / (r_transition**(-8) * t**(-3))
        rho_dyn = norm * eta_theta * r**(-8) * t**(-3)
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
        rho_floor = 1e-17 * t_day**-3  # 1e-20g/cm^3 = 1e-17 kg/m^3 

        # Dynamical ejecta
        if v_d_min <= v < 0.4:
            rho_dyn = eta_theta * r**-4 * t**-3
        elif 0.4 <= v < v_max:
            r_transition = 0.4 * t * c
            norm =  r_transition**-4 * t**-3 / (r_transition**-8 * t**-3)
            rho_dyn = norm * eta_theta * r**-8 * t**-3
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

def compute_ejecta_mass(
    rho_profile_func,
    t,
    theta,
    v_pm_min, v_pm_max, v_d_min, v_max,
    f_d_pol=0.01, n_r=200
):
    """
    Compute the cumulative mass of the ejecta at a given angle theta by integrating the density profile over radius (in spherical coordinates).
    Parameters:
    - rho_profile_func: 
        Function that computes the density profile (e.g., rho_profile_model1 or rho_profile_model2)
    - t: 
        Final time at which we evaluate the mass [s] 
    - theta: 
        Polar angle at which to compute the mass [rad]
    - v_pm_min, v_pm_max, v_d_min, v_max: 
        Velocity parameters (in units of c) of the ejecta components
    - f_d_pol: 
        Density contrast for polar region (default 0.01)
    - n_r: 
        Number of radial points for integration (default 200)
    Returns:
    - mass_theta_dyn: cumulative mass of dynamical ejecta at angle theta [kg]
    - mass_theta_pm: cumulative mass of post-merger ejecta at angle theta [kg]
    - mass_theta: cumulative total mass at angle theta [kg]
    """
    r_max = v_max * C_SI * t
    r_min = 0.1 * C_SI * t  # avoid starting at r=0 to prevent singularities in density
    r_grid = np.linspace(r_min, r_max, n_r)
    dr = np.gradient(r_grid)
    mass_theta_dyn = 0.0
    mass_theta_pm = 0.0
    mass_theta = 0.0

    mass_dyn_arr = []
    mass_pm_arr = []
    mass_tot_arr = []
    for idx, r in enumerate(r_grid):
        rho_dyn, rho_pm, rho_tot = rho_profile_func(
            r, t, theta, v_pm_min, v_pm_max, v_d_min, v_max, f_d_pol
        )
        dV = 2 * np.pi * r**2 * np.sin(theta) * dr[idx]  # dV sphérique
        mass_theta_dyn += rho_dyn * dV
        mass_theta_pm += rho_pm * dV
        mass_theta += rho_tot * dV
        # Store intermediate values for plotting & transform r to velocity in units of c
        mass_dyn_arr.append((r / (t * c), mass_theta_dyn))
        mass_pm_arr.append((r / (t * c), mass_theta_pm))
        mass_tot_arr.append((r / (t * c),mass_theta))

    # Convert lists to numpy arrays for easier handling
    mass_dyn_arr = np.array(mass_dyn_arr)
    mass_pm_arr = np.array(mass_pm_arr)
    mass_tot_arr = np.array(mass_tot_arr)

    return mass_dyn_arr, mass_pm_arr, mass_tot_arr

def compute_total_ejecta_mass(
    rho_profile_func,
    t,
    v_pm_min, v_pm_max, v_d_min, v_max,
    f_d_pol=0.01, n_r=200, n_theta=100, n_t=100
):
    """
    Compute the total mass of the ejecta by integrating the density profile over radius, all polar angles θ, and over time from 0 to t.
    Returns:
    - total_mass_dyn: total mass of dynamical ejecta [kg]
    - total_mass_pm: total mass of post-merger ejecta [kg]
    - total_mass: total mass [kg]
    """
    t_grid = np.linspace(1e-6, t, n_t)  # Avoid t=0 to prevent division by zero
    dt = np.gradient(t_grid)
    total_mass_dyn = 0.0
    total_mass_pm = 0.0
    total_mass = 0.0

    for i_t, t_i in enumerate(t_grid):
        r_max = v_max * C_SI * t_i
        r_min = 0.1 * C_SI * t_i
        r_grid = np.linspace(r_min, r_max, n_r)
        dr = np.gradient(r_grid)
        theta_grid = np.linspace(0, np.pi, n_theta)
        dtheta = np.gradient(theta_grid)

        for i_theta, theta in enumerate(theta_grid):
            sin_theta = np.sin(theta)
            for i_r, r in enumerate(r_grid):
                rho_dyn, rho_pm, rho_tot = rho_profile_func(
                    r, t_i, theta, v_pm_min, v_pm_max, v_d_min, v_max, f_d_pol
                )
                dV = 2 * np.pi * r**2 * sin_theta * dr[i_r] * dtheta[i_theta]
                total_mass_dyn += rho_dyn * dV * dt[i_t]
                total_mass_pm += rho_pm * dV * dt[i_t]
                total_mass += rho_tot * dV * dt[i_t]

    return total_mass_dyn, total_mass_pm, total_mass

def compute_total_ejecta_mass_norm(
    rho_profile_func,
    t,
    v_pm_min, v_pm_max, v_d_min, v_max,
    f_d_pol=0.01, n_r=200, n_theta=100, n_t=100,
    M_dyn_target=0.003, M_pm_target=0.02, time_sat=10
):
    """
    Compute the total mass of the ejecta by integrating the density profile over radius, all polar angles θ, and over time from 0 to t.
    The result is normalized so that at t=time_sat days, the dynamical and post-merger masses match the given values.
    Parameters:
    - rho_profile_func: function that computes the density profile (e.g., rho_profile_model1 or rho_profile_model2)
    - t: final time at which to compute the mass [s]
    - v_pm_min, v_pm_max, v_d_min, v_max: velocity parameters (in units of c) of the ejecta components
    - f_d_pol: density contrast for polar region (default 0.01)
    - n_r: number of radial points for integration (default 200)
    - n_theta: number of polar angle points for integration (default 100)
    - n_t: number of time points for integration (default 100)
    - M_dyn_target: target mass for dynamical ejecta at time_sat days (in solar masses)
    - M_pm_target: target mass for post-merger ejecta at time_sat days (in solar masses)
    - time_sat: time in days at which to match the target masses (default 10 days, typical time for mass estimates in kilonova observations)
    Returns:
    - total_mass_dyn: total mass of dynamical ejecta [Msun]
    - total_mass_pm: total mass of post-merger ejecta [Msun]
    - total_mass: total mass [Msun]
    """
    M_dyn_target = M_dyn_target * MSUN_SI
    M_pm_target = M_pm_target * MSUN_SI
    t_norm = time_sat * 86400  # 10 days in seconds

    # First, compute unnormalized masses at t_norm
    t_grid_norm = np.linspace(1e-6, t_norm, n_t)
    dt_norm = np.gradient(t_grid_norm)
    mass_dyn_raw = 0.0
    mass_pm_raw = 0.0

    for i_t, t_i in enumerate(t_grid_norm):
        r_max = v_max * C_SI * t_i
        r_min = 0.1 * C_SI * t_i
        r_grid = np.linspace(r_min, r_max, n_r)
        dr = np.gradient(r_grid)
        theta_grid = np.linspace(0, np.pi, n_theta)
        dtheta = np.gradient(theta_grid)

        for i_theta, theta in enumerate(theta_grid):
            sin_theta = np.sin(theta)
            for i_r, r in enumerate(r_grid):
                rho_dyn, rho_pm, _ = rho_profile_func(
                    r, t_i, theta, v_pm_min, v_pm_max, v_d_min, v_max, f_d_pol
                )
                dV = 2 * np.pi * r**2 * sin_theta * dr[i_r] * dtheta[i_theta]
                mass_dyn_raw += rho_dyn * dV * dt_norm[i_t]
                mass_pm_raw += rho_pm * dV * dt_norm[i_t]

    # Compute normalization factors
    norm_dyn = M_dyn_target / mass_dyn_raw if mass_dyn_raw > 0 else 1.0
    norm_pm = M_pm_target / mass_pm_raw if mass_pm_raw > 0 else 1.0

    # Now compute normalized masses for requested t
    t_grid = np.linspace(1e-6, t, n_t)
    dt = np.gradient(t_grid)
    total_mass_dyn = 0.0
    total_mass_pm = 0.0
    total_mass = 0.0

    for i_t, t_i in enumerate(t_grid):
        r_max = v_max * C_SI * t_i
        r_min = 0.1 * C_SI * t_i
        r_grid = np.linspace(r_min, r_max, n_r)
        dr = np.gradient(r_grid)
        theta_grid = np.linspace(0, np.pi, n_theta)
        dtheta = np.gradient(theta_grid)

        for i_theta, theta in enumerate(theta_grid):
            sin_theta = np.sin(theta)
            for i_r, r in enumerate(r_grid):
                rho_dyn, rho_pm, rho_tot = rho_profile_func(
                    r, t_i, theta, v_pm_min, v_pm_max, v_d_min, v_max, f_d_pol
                )
                dV = 2 * np.pi * r**2 * sin_theta * dr[i_r] * dtheta[i_theta]
                total_mass_dyn += norm_dyn * rho_dyn * dV * dt[i_t]
                total_mass_pm += norm_pm * rho_pm * dV * dt[i_t]
                total_mass += (norm_dyn * rho_dyn + norm_pm * rho_pm) * dV * dt[i_t]

    return total_mass_dyn/MSUN_SI, total_mass_pm/MSUN_SI, total_mass/MSUN_SI


def compute_memory_from_density_profile(
    rho_profile_func,
    t,
    theta,
    v_pm_min, v_pm_max, v_d_min, v_max,
    distance_pc=40e6,
    tau=1e-3,
    model='exponential'
):
    """    
    Calculates the phenomenological linear memory for a given angle from a density profile.    
    Parameters:    
    ------------    
    rho_profile_func: function    
        Density profile function (e.g., rho_profile_model1 or model2)
    t: float
        Time at which to compute the memory [s] - time at which the ejecta has expanded and we want to evaluate the memory
    theta: float
        Polar angle (rad)
    v_pm_min, v_pm_max, v_d_min, v_max: float
        Velocity parameters (in units of c) of the ejecta components
    distance_pc: float
        Distance to the source [pc], default 40 Mpc (distance of GW170817)
    tau: float
        Characteristic time for memory rise [s]
    model: str
    'exponential' or 'tanh'
    Return:
    --------
    t_arr: array :
         Time array associated to the memory evolution
    h_t: array :
        Memory amplitude as a function of time (t_arr) for the given angle theta
    """
    r_min = 0.1 * C_SI * t
    r_max = v_max * C_SI * t
    distance_m = distance_pc * PC_SI

    # For each time step, compute the mass of the ejecta at this angle 
    m_ejecta_tot, _, _ = compute_ejecta_mass(
        rho_profile_func, t, theta,
        v_pm_min, v_pm_max, v_d_min, v_max,
    )
    # Masse totale à cet angle
    M_ej = m_ejecta_tot[-1, 1]  # Take total mass in this case - should be modified to give dyn, wind, tot
    print(M_ej)
    # ---
    # Caracteristic velocity based on the maximum radius reached at the last time step (correct ?)
    # Should take the mean ?
    v_char = r_max / t / C_SI # in units of c
    # ---
    # Angular factor to account for the TT gauge analytically 
    angular_fact = np.sin(theta)**2 / (1 - v_char * np.cos(theta)) # Add a factor beta = v/c like this?
    # Linear memory amplitude
    delta_h_theta = 2 * G_SI / (C_SI**4 * distance_m) * M_ej * (v_char * C_SI)**2 * angular_fact
    print(f"delta_h_theta at t={t:.3f} s and theta={theta:.3f} rad: {delta_h_theta:.3e}")
    # As we computed delta_h_theta, we can now apply the time evolution model to get h(t) for each time step
    t_arr = np.logspace(-4, 2, 5000)  # Time array from 0.1 ms 
    if model == 'exponential':
        h_t = delta_h_theta / (1 + np.exp(-t_arr / tau))
    elif model == 'tanh':
        h_t = delta_h_theta * 0.5 * np.tanh(t_arr / tau) + delta_h_theta * 0.5
    else:
        raise ValueError("model must be 'exponential' or 'tanh'")

    return t_arr, h_t

def plot_memory_vs_theta(
    rho_profile_func,
    t_arr,
    theta_arr,
    v_pm_min, v_pm_max, v_d_min, v_max,
    f_d_pol=0.01, n_r=2000,
    distance_pc=40e6,
    tau=1e-3,
    model='exponential'
):
    """
    Plots the final memory amplitude (at the last time step) as a function of the viewing angle theta for a given density profile model.
    This function should be used in pair with compute_memory_from_density_profile to evaluate the memory at different angles and then plot the results.
    Parameters:
------------
    rho_profile_func: function
        Density profile function (e.g., rho_profile_model1 or model2)
    t_arr: array
        Time array for memory evolution (used to compute h(t) for each angle)
    theta_arr: array
        Array of viewing angles (in radians) to evaluate the memory
    v_pm_min, v_pm_max, v_d_min, v_max: float
        Velocity parameters (in units of c) of the ejecta components
    f_d_pol: float
        Density contrast for polar region (default 0.01)
    n_r: int
        Number of radial points for mass integration (default 2000 for better accuracy in memory calculation)
    distance_pc: float
        Distance to the source [pc], default 40 Mpc (distance of GW170817)
    tau: float
        Characteristic time for memory rise [s]
    model: str
        Time evolution model for memory rise ('exponential' or 'tanh')
    """
    memory_map = []
    for theta in theta_arr:
        h_t = compute_memory_from_density_profile(
            rho_profile_func, t_arr, theta,
            v_pm_min, v_pm_max, v_d_min, v_max,
            f_d_pol, n_r, distance_pc, tau, model
        )
        memory_map.append(h_t[-1]) # Final memory at the last time step
    memory_map = np.array(memory_map)
    plt.figure(figsize=(8, 6))
    plt.plot(theta_arr, memory_map, marker='o', color='crimson')
    plt.yscale('log')
    plt.xlabel(r'Viewing angle $\theta$ [rad]')
    plt.ylabel(r'Final linear memory $h$')
    plt.title('Final GW linear memory vs viewing angle')
    plt.minorticks_on()
    plt.show()
    

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
    
  