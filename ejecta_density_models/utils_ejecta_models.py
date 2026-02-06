import numpy as np
import matplotlib.pyplot as plt
from lal import C_SI

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
    rho_floor = 1e-20 * t_day**-3  # g/cm^3
    if v_pm_max > v_d_min:
        raise ValueError("In this model, v_pm_max must be less than v_d_min !")
    # Dynamical ejecta
    if v_d_min <= v < 0.4:
        rho_dyn = eta_theta * r**-4 * t**-3
    elif 0.4 <= v < v_max:
        rho_dyn = eta_theta * r**-8 * t**-3
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
        im = ax.contourf(VX, VZ, np.log10(data + 1e-30), levels=levels, cmap=cmap)
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