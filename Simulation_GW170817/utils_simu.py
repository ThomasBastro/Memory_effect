import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import os 
outdir = "gw170817_memory_outputs"

# Conversion factors
erg_to_J = 1e-7
day_to_s = 24 * 3600


# --- First part : Energy distribution on the sky ---

# Energy evolution sanity check function
def sanity_check_energy_evolution(t_days, e_erg, theta, cone_radius = 0.25):
    """
    Analyzes the evolution of energy over time for different angles (north pole, equator, south pole, and intermediate angles)
    Plots the energy vs time for each angle and provides statistics such as total energy, number of photons, and peak time
    Parameters:
    -----------
    t_days : array-like
        Array of time values in days for each photon
    e_erg : array-like
        Array of energy values in erg for each photon
    theta : array-like
        Array of polar angles (in radians) for each photon
    cone_radius : float
        Radius of the cone (in radians) around each reference angle to select photons
        Initially set to 0.25 radians (~14.3 degrees). Increase if needed to include more photons.
    """
    
    # Definitions of reference angles
    angles_ref = {
        'North Pole': 0,
        rf'$\theta=45^\circ$': np.pi/4, 
        'Equator': np.pi/2,
        rf'$\theta=135^\circ$': 3*np.pi/4,
        'South Pole': np.pi
    }
    
    # Binning for time evolution
    t_bins = np.linspace(t_days.min(), t_days.max(), 500) 
    t_centers = 0.5 * (t_bins[1:] + t_bins[:-1]) # Take the bin centers (to be used in plotting)
    
    # Figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    
    # For each reference angle, compute and plot the energy evolution
    for i, (name, theta_ref) in enumerate(angles_ref.items()):
        # We select photons within a cone around the reference angle
        cone_radius = cone_radius  # Adjustable - increase to accommodate a larger cone of photons (in radians)
        mask_direction = np.abs(theta - theta_ref) < cone_radius # Mask for photons within the cone
        
        if not np.any(mask_direction): # Skip if no photons in this direction
            continue
        
        # Histogram: energy (e_erg) vs time (in days) for the selected direction
        energy_vs_time, _ = np.histogram(t_days[mask_direction], 
                                       bins=t_bins, 
                                       weights=e_erg[mask_direction])
        
        # Plot energy vs time for this direction based on the histogram
        axes[i].loglog(t_centers, energy_vs_time, 
                    color=colors[i], linewidth=2, marker='o', markersize=4)
        axes[i].set_xlabel('t(days)')
        axes[i].set_ylabel('Energy (erg)')
        axes[i].set_title(name)
        axes[i].grid(True, alpha=0.3)
        
        # General statistics
        total_energy = np.sum(e_erg[mask_direction]) # Total energy in erg = sum of photons energies
        n_photons = np.sum(mask_direction) # Number of photons in this direction
        peak_time = t_centers[np.argmax(energy_vs_time)] # Time of peak energy emission

        
        # Add these statistics to the plot
        axes[i].text(0.98, 0.98, 
                f'# photons ={n_photons:,}\nE_tot={total_energy:.1e} erg\nPic: {peak_time:.1f}j',
                transform=axes[i].transAxes, 
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Last plot : all directions comparison - same as above but all on the same plot
    axes[5].set_title('All directions comparison')
    for i, (name, theta_ref) in enumerate(angles_ref.items()):
        mask_direction = np.abs(theta - theta_ref) < 0.25
        if np.any(mask_direction):
            energy_vs_time, _ = np.histogram(t_days[mask_direction], 
                                           bins=t_bins, 
                                           weights=e_erg[mask_direction])
            axes[5].loglog(t_centers, energy_vs_time, 
                        color=colors[i], linewidth=2, label=name)
    
    axes[5].set_xlabel('t(days)')
    axes[5].set_ylabel('Energy (erg)')
    axes[5].set_yscale('log')
    axes[5].legend(fontsize=9)
    axes[5].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('energy_evolution_by_angle.png', dpi=150, bbox_inches='tight')
    plt.show()






def analyze_energy_vs_time_and_angle_simpler(t_days, e_erg, theta, n_theta=20, n_time=50, cone_radius=0.25):
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    
    n_theta_bins = n_theta
    n_time_bins = n_time
    theta_bins = np.linspace(0, np.pi, n_theta_bins + 1)
    theta_centers = 0.5 * (theta_bins[1:] + theta_bins[:-1])
    t_bins = np.linspace(t_days.min(), t_days.max(), n_time_bins + 1)
    t_centers = 0.5 * (t_bins[1:] + t_bins[:-1])
    energy_matrix = np.zeros((n_time_bins, n_theta_bins))

    for j, theta_center in enumerate(theta_centers):
        mask_direction = np.abs(theta - theta_center) < cone_radius
        if np.any(mask_direction):
            energy_vs_time, _ = np.histogram(t_days[mask_direction], bins=t_bins, weights=e_erg[mask_direction])
            energy_matrix[:, j] = energy_vs_time

    energy_matrix[energy_matrix == 0] = np.nan

        # 1. Profil temporel à différents angles (cmap inferno)
    plt.figure(figsize=(12, 8))
    T_mesh, Theta_mesh = np.meshgrid(t_centers, np.degrees(theta_centers))
    im1 = plt.pcolormesh(T_mesh, Theta_mesh, energy_matrix.T,
                        norm=LogNorm(vmin=np.nanmin(energy_matrix[energy_matrix > 0]), vmax=np.nanmax(energy_matrix)),
                        cmap='inferno', shading='auto')
    plt.xscale('log')
    plt.xlabel('t [days]')
    plt.ylabel(r'$\theta$ [deg]')
    cbar1 = plt.colorbar(im1, pad=0.02)
    cbar1.set_label('Energy [erg]')
   
    plt.savefig('energy_temporal_profile.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 2. Profil angulaire à différents temps (cmap viridis)
    plt.figure(figsize=(12, 8))
    for idx, color in zip([0, n_time_bins//4, n_time_bins//2, 3*n_time_bins//4, n_time_bins-1], plt.cm.viridis(np.linspace(0,1,5))):
        profile = energy_matrix[idx, :]
        mask_nonzero = ~np.isnan(profile)
        if np.any(mask_nonzero):
            plt.semilogy(np.degrees(theta_centers[mask_nonzero]), profile[mask_nonzero],
                        color=color, linewidth=2, marker='o', markersize=4,
                        label=f't = {t_centers[idx]:.2f} d')
    plt.xlabel(r'$\theta$ [deg]')
    plt.ylabel('Energy [erg]')
    plt.legend(loc='best', frameon=False, fontsize=16, bbox_to_anchor=(1.005, 1))
  
    plt.savefig('energy_angular_profile.png', dpi=150, bbox_inches='tight')
    plt.show()
    


def analyze_energy_vs_time_and_angle(t_days, e_erg, theta, n_theta=20, n_time=50, cone_radius=0.25):
    """
    Performs a comprehensive analysis of energy evolution as a function of time and angle
    Parameters:
    -----------
    t_days : array-like
        Array of time values in days for each photon
    e_erg : array-like
        Array of energy values in erg for each photon
    theta : array-like
        Array of polar angles (in radians) for each photon
    n_theta : int
        Number of angular bins for the analysis
    n_time : int
        Number of temporal bins for the analysis
    cone_radius : float
        Radius of the cone (in radians) around each angle bin to select photons
    """
    
    # Analysis grid parameters
    n_theta_bins = n_theta  # Number of angular bins
    n_time_bins = n_time   # Number of temporal bins
    cone_radius = cone_radius  # Cone radius for angular sampling
    
    # Analysis grids
    theta_bins = np.linspace(0, np.pi, n_theta_bins + 1)
    theta_centers = 0.5 * (theta_bins[1:] + theta_bins[:-1])
    
    t_bins = np.linspace(t_days.min(), t_days.max(), n_time_bins + 1)
    t_centers = 0.5 * (t_bins[1:] + t_bins[:-1])
    
    # Energy matrix (time, angle)
    energy_matrix = np.zeros((n_time_bins, n_theta_bins))
    photon_count_matrix = np.zeros((n_time_bins, n_theta_bins))

    
    # Calculation for each angle
    for j, theta_center in enumerate(theta_centers):
        # Select photons within a cone around this angle
        mask_direction = np.abs(theta - theta_center) < cone_radius
        
        if np.any(mask_direction):
            # Temporal evolution for this angle
            energy_vs_time, _ = np.histogram(t_days[mask_direction], 
                                           bins=t_bins, 
                                           weights=e_erg[mask_direction])
            
            photon_count, _ = np.histogram(t_days[mask_direction], bins=t_bins)
            
            energy_matrix[:, j] = energy_vs_time
            photon_count_matrix[:, j] = photon_count
    
    plt.figure(figsize=(12, 8))
    
    # Mask zero values 
    energy_plot = energy_matrix.copy()
    energy_plot[energy_plot == 0] = np.nan
    
    # Convert axes for display - meshgrid to gather data
    T_mesh, Theta_mesh = np.meshgrid(t_centers, np.degrees(theta_centers))
    
    im = plt.pcolormesh(T_mesh, Theta_mesh, energy_plot.T, 
                       norm=LogNorm(vmin=np.nanmin(energy_plot[energy_plot > 0]), 
                                   vmax=np.nanmax(energy_plot)), # lognorm for better energy variation visualization (so only keep positive values)
                       cmap='hot', shading='auto')
    
    plt.xscale('log')
    plt.xlabel('t(days)', fontsize=16)
    plt.ylabel(r'Viewing Angle $\theta$ (degrees)', fontsize=16)
    plt.title('Energy Evolution', fontsize=18)
    
    # Colorbar
    cbar = plt.colorbar(im, pad=0.02)
    cbar.set_label('Energy (erg)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('energy_heatmap_main.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 4 subplots for analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    #1: Temporal profiles for selected angles
    ax1 = axes[0, 0]
    
    selected_angles_idx = [0, n_theta_bins//4, n_theta_bins//2, 3*n_theta_bins//4, n_theta_bins-1]
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    
    for i, idx in enumerate(selected_angles_idx):
        profile = energy_matrix[:, idx]
        mask_nonzero = profile > 0
        if np.any(mask_nonzero):
            ax1.loglog(t_centers[mask_nonzero], profile[mask_nonzero], 
                          color=colors[i], linewidth=2, marker='o', markersize=4,
                          label=r'$\theta$ = {:.0f}°'.format(np.degrees(theta_centers[idx])))
    
    ax1.set_xlabel('t(days)', fontsize=12)
    ax1.set_ylabel('Energy (erg)', fontsize=12)
    ax1.set_title('Temporal Profiles', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2: Angular profiles for selected times
    ax2 = axes[0, 1]
    
    selected_times_idx = [0, n_time_bins//4, n_time_bins//2, 3*n_time_bins//4, n_time_bins-1]
    
    for i, idx in enumerate(selected_times_idx):
        profile = energy_matrix[idx, :]
        mask_nonzero = profile > 0
        if np.any(mask_nonzero):
            ax2.semilogy(np.degrees(theta_centers[mask_nonzero]), profile[mask_nonzero], 
                        color=colors[i], linewidth=2, marker='s', markersize=4,
                        label='t = {:.2f}d'.format(t_centers[idx]))
    
    ax2.set_xlabel(r'Viewing Angle $\theta$ (degrees)', fontsize=12)
    ax2.set_ylabel('Energy (erg)', fontsize=12)
    ax2.set_title('Angular Profiles', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3: Total energy vs time (integrated over all angles)
    ax3 = axes[1, 0]
    
    total_energy_vs_time = np.sum(energy_matrix, axis=1)
    mask_nonzero = total_energy_vs_time > 0
    
    ax3.loglog(t_centers[mask_nonzero], total_energy_vs_time[mask_nonzero], 
              'black', linewidth=3, marker='o', markersize=3)
    ax3.set_xlabel('t(days)', fontsize=12)
    ax3.set_ylabel('Total Energy (erg)', fontsize=12)
    ax3.set_title('Total Energy vs Time', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Total angular distribution (integrated over all times)
    ax4 = axes[1, 1]
    
    total_energy_vs_angle = np.sum(energy_matrix, axis=0)
    mask_nonzero = total_energy_vs_angle > 0
    
    ax4.semilogy(np.degrees(theta_centers[mask_nonzero]), total_energy_vs_angle[mask_nonzero], 
                'darkred', linewidth=3, marker='s', markersize=4)
    ax4.set_xlabel(r'Viewing Angle $\theta$(degrees)', fontsize=12)
    ax4.set_ylabel('Total Energy (erg)', fontsize=12)
    ax4.set_title('Total Energy vs Angle', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('energy_analysis_subplots.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Global statistics
    print("Total energy: {:.2e} erg".format(np.sum(energy_matrix)))
    print("Global peak time: {:.2f} days".format(t_centers[np.argmax(total_energy_vs_time)]))
    print("Global peak angle: {:.1f}°".format(np.degrees(theta_centers[np.argmax(total_energy_vs_angle)])))
    
    return energy_matrix, t_centers, theta_centers


def energy_dist_all_visualization(t_days, e_erg, theta, n_theta=20, n_time=50, cone_radius=0.25):
    """
    Creates a 3D visualization of energy as a function of time and angle
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Reuse data from previous analysis
    energy_matrix, t_centers, theta_centers = analyze_energy_vs_time_and_angle(t_days, e_erg, theta, n_theta=20, n_time=50, cone_radius=0.25)
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrids for 3D surface
    T_mesh, Theta_mesh = np.meshgrid(t_centers, np.degrees(theta_centers))
    Energy_mesh = energy_matrix.T # Transpose to match dimensions
    
    # Mask zero values
    Energy_mesh[Energy_mesh == 0] = np.nan
    
    # 3D Surface
    surf = ax.plot_surface(T_mesh, Theta_mesh, np.log10(Energy_mesh), 
                          cmap='viridis', alpha=0.8, 
                          linewidth=0)
    
    ax.set_xlabel('t(day)', fontsize=12)
    ax.set_ylabel(r'Viewing Angle $\theta$ (degrees)', fontsize=12)
    ax.set_zlabel('Energy (log(erg))', fontsize=12)
    ax.set_title('3D Energy Distribution', fontsize=14)
    
    # Colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, 
                 label='Energy log(erg)', norm=LogNorm())
    
    plt.savefig('energy_3d_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig, ax

def compute_hemispheric_anisotropy_tot(pix, npix, nside,e_erg, plot=True):
    """
    Computes the North vs South energy anisotropy over the entire dataset
    Parameters:
    -----------
    pix : array-like
        Array of HEALPix pixel indices for each photon
    npix : int
        Total number of HEALPix pixels
    nside : int
        HEALPix nside parameter
    e_erg : array-like
        Array of energy values in erg for each photon
    plot : bool
        Whether to generate plots of the anisotropy maps
    """

    # Integrated energy per pixel (erg) final
    energy_map = np.bincount(pix, weights=e_erg, minlength=npix).astype(float)  # bincount to sum energies per pixel

    # Pixel vectors
    x, y, z = hp.pix2vec(nside, np.arange(npix))
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # North/South mask (z>0 north, z<0 south). Equatorial plane (z==0) is excluded
    mask_north = z > 0.0
    mask_south = ~mask_north # mask_south is the opposite

    E_north = np.sum(energy_map[mask_north])
    E_south = np.sum(energy_map[mask_south])

    # Relative anisotropy (North - South) / (North + South); value between -1 and 1
    denom = (E_north + E_south)
    anisotropy_rel = (E_north - E_south) / denom * 100 if denom != 0 else 0.0

    # Equatorial mirror map: for each pixel, find the symmetric pixel with respect to the equator
    mirror_idx = hp.vec2pix(nside, x, y, -z)  # invert z sign for equatorial mirror
    mirror_idx = np.asarray(mirror_idx, dtype=int)

    # Difference pixel vs mirror pixel (north - south)
    diff_map = energy_map - energy_map[mirror_idx]

    if plot:
        # Mirror difference map (linear): positive => pixel has more energy than its equatorial mirror
        # Symmetrize scale for RdBu
        absmax = np.nanmax(np.abs(diff_map))

        hp.mollview(diff_map, title="Energy - Equatorial-mirror (North - South local)", unit="erg",
                    cmap="RdBu_r", min=-absmax, max=absmax)
        plt.savefig(os.path.join(outdir, "energy_north_minus_south_mollview.png"), dpi=150, bbox_inches="tight")
        plt.show()

        # Clearer: local relative map ((E - E_mirror) / (E + E_mirror))
        denom_local = energy_map + energy_map[mirror_idx]  # Local sum of north/south energies
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_map = np.where(denom_local > 0, (energy_map - energy_map[mirror_idx]) / denom_local, 0.0)
            # Relative map: value between -1 and 1, where +1 means all energy in north pixel, -1 in south pixel, 0 means equality

        hp.mollview(rel_map, title="Local relative asymmetry (E - E_mirror)/(E + E_mirror)",
                    unit="", cmap="RdBu_r", min=-1, max=1)
        plt.savefig(os.path.join(outdir, "energy_north_minus_south_relative_mollview.png"), dpi=150, bbox_inches="tight")
        plt.show()
        # See anisotropy_rel for global north/south asymmetry
        print(f"Global North-South energy asymmetry: {anisotropy_rel:.2%} (North > South)" if anisotropy_rel > 0 else f"Global North-South energy asymmetry: {anisotropy_rel:.2%} (South > North)")

# --- Second part : Memory signal on the sky ---

def build_TT_projector(n_observer):
    """
    Returns the TT (Transverse-Traceless) projector for a given observer direction vector n.
    
    Parameters:
    -----------
    n_observer : array-like
        3D observer direction vector (should be a unit vector - will be normalized if not)
    Returns:
    --------
    Lambda : ndarray
        4D array representing the TT projector Lambda_ijkl
    """
    n = n_observer 
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)  # Normalize the observer direction vector
    
    # Transverse projector: P_ij = δ_ij - n_i n_j, which projects onto the plane perpendicular to n
    P = np.eye(3) - np.outer(n, n)  # P = δ_ij - n_i n_j, where n is the observer direction
    
    # Full TT projector: Lambda_ijkl = P_ik P_jl - 0.5 P_ij P_kl
    # This ensures the tensor is both transverse (perpendicular to n) and traceless (trace = 0)
    Lambda = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    Lambda[i, j, k, l] = P[i, k] * P[j, l] - 0.5 * P[i, j] * P[k, l]
    return Lambda

def TT_projection(T, n_observer):
    """
    Projects a 3x3 tensor T onto its transverse-traceless (TT) part with respect to the observer direction n_observer.
    
    This function applies the TT projector to remove non-physical components, resulting in a tensor that represents the physical degrees of freedom in GWs.
    """
    Lambda = build_TT_projector(n_observer)  # Get the TT projector for the given observer direction
    # Use einsum to contract the projector with the tensor: T_TT_ij = Lambda_ijkl T_kl
    T_TT = np.einsum('ijkl,kl->ij', Lambda, T)  # einsum performs the contraction over indices k and l, keeping i and j
    return T_TT

# --- Core function : Memory signal computation ---

from lal import G_SI, PC_SI, C_SI
def compute_linear_memory_TT(t_days, e_erg, obs_dir, pix, npix, nside, distance_Mpc=40.0, nbins_time=1000):
    """
    Computes the linear memory effect in TT gauge for a given observer direction based on photon data
    Parameters:
    -----------
    t_days : array-like
        Array of time values in days for each photon
    e_erg : array-like
        Array of energy values in erg for each photon
    obs_dir : array-like
        3D observer direction unit vector
    pix : array-like
        Array of HEALPix pixel indices for each photon
    npix : int
        Total number of HEALPix pixels
    nside : int
        HEALPix nside parameter
    distance_Mpc : float
        Distance to the source in Megaparsecs (default: 40.0 Mpc - GW170817 distance)
    nbins_time : int
        Number of temporal bins for the analysis (default: 1000)
    Returns:
    --------
    h_memory_map : array-like
        Array of memory contributions for each pixel
    h_memory_total : float
        Total memory signal integrated over all pixels
    """
    # Conversions + constants
    erg_to_J = 1e-7
    day_to_s = 24 * 3600
    r_m = distance_Mpc * 1e6 * PC_SI

    t_s = t_days * day_to_s
    e_J = e_erg * erg_to_J

    # Temporal binning
    t_bins = np.linspace(t_s.min(), t_s.max(), nbins_time)
    dt = np.median(np.diff(t_bins))  # median time step
    nt = len(t_bins) - 1  # Number of time intervals

    # Energy per pixel and per time step
    maps_J = np.zeros((nt, npix))
    for i in range(nt):
        mask = (t_s >= t_bins[i]) & (t_s < t_bins[i + 1])
        np.add.at(maps_J[i], pix[mask], e_J[mask])  # maps_J is a matrix (nt, npix) giving energy per pixel at each time

    # Luminosity map
    dE_dt = maps_J / dt  # J/s
    dOmega = 4.0 * np.pi / npix  # steradian per pixel
    dL_dOmega = dE_dt / dOmega  # Luminosity per pixel (J/s/sr)

    # Temporal integral
    cumulative_dL = np.cumsum(dL_dOmega * dt, axis=0)  # Sum over time axis
    L_final = cumulative_dL[-1]  # Final integrated luminosity = last element

    # TT memory calculation
    # Pre-compute unit vectors for each pixel
    
    factor = 4.0 * G_SI / (r_m * C_SI**4)
    
    vecs = np.array(hp.pix2vec(nside, np.arange(npix))).T  # Returns unit vectors for each pixel in shape (npix, 3)
    # Will be used to compute cos(θ) : where θ is the angle between the observer position and the radiation source point
    # and to compute the TT projection of n_i n_j
    eps = 1e-8  # To avoid division by zero in 1/(1 - cos(θ))
    
    h_memory_map = np.zeros(npix)  # Map of memory contribution for each pixel
    for p in range(npix):
        if L_final[p] == 0:  # If no energy emitted in this direction, skip
            continue
            
        n_vec = vecs[p]
        cos_theta = np.dot(n_vec, obs_dir) 
        den = 1.0 - cos_theta
        if den < eps:
            den = eps  # Compute denominator
        
        # TT projection on vectors n - The tensor to project is n_i n_j (outer product of n_vec with itself)
        n_tensor_TT = TT_projection(np.outer(n_vec, n_vec), obs_dir)  # Compute n_i n_j (3x3 tensor) projected TT where Lambda depends on n_observer
        
        # Compute memory tensor
        h_tensor = factor * L_final[p] * n_tensor_TT / den
        
        # + polarization: h_+ = -h_x
        h_plus = -h_tensor[0, 0] 
        h_memory_map[p] = h_plus  # Memory contribution for this pixel

    # Total memory calculation = sum over all pixels
    h_memory_total = np.sum(h_memory_map)

    return h_memory_map, h_memory_total 

from lal import G_SI, PC_SI, C_SI
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

def compute_memory_vs_time_TT(t_s, e_J, npix, pix, obs_dir, nside=64, distance_Mpc=40.0, nbins_time=1000, plot=True):
    """
    Computes the linear TT memory effect as a function of time
    """

    r_m = distance_Mpc * 1e6 * PC_SI

    deltaOmega = 4 * np.pi / npix

    # Same as for the memory calculation
    t_bins = np.linspace(t_s.min(), t_s.max(), nbins_time)
    t_centers = 0.5 * (t_bins[1:] + t_bins[:-1])
    t_centers_days = t_centers / day_to_s
    dt = np.median(np.diff(t_bins))
    nt = len(t_bins) - 1

    maps_J = np.zeros((nt, npix))
    for i in range(nt):
        mask = (t_s >= t_bins[i]) & (t_s < t_bins[i + 1])
        np.add.at(maps_J[i], pix[mask], e_J[mask])

    dE_dt = maps_J / dt  
    dL_dOmega = dE_dt / deltaOmega

    vecs = np.array(hp.pix2vec(nside, np.arange(npix))).T
    factor = 4.0 * G_SI / (r_m * C_SI**4)
    eps = 1e-8

    tt_tensors = np.zeros((npix, 3, 3))
    geometric_factors = np.zeros(npix)
    
    for p in range(npix):
        n_vec = vecs[p]
        cos_theta = np.dot(n_vec, obs_dir)
        den = 1.0 - cos_theta
        if den < eps:
            den = eps
        
        geometric_factors[p] = 1.0 / den
        tt_tensors[p] = TT_projection(np.outer(n_vec, n_vec), obs_dir)

    # Compute memory as a function of time
    h_memory_vs_time = np.zeros(nt)
    cumulative_dL = np.zeros((nt, npix))
    
    # Cumulative time integration
    for t_idx in range(nt):
        # Compute L(t) for each time step given the integral
        if t_idx == 0:
            cumulative_dL[t_idx] = dL_dOmega[t_idx] * dt
        else:
            cumulative_dL[t_idx] = cumulative_dL[t_idx-1] + dL_dOmega[t_idx] * dt

        # Compute the total memory at this time step
        h_total_at_time = 0.0

        for p in range(npix):
            if cumulative_dL[t_idx, p] == 0:
                continue

            # Compute the memory tensor for this pixel
            h_tensor = factor * cumulative_dL[t_idx, p] * tt_tensors[p] * geometric_factors[p]
            h_plus = -h_tensor[0, 0]  # Plus polarization

            # Sum for each pixel in npix at time t_idx
            h_total_at_time += h_plus

        h_memory_vs_time[t_idx] = h_total_at_time

    # --- Create the plot ---
    if plot:
        plt.figure(figsize=(12, 10))
        
        # Filter valid values to avoid plotting NaNs due to division/TT or errors
        valid_mask = np.isfinite(h_memory_vs_time)
        # Compute the observer's angle in spherical coordinates
        obs_theta = np.arccos(obs_dir[2])
        obs_phi = np.arctan2(obs_dir[1], obs_dir[0])
        
        if np.any(valid_mask):
            # Plot
            plt.subplot(2, 1, 1)
            plt.semilogx(t_centers_days[valid_mask], h_memory_vs_time[valid_mask], 
                    'b-', linewidth=2)
            plt.xlabel('t(days)')
            plt.ylabel('Memory (strain)')
            plt.title(' Memory Amplitude Evolution - TT Gauge for observer at ' +
                    rf'$\theta$={np.degrees(obs_theta):.1f}°, $\phi$={np.degrees(obs_phi):.1f}°')
            plt.grid(True, alpha=0.3)

            # Display the final cumulative value
            max_memory = h_memory_vs_time[valid_mask][-1]
            plt.text(
                0.98, 0.02,
                f"Max memory (end): {max_memory:.2e}",
                transform=plt.gca().transAxes,
                fontsize=16,
                color='black',
                ha='right',
                va='bottom',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )
            
        # Save
        #outdir = "gw170817_memory_outputs"
        #import os
        #os.makedirs(outdir, exist_ok=True)
        #plt.savefig(f"{outdir}/memory_vs_time_TT_linear.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    return t_centers_days, h_memory_vs_time

from scipy.fft import fft, fftfreq
from scipy.signal import windows

# FFT of the memory as a function of time - adds interpolation to reach up to 1Hz frequency
def memory_fft(t_centers_days, h_memory_vs_time, window_type='hann', plot_results=True):
    """
    Computes the FFT of a time series of memory as a function of time
    Parameters:
    -----------
    t_centers_days : array-like
        Time centers in days for the memory signal
    h_memory_vs_time : array-like
        Memory signal as a function of time (strain)
        
    Remark : 
    The parameters t_centers_days and h_memory_vs_time should be arrays of the same length provided by compute_memory_vs_time_TT
    
    window_type: str or None
        Type of window to apply before FFT (e.g., 'hann', 'hamming', 'blackman'), or None for no window
    plot_results: whether to plot the FFT results
    Returns:
    --------
    frequencies : array-like
        Frequencies corresponding to the FFT result
    fft_h : array-like
        FFT amplitude of the memory signal
    """
    
    # Filter finite values
    valid_mask = np.isfinite(h_memory_vs_time) & np.isfinite(t_centers_days) # Ensure both arrays are valid (should be the same from previous function)
    t_valid = t_centers_days[valid_mask]
    h_valid = h_memory_vs_time[valid_mask]
    
    # Convert to seconds to get frequencies in Hz
    day_to_s = 24 * 3600
    t_seconds = t_valid * day_to_s
    
    # Check for regular sampling -> interpolate onto a regular grid
    t_regular = np.linspace(t_seconds.min(), t_seconds.max(), len(t_seconds))
    h_interp = np.interp(t_regular, t_seconds, h_valid)
    t_seconds = t_regular
    h_valid = h_interp
    dt_mean = np.mean(np.diff(t_seconds))
    
    # If current Nyquist < 1 Hz, resample to reach at least 1 Hz
    desired_fmax = 1.0  # Hz
    dt_required = 1.0 / (2.0 * desired_fmax)  # dt <= 0.5 s for Nyquist >= 1 Hz
    if dt_mean > dt_required: 
        print(f"Resampling time series to reach Nyquist >= {desired_fmax} Hz")
        # build a time grid with step dt_required (keep same duration)
        t_new = np.arange(t_seconds[0], t_seconds[-1], dt_required)
        # ensure the last value covers the end
        if t_new.size == 0 or t_new[-1] < t_seconds[-1]:
            t_new = np.append(t_new, t_seconds[-1])
        h_resampled = np.interp(t_new, t_seconds, h_valid) # Interpolate for resampled versions
        t_seconds = t_new
        h_valid = h_resampled
        dt_mean = np.mean(np.diff(t_seconds))
    
    N = len(h_valid)
    
    # Apply a window to reduce spectral leakage - optional
    if window_type:
        if window_type == 'hann':
            window = windows.hann(N)
        elif window_type == 'hamming':
            window = windows.hamming(N)
        elif window_type == 'blackman':
            window = windows.blackman(N)
        else:
            raise ValueError(f"{window_type} not supported")
        h_windowed = h_valid * window
    else:
        h_windowed = h_valid

    # Compute the FFT
    fft_h = fft(h_windowed)
    frequencies = fftfreq(N, dt_mean)[:N//2]  # Corresponding frequencies
    fft_h = 2.0/N * np.abs(fft_h[:N//2])  # Take positive half and scale
    
    # Remove zero frequency for logplot
    pos_mask = frequencies > 0
    frequencies_plot = frequencies[pos_mask]
    fft_h_plot = fft_h[pos_mask]
    
    # Plot 
    if plot_results:
        plt.figure(figsize=(15, 10))
        plt.loglog(frequencies_plot, fft_h_plot, 'r-', linewidth=2, label=(f'{window_type}' if window_type else None))
        if window_type:
            plt.legend()
        plt.tick_params(top=True, right=True, axis='both', which='major', labelsize=12, direction='in', length=6, width=1.2)
        plt.xlabel('f [Hz]', fontsize=14)
        plt.ylabel('Amplitude (strain)', fontsize=14)
        plt.title('FFT memory signal')
        # limit x-axis to keep a useful range (at least up to 1 Hz, use 200 Hz for better visualization)
        plt.xlim(1e-6, 200)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return frequencies_plot, fft_h_plot