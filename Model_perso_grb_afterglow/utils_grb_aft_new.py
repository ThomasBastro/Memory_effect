import numpy as np
import matplotlib.pyplot as plt

PC_SI = 3.086e16  # parsec en mètres
G_SI = 6.67430e-11  # constante gravitationnelle en m^3 kg^-1 s^-2
C_SI = 299792458  # vitesse de la lumière en m/s

def grb_afterglow_model(t, E_GRB, T_90, E_aft, beta, t_dec, t_jet_break, d, theta, theta_0):
    
    E_GRB *= 1e-7 # convert erg to J
    E_aft *= 1e-7 # convert erg to J
    d *=  PC_SI # convert pc to m
    
    delta_h = np.zeros_like(t)
    
    # GRB contribution
    delta_h_GRB =(G_SI/C_SI**4) * (2 * E_GRB * beta**2 /d)  * (np.sin(theta)**2 / (1-(beta *np.cos(theta)))) 
    for i in range(len(t)):
        if t[i] < 0:
            delta_h[i] = 0
        elif 0 <= t[i] < T_90:
            delta_h[i] = delta_h_GRB * (t[i]/T_90)
        elif t[i] >= T_90:
            delta_h[i] = delta_h_GRB
    # Afterglow contribution
    delta_h_aft = (G_SI/C_SI**4) * (2 * E_aft * beta**2 /d)  * (np.sin(theta)**2 / (1-(beta *np.cos(theta)))) 
    for i in range(len(t)):
        if t[i] < t_dec:
            delta_h[i] += 0
        elif t_dec <= t[i] < t_jet_break:
            delta_h[i] += delta_h_aft * ((t[i]-t_dec)/(t_jet_break-t_dec))
        elif t[i] >= t_jet_break:
            delta_h[i] += delta_h_aft
    return delta_h, delta_h_GRB, delta_h_aft

def fft(f, h_GRB, T_90, h_aft, t_dec, t_jet):
    """
    Calcul de |h̃(f)| pour le modèle GRB + afterglow (approximation analytique).
    """
    if t_dec > t_jet:
        t_dec, t_jet = t_jet, t_dec  # s'assurer que t_dec < t_jet
    f = np.asarray(f)
    dt_aft = np.abs(t_jet - t_dec)

    def fourier_amplitude_squared(f, h, tau):
        a = h/(4*np.pi**2 * f**2 * tau)
        H = 4* a**2 * np.sin(np.pi * f * tau)**4 + (a * np.sin(2 * np.pi * f * tau))**2
        return H

    H_GRB = fourier_amplitude_squared(f, h_GRB, T_90)
    H_aft = fourier_amplitude_squared(f, h_aft, dt_aft)
    H_total = H_GRB + H_aft + 2 * H_aft * H_GRB  # terme croisé
    H_GRB = np.sqrt(H_GRB)
    H_aft = np.sqrt(H_aft)
    H_total = np.sqrt(H_total)
    return H_total, H_GRB, H_aft

def plot_grb_afterglow(t, delta_h, f, h_ft, h_GRB, h_aft, T_90, t_dec, t_jet_break, delta_h_GRB, delta_h_aft):
    """
    Affiche le signal mémoire dans le domaine temporel (gauche) et fréquentiel (droite).
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Domaine temporel
    axs[0].plot(t, delta_h, label='GRB + Afterglow', color='k', lw=2.5)

    axs[0].fill_betweenx([delta_h_GRB, delta_h_GRB + delta_h_aft], t_dec, t_jet_break, color='navy', alpha=0.3, label='Afterglow Growth')

    axs[0].fill_betweenx([0, delta_h_GRB], 0, T_90, color='forestgreen', alpha=0.3, label='GRB Growth')
    # add text with double arrow for T_90, t_dec and t_jet_break
    axs[0].annotate('', xy=(T_90, 1e-25), xytext=(0, 1e-25), arrowprops=dict(arrowstyle='<->', color='forestgreen'))
    axs[0].annotate(r'$T_{90}$', xy=(T_90/2, 1e-25), xytext=(T_90/2,1e-25), ha='center', color='forestgreen', va='bottom', fontsize=12)
    # add arrpw at t = T90 to show the jump in amplitude of GRB
    ymin = axs[0].get_ylim()[0]

    axs[0].annotate('', xy=(T_90, delta_h_GRB), xytext=(T_90, delta_h_GRB/25), arrowprops=dict(arrowstyle='->', color='forestgreen'))
    axs[0].annotate(r'$\Delta h_{\rm GRB}$', xy=(T_90, delta_h_GRB*0.55), xytext=(T_90, delta_h_GRB*0.55), ha='left', color='forestgreen', va='bottom', fontsize=14)
    # same for t_jet_break - t_dec starting from 0 to t_jet_break - t_dec
    center = np.sqrt(t_dec * t_jet_break)
    axs[0].annotate('', xy=(t_dec, 1e-24), xytext=(t_jet_break, 1e-24), arrowprops=dict(arrowstyle='<->', color='navy'))
    axs[0].annotate(r'$t_{jet} - t_{dec}$', xy=(center, 1e-24), xytext=(center,1e-24), ha='center', color='navy', va='bottom', fontsize=14)
    # arrow to show the jump in amplitude of afterglow at t_jet_break
    axs[0].annotate('', xy=(t_jet_break, delta_h_GRB + delta_h_aft), xytext=(t_jet_break, delta_h_GRB ), arrowprops=dict(arrowstyle='->', color='navy'))
    axs[0].annotate(r'$\Delta h_{\rm aft}$', xy=(t_jet_break, (delta_h_GRB + delta_h_aft)*0.5), xytext=(t_jet_break, (delta_h_GRB + delta_h_aft)*0.5), ha='left', color='navy', va='bottom', fontsize=14)
    

    
    axs[0].set_xlabel('t [s]')
    axs[0].set_ylabel(r'$h_+$')
    axs[0].set_title('Temporal Domain')
    axs[0].legend(loc ='lower right', frameon=False, fontsize=14)
    axs[0].set_yscale('log')
    
    axs[0].set_xlim([t[0], min(t[-1], t_jet_break*1.2)])

    # Domaine fréquentiel
    axs[1].loglog(f, 2*f*h_ft, label='Total GRB + Afterglow', color='k', lw=3, zorder=2, alpha=0.85)
    axs[1].loglog(f, 2*f*h_GRB, label='GRB', alpha=0.75, lw=1.5, ls='-.', zorder=5, color='forestgreen')
    axs[1].loglog(f, 2*f*h_aft, label='Afterglow', alpha=0.75, lw=1.5, ls=':', zorder=5, color='navy')
 
    axs[1].set_xlabel('f [Hz]')
    axs[1].set_ylabel(r'$h_c~[f]$')
    axs[1].set_title('Frequency Domain')

    return fig, axs

# --- Mass distribution in the jet ---
# Models from http://arxiv.org/abs/1302.5713 eq. III.1 
from tqdm import tqdm


def f_theta(theta, theta_0=0.1, Gamma=100, jet_type='uniform'):
    
    theta_core = 1.0 / Gamma

    if jet_type == 'uniform':
        return np.where(theta <= theta_0, 1.0, 0.0)
    elif jet_type == 'structured':
        f = np.zeros_like(theta)
        mask_core = theta < theta_core
        mask_wing = (theta >= theta_core) & (theta < theta_0)
        f[mask_core] = 1.0
        f[mask_wing] = (Gamma * theta[mask_wing])**-2
        return f
    else:
        raise ValueError("jet_type doit être 'uniform' ou 'structured'")



def grb_memory_jet_grb(
    t, E_GRB, T_90, E_aft, beta, t_dec, t_jet_break, d_pc,
    theta_v, theta_0, Gamma, jet_type='uniform', n_theta=100, n_phi=100
):
    d = d_pc * PC_SI

    # 1. Grilles angulaires
    eps = 1e-8  # éviter theta=0 exact
    thetas = np.linspace(eps, theta_0, n_theta)
    phis = np.linspace(0, 2 * np.pi, n_phi)
    dtheta = thetas[1] - thetas[0]
    dphi = phis[1] - phis[0]

    # 2. Normalisation de f(theta)
    f_vals = f_theta(thetas, theta_0, Gamma, jet_type)
    # Intégrale de f sur la sphère : \int f sin(theta) dtheta dphi
    norm = np.sum(f_vals * np.sin(thetas)) * dtheta * dphi * n_phi
    f_vals /= norm
 
    
    

    # 3. Intégration 2D correcte
    delta_h_jet_grb = 0.0
 
    for i in range(n_theta):
     

        theta = thetas[i]
        if theta < eps:
            continue

        for j in range(n_phi):
            phi = phis[j]
            cos_xi = (
                np.cos(theta) * np.cos(theta_v)
                + np.sin(theta) * np.sin(theta_v) * np.cos(phi)
            )
            xi = np.arccos(np.clip(cos_xi, -1, 1))
            _, dh_grb_xi, _ = grb_afterglow_model(
                t, E_GRB, T_90, E_aft, beta, t_dec, t_jet_break, d_pc, xi, theta_0
            )
            delta_h_jet_grb += (
                f_vals[i]
                * dh_grb_xi
                * np.sin(theta)
                * dtheta
                * dphi
            )

    # Signal total (jet GRB + afterglow)

    t_array = np.array(t)
    delta_h = np.zeros_like(t_array)
    
    # Contribution Prompt (GRB)
    mask_prompt = (t_array >= 0) & (t_array < T_90)
    mask_after_prompt = (t_array >= T_90)
    delta_h[mask_prompt] = delta_h_jet_grb * (t_array[mask_prompt] / T_90)
    delta_h[mask_after_prompt] = delta_h_jet_grb

    # 5. Contribution Afterglow (Attention : Formule point-source ici)
    # Si tu veux être rigoureux, delta_h_aft devrait aussi être dans l'intégrale ci-dessus
    prefactor_aft = (G_SI / C_SI**4) * (2 * E_aft * beta**2 / d)
    geom_term = (np.sin(theta_v)**2 / (1 - beta * np.cos(theta_v)))
    delta_h_aft_total = prefactor_aft * geom_term

    mask_aft = (t_array >= t_dec) & (t_array < t_jet_break)
    mask_after_aft = (t_array >= t_jet_break)
    
    delta_h[mask_aft] += delta_h_aft_total * ((t_array[mask_aft] - t_dec) / (t_jet_break - t_dec))
    delta_h[mask_after_aft] += delta_h_aft_total

    return delta_h, delta_h_jet_grb, delta_h_aft_total