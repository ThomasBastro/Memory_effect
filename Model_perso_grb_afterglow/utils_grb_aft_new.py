import numpy as np
import matplotlib.pyplot as plt

from lal import PC_SI, C_SI, G_SI

def grb_afterglow_model(t, E_GRB, T_90, E_aft, beta, t_dec, t_jet_break, d, theta, theta_j):
    
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
    f = np.asarray(f)
    dt_aft = t_jet - t_dec

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
    axs[0].axvline(x=T_90, color='g', linestyle='--', label='GRB Saturation')
    axs[0].axvline(x=t_dec, color='b', linestyle='--', label='Afterglow Start')
    axs[0].axvline(x=t_jet_break, color='m', linestyle='--', label='Afterglow Saturation')
    axs[0].axhline(y=delta_h_GRB, color='c', linestyle='--', label='GRB Memory Level')
    axs[0].axhline(y=delta_h_aft + delta_h_GRB, color='y', linestyle='dotted', label='Afterglow Memory Level')
    axs[0].set_xlabel('t [s]')
    axs[0].set_ylabel(r'$h_+$')
    axs[0].set_title('Temporal Domain')
    axs[0].legend()
    axs[0].set_yscale('log')
    
    axs[0].set_xlim([t[0], min(t[-1], t_jet_break*1.2)])

    # Domaine fréquentiel
    axs[1].loglog(f, 2*f*h_ft, label='Total GRB + Afterglow', color='crimson', lw=2, zorder = 5)
    axs[1].loglog(f, 2*f*h_GRB, label='GRB', alpha=0.7, lw=1.5, ls = '-.', zorder = 10, color = 'teal')
    axs[1].loglog(f, 2*f*h_aft, label='Afterglow', alpha=0.7, lw=1.5, ls='-.', zorder =10, color = 'navy')
    axs[1].plot(f, 2*h_ft[0] * (f / f[0])**(-1), 'k--', label=r'$f^{-1}$')
    axs[1].set_xlabel('f [Hz]')
    axs[1].set_ylabel(r'$h_c~[f]$')
    axs[1].set_title('Frequency Domain')

    axs[1].legend()


    return fig, axs