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