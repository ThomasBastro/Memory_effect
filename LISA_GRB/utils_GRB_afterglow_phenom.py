import numpy as np
import matplotlib.pyplot as plt
from lal import MSUN_SI, PC_SI, C_SI, G_SI
#--- GRB + afterglow modesl from https://arxiv.org/pdf/2301.12590

def memory_initial_acceleration_GRB(Ej, theta_ej, phi_ej, d, beta=0.99):
    """
    Initial memory from the jet acceleration phase (Eq. 11) of https://arxiv.org/pdf/2301.12590: h_in
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
    
    Returns:
    ---------
    h_in : float
        Amplitude of the initial memory from the jet acceleration phase
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
    Additional memory from the afterglow injection phase  https://arxiv.org/pdf/2301.12590: h_m 
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
    -------
    Returns:
    h_m : float
        Additional memory amplitude from the afterglow injection phase
    
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
    
    --------
    Returns:
    t_obs : array-like
        Time array for the observed memory signal [s]
    h_total : array-like
        Total memory signal combining the initial and afterglow contributions at each time in t_obs
        
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
  
    return t_obs, h_total

def total_waveform_fft(h_in, h_m, t_end_injection, theta_ej, radius=0.01):
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
    
    --------
    Returns:
    f : array-like
        Frequency array for the FFT of the total memory signal [Hz]
    fft_total : array-like
        FFT of the total memory signal combining the initial and afterglow contributions at each frequency in f
    """
    f = np.logspace(-6, 6, int(1e6))  # Frequencies from 1e-6 to 10000 Hz with 10,000 points
    t_m = t_end_injection + (radius* PC_SI) * (1-np.cos(theta_ej))/ C_SI
    
    a = h_m/(4*np.pi**2 * f**2 * t_m)
    b = h_in/(2*np.pi* f)
    
    fourier_h_square = 4*a**2 * (np.sin(np.pi * f * t_m))**4 +(a*np.sin(2*np.pi * f * t_m) + b)**2
    
    return f, np.sqrt(fourier_h_square)