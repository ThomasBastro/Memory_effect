import numpy as np
from scipy.signal.windows import tukey, hann
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from lal import MSUN_SI, MTSUN_SI, PC_SI, C_SI, G_SI

# Signal processing utilities

def fft_signal(time, signal):
    """Compute the FFT of a given signal"""
    N = len(signal)
    T = time[1] - time[0]
    yf = fft(signal)
    xf = fftfreq(N, T)[:N // 2]
    return xf, 2.0 / N * np.abs(yf[0:N // 2])

def align_signal(signal, time):
    """Align the signal so that the merger (maximum amplitude) is at t=0"""
    amplitude = np.abs(signal)
    merger_idx = np.argmax(amplitude)
    merger_time = time[merger_idx]
    t_aligned = time - merger_time
    return t_aligned, signal


def apply_window(signal, window_type='tukey', alpha=0.2):
    """
    Apply a window (Tukey or Hann) to the signal, covering the entire signal
    window_type: 'tukey' or 'hann'
    alpha: parameter for the tukey window (alpha = 0 is rectangular, alpha = 1 is hann)
    """
    n_signal = len(signal)
    if window_type == 'tukey':
        window = tukey(n_signal, alpha)
    elif window_type == 'hann':
        window = hann(n_signal)
    else:
        raise ValueError("Window type must be 'tukey' or 'hann'")
    return signal * window

def pad_signal(signal, time, r):
    """
    Pad the signal and time array with zeros to the right.
    r: ratio between the number of padding samples and the original data samples (as in III.B https://arxiv.org/pdf/2406.16636#page=10.46).
    """
    pad_size = int(np.round(r * len(signal)))
    if pad_size > 0:
        dt = time[1] - time[0]
        signal_padded = np.pad(signal, (0, pad_size), 'constant')
        time_padded = np.concatenate([time, time[-1] + dt * np.arange(1, pad_size + 1)])
        return time_padded, signal_padded
    else:
        return time, signal

def bandpass_filter(signal, fs, lowcut, highcut, order=4):
    """Filter the signal with a bandpass filter between lowcut and highcut frequencies."""
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 1 - 1e-6)
    if low >= high:
        raise ValueError("lowcut must be less than highcut and both < Nyquist")
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# Compute the memory using spin-weighted spherical harmonics
# + PyCBC waveform extraction 
def fac(n):
    """Calculate n!"""
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def dlms(l, m, s, Theta):
    """Wigner d-matrix elements"""
    sq = np.sqrt(fac(l + m) * fac(l - m) * fac(l + s) * fac(l - s))
    d = 0.
    for k in range(max(0, m - s), min(l + m, l - s) + 1):
        d += (-1.)**k * np.sin(Theta / 2.)**(2. * k + s - m) * np.cos(Theta / 2.)**(2. * l + m - s - 2. * k) / (
            fac(k) * fac(l + m - k) * fac(l - s - k) * fac(s - m + k))
    return sq * d

def sYlm(s, l, m, Theta, Phi):
    """Spin-weighted spherical harmonics"""
    res = (-1.)**(-s) * np.sqrt((2. * l + 1) / (4 * np.pi)) * dlms(l, m, -s, Theta)
    if res == 0:
        return 0
    else:
        return complex(res * np.cos(m * Phi), res * np.sin(m * Phi))


def compute_memory(signal, distance, dt):
    """Compute the GW nonlinear memory based on the (l=2, m=2) dominant mode of the signal"""
    h22 = signal / sYlm(-2, 2, 2, np.pi/2, np.pi/2) # check for a factor of sqrt(2) !
    hdot = np.gradient(h22, dt)
    hdot_c = np.conjugate(hdot)
    integrand = hdot * hdot_c
    cumint = np.cumsum(integrand) * dt
    prefactor = (distance * 1e6 * PC_SI / C_SI) * (np.sin(np.pi/2))**2 * (17 + np.cos(np.pi/2)**2) / (192 * np.pi) 
    hmem = prefactor * cumint
    return hmem


def get_hp_hc_waveform(approximant, mass1, mass2, distance, delta_t=1.0/4096, f_lower = 20.0, inclination=np.pi/2,
                       spin_1z=0.0, spin_2z=0.0, spin_1y=0.0, spin_2y=0.0, spin_1x=0.0, spin_2x=0.0, phase=0.0):
    """
    Generate hp and hc waveforms using PyCBC's get_td_waveform
    Returns: time array, hp, hc (numpy arrays)
    Arguments:
    - approximant: waveform approximant string
    - mass1, mass2: component masses in solar masses
    - distance: distance to the source in Mpc
    - delta_t: time step in seconds
    - f_lower: starting frequency in Hz
    - inclination: inclination angle in radians
    - spin_1z, spin_2z: dimensionless spins along z-axis
    - spin_1y, spin_2y: dimensionless spins along y-axis
    - spin_1x, spin_2x: dimensionless spins along x-axis
    - phase: initial phase in radians
    """
    from pycbc.waveform import get_td_waveform
    hp, hc = get_td_waveform(
        approximant=approximant,
        mass1=mass1,
        mass2=mass2,
        delta_t=delta_t,
        f_lower=f_lower,
        distance=distance,
        inclination=inclination,
        spin_1z=spin_1z, spin_2z=spin_2z,
        spin_1y=spin_1y, spin_2y=spin_2y,
        spin_1x=spin_1x, spin_2x=spin_2x,
        phase=phase
    )
    time = hp.sample_times.numpy()
    hp = hp.numpy()
    hc = hc.numpy()
    return time, hp, hc