import matplotlib.pyplot as plt
import numpy as np
import warnings
import scipy.fftpack

def fftPlot(sig, dt=None, plot=True):
    # Here it's assumes analytic signal (real signal...) - so only half of the axis is required

    if dt is None:
        dt = 1
        t = np.arange(0, sig.shape[-1])
        xLabel = 'samples'
    else:
        t = np.arange(0, sig.shape[-1]) * dt
        xLabel = 'freq [Hz]'

    if sig.shape[0] % 2 != 0:
        warnings.warn("signal preferred to be even in size, autoFixing it...")
        t = t[0:-1]
        sig = sig[0:-1]

    sigFFT = np.fft.fft(sig) / t.shape[0]  # Divided by size t for coherent magnitude

    freq = np.fft.fftfreq(t.shape[0], d=dt)

    # Plot analytic signal - right half of frequence axis needed only...
    firstNegInd = np.argmax(freq < 0)
    freqAxisPos = freq[0:firstNegInd]
    sigFFTPos = 2 * sigFFT[0:firstNegInd]  # *2 because of magnitude of analytic signal

    if plot:
        plt.plot(freqAxisPos, np.abs(sigFFTPos))
        plt.xlabel(xLabel)
        plt.ylabel('mag')
        plt.title('Analytic FFT plot')

    return sigFFTPos, freqAxisPos

if __name__ == "__main__":
    samples = 1000
    dt = 1 / samples 

    # Build a signal within Nyquist - the result will be the positive FFT with actual magnitude
    f0 = 200  # [Hz]
    t = np.arange(0, 1 + dt, dt)
    sig = 1 * np.sin(2 * np.pi * f0 * t) + \
        10 * np.sin(2 * np.pi * f0 / 2 * t) + \
        3 * np.sin(2 * np.pi * f0 / 4 * t) +\
        7.5 * np.sin(2 * np.pi * f0 / 5 * t)
    plots = 6
    plt.subplot(plots, 1, 1) 
    plt.title("function plot")
    plt.plot(t, sig) 

    plt.subplot(plots, 1, 2) 
    plt.title("function fft plot")
    fftPlot(sig, dt)

    plt.subplot(plots, 1, 3) 
    noise =  sig + 30 * np.sin(2 * np.pi * f0 * 32 * t)
    plt.title("function with noise")
    plt.plot(t, noise) 

    plt.subplot(plots, 1, 4) 
    plt.title("fft, function with noise")
    fftPlot(noise, dt)

    plt.subplot(plots, 1, 5) 
    plt.title("function with noise fft, filtered")
    sigFFTFiltered = np.fft.fft(sig) / t.shape[0] * ( 1 - np.heaviside(t - 0.5, 1))
    plt.plot(t, sigFFTFiltered)

    plt.subplot(plots, 1, 6) 
    plt.title("function with reversed")
    plt.plot(t,np.fft.ifft(sigFFTFiltered))

    plt.show()
