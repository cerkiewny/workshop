import matplotlib.pyplot as plt
import numpy as np
import warnings
import scipy.fftpack
import math
import sys

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
    return sigFFTPos, freqAxisPos

n = int(sys.argv[1]) + 1
f0 = 20  # [Hz]
dt = 1/1000

t = np.arange(0, 1 + dt, dt)
sig = np.sin(2 * math.pi * f0 * t)
for i in range(2, n):
    sig += np.sin(2 * math.pi * (2* i - 1) * f0 * t) / (2 * i - 1)

plt.subplot(2, 1, 1) 
plt.title("function plot")
plt.plot(t, sig) 
plt.subplot(2, 1, 2) 
fftPlot(sig)
plt.show()
