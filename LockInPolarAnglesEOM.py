import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import imshow
import random
import cmath
from scipy import signal
from scipy.optimize import curve_fit
from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from matplotlib import colors
from scipy.stats import norm
from scipy.stats import cauchy
import matplotlib.mlab as mlab
from scipy import stats
from numpy.fft import fft, ifft
from scipy.fft import fftfreq, rfftfreq

# calculating the full equations of motion for theta and phi as derived by Kotler et al for single-axis control of a 2LS

t0 = 0
tf = 1
dt = 10**(-4)

N = np.int((tf-t0)/dt)

time = np.arange(t0, tf, dt)

# number of pulses in each sequence
n = 10

# dynamical variables for CPMG and UDD

phiUDD = np.zeros(N)
thetaUDD = np.zeros(N)
phiCPMG = np.zeros(N)
thetaCPMG = np.zeros(N)

# and those calculated with the naive FF approach, assuming no change in theta

phiUDDff = np.zeros(N)
phiCPMGff = np.zeros(N)

fUDD = np.zeros(N)
fCPMG = np.zeros(N)

# define the control = Omega[t], its time integral beta = int_0^t Omega[t] dt, and signal = M[t] waveforms

OmegaUDD = np.zeros(N)
OmegaCPMG = np.zeros(N)

BetaUDD = np.zeros(N)
BetaCPMG = np.zeros(N)

M = np.zeros(N)


phiUDD[0] = 0
thetaUDD[0] = 3.14/2
phiCPMG[0] = 0
thetaCPMG[0] = 3.14/2

phiUDDff[0] = 0
phiCPMGff[0] = 0

# pulse errors; deviation from pi pulses
eps = 3.14/2

# Add the pulses in sequentially; j = pulse number
for j in range(0, n):
    print("calculating sequences in time domain")
    # when the jth time interval ends for each sequence
    PulseTimeUDD = tf*(np.sin(3.14*(j+1)/(2*(n+1)))**2)
    PulseTimeCPMG = j*tf/n + tf/(2*n)

    timeErrorUDD = np.min(np.abs(PulseTimeUDD - time))
    timeErrorCPMG = np.min(np.abs(PulseTimeCPMG - time))

    #Define the drive term Omega for instantaneous pulses
    timeIndex = 0
    for t in time:
        if np.abs(t - PulseTimeUDD) < (timeErrorUDD + dt/10):
            OmegaUDD[timeIndex] = (3.14 + eps)/dt
        if np.abs(t - PulseTimeCPMG) < (timeErrorCPMG + dt/10):
            OmegaCPMG[timeIndex] = (3.14 + eps)/dt
        timeIndex+=1


# define M[t] and calculate beta[t] based on Omega[t]
# here say M[t] is a harmonic function with frequency omega, phase delta, and amplitude A

omegaSig = np.round((2*3.14) / (4*(tf - t0)/n), decimals=2) # defined here to be in resonance with the pulse spacing
delta = 0
A = 2

timeIndex = -1
for t in time:
    M[timeIndex] = A*np.cos(omegaSig*t + delta)
    BetaUDD[timeIndex+1] = BetaUDD[timeIndex] + dt*OmegaUDD[timeIndex]
    BetaCPMG[timeIndex+1] = BetaCPMG[timeIndex] + dt*OmegaCPMG[timeIndex]
    timeIndex+=1

for j in range(0, n):

    # time intervals between the jth and (j+1)th pulse
    timeUDDmax = tf*(np.sin(3.14*(j+1)/(2*(n+1)))**2)
    timeUDDmin = tf*(np.sin(3.14*j/(2*(n+1)))**2)
    timeCPMGmin = (j-1)*tf/n + tf/(2*n)
    timeCPMGmax = j*tf/n + tf/(2*n)

    # Calculate the filter functions
    timeIndex = 0
    for t in time:
        if t>0:
            if t>timeUDDmin and t<timeUDDmax:
                fUDD[timeIndex]=np.cos(BetaUDD[timeIndex])
            if t>timeCPMGmin and t<timeCPMGmax:
                fCPMG[timeIndex]=np.cos(BetaCPMG[timeIndex])
        timeIndex+=1




for t in range(0, N-1):
    print("Integrating polar angles")
    phiUDD[t+1] = phiUDD[t] + dt*M[t]*(np.cos(BetaUDD[t]) - np.sin(BetaUDD[t])*np.sin(phiUDD[t])/np.tan(thetaUDD[t]))
    thetaUDD[t+1] = thetaUDD[t] + dt*M[t]*np.sin(BetaUDD[t])*np.cos(phiUDD[t])
    phiCPMG[t+1] = phiCPMG[t] + dt*M[t]*(np.cos(BetaCPMG[t]) - np.sin(BetaCPMG[t])*np.sin(phiCPMG[t])/np.tan(thetaCPMG[t]))
    thetaCPMG[t+1] = thetaCPMG[t] + dt*M[t]*np.sin(BetaCPMG[t])*np.cos(phiCPMG[t])

    phiUDDff[t+1] = phiUDDff[t] + fUDD[t]*M[t]*dt
    phiCPMGff[t+1] = phiCPMGff[t] + fCPMG[t]*M[t]*dt


plt.figure(1)
plt.plot(time, phiUDD, label = 'UDD Full Calculation')
plt.plot(time, phiCPMG, label = 'CPMG Full Calculation')
plt.plot(time, phiUDDff, label = 'UDD FF Calculation')
plt.plot(time, phiCPMGff, label = 'CPMG FF Calculation')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Phase Phi')
plt.title("n={} pulses, signal = {}cos({}t+{}), rotation angle = pi+{}".format(n, A, omegaSig, delta, eps))

plt.figure(2)
plt.plot(time, OmegaUDD*dt/(3.14+eps), label = "UDD Drive")
plt.plot(time, OmegaCPMG*dt/(3.14+eps), label = "CPMG Drive")
plt.plot(time, M/A, label = 'Signal')
plt.legend()
plt.xlabel('Time')
plt.ylabel("Single-Axis Drive Omega_x")


plt.show()
