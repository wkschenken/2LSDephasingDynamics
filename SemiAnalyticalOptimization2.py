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

pi = 3.14159265
N = 7 # pulses; code is currently set to take in an odd number of pulses
T0 = 1
T = T0*(N+1)/2 #final time of pulse sequence
Nf = 10**4#number of data points in frequency
nx = 8 # zero padding time domain signal; nx*Nf = # of data points in fft
dt = T/Nf
ft = np.zeros(Nf) # real-valued time-domain filter function; keep the size at just Nf for now, to be updated later in code


# define time array
timep = np.linspace(0, T, Nf)
timen = np.linspace(-T, 0, Nf)
time = np.append(timep, timep+T)
time = np.append(time, timep+2*T)
time = np.append(time, timep+3*T)
time = np.append(time, timep+4*T)
time = np.append(timen, time)
time = np.append(timen - T, time)
time = np.append(timen - 2*T, time)


# define pulse spacings and pulse timings for a symmetric pulse sequence (no DC component for odd number of pulses)
dtArray = np.zeros(np.uint((N+1)/2)) # only filling out the first half of the pulse spacing; symmetrized on the following lines
pulseTiming = np.zeros(N)

# Check that pulses land in the desired time domain 0<t<T; else, redo the sequence
check = 1
while check>0:
    for tt in range(0, np.size(dtArray) - 1, 1):
        dtArray[tt] = T0*random.random()
    dtArray[-1] = T/2 - np.sum(dtArray)
    if dtArray[-1]>0:
        check = -1
    else:
        check = 1


dtArray = np.append(dtArray, np.flip(dtArray))
pulseTiming[0] = dtArray[0]
for ii in range(1, N, 1):
    pulseTiming[ii] = dtArray[ii] + pulseTiming[ii-1]

pulses = np.ones(pulseTiming.shape[-1]) # for plotting

# calculate f(t) for the sequences defined above
pulseIndex = 0
for tt in range(0, Nf, 1):
    if tt*dt<pulseTiming[pulseIndex]:
        ft[tt] = (-1)**pulseIndex
    elif tt*dt>pulseTiming[-1]:
        ft[tt] = (-1)**(pulseIndex+1)
    else:
        pulseIndex += 1

# zero pad f(t)
ft = np.append(np.zeros(Nf), ft)
ft = np.append(np.zeros(Nf), ft)
ft = np.append(np.zeros(Nf), ft)
ft = np.append(ft, np.zeros(Nf))
ft = np.append(ft, np.zeros(Nf))
ft = np.append(ft, np.zeros(Nf))
ft = np.append(ft, np.zeros(Nf))


# calculate the filter function by fourier transforming f(t)
ff = fft(ft)/(nx*Nf)
frequency_axis = fftfreq(nx*Nf, d=dt)
df = frequency_axis[2] - frequency_axis[1]
FF = np.abs(ff)**2
FF = FF/np.sqrt(np.dot(FF, FF))


# Define noise spectrum; trying a few different functions
S0 = 1
sigmaNoise = 0.5
mu = 1
T = 0.01
p = 5
# noise = S0*np.exp(-(frequency_axis**2)/(2*sigmaNoise**2)) + (1/4)*(S0*np.exp(-((frequency_axis - 6*sigmaNoise)**2)/(2*sigmaNoise**2)) + S0*np.exp(-((frequency_axis + 6*sigmaNoise)**2)/(2*sigmaNoise**2)))
# noise = S0/(1+np.exp((np.abs(frequency_axis)-mu)/T))
# noise = np.exp(-((frequency_axis)**p)/sigmaNoise)
# noise = np.zeros(frequency_axis.shape[-1])
# for ff in range(frequency_axis.shape[-1]):
#     if np.abs(frequency_axis[ff])<mu:
#         noise[ff] = 1
#     elif np.abs(frequency_axis[ff])>1.5*mu and np.abs(frequency_axis[ff])<2*mu:
#         noise[ff] = 1
# noise = noise + S0*np.exp(-(((frequency_axis - 5*sigmaNoise)**(2))/(2*sigmaNoise**2))) + S0*np.exp(-(((frequency_axis + 5*sigmaNoise)**(2))/(2*sigmaNoise**2)))
noise = np.exp(-np.abs(frequency_axis)/sigmaNoise) + (1/4)*((1+(frequency_axis - 4)**2)**(-1) + (1+(frequency_axis + 4)**2)**(-1))
noise_scaled = (np.max(FF)/np.max(noise))*noise


# plot results before optimization procedure
#
# plot1 = plt.figure(1)
# plt.plot(pulseTiming)
# plt.xlabel('Time')
# plt.title('Pulse timings pre-optimization')

plot2 = plt.figure(1)
plt.stem(pulseTiming, pulses)
plt.xlabel('Time')
plt.title('Pulse sequence pre-optimization')

plot3 = plt.figure(2)
plt.plot(time, ft)
plt.xlabel('Time')
plt.ylabel('Filter Function f(t)')

plot4 = plt.figure(3)
plt.plot(frequency_axis, FF, label = 'Filter Function')
plt.plot(frequency_axis, noise_scaled, label = 'Noise spectrum')
plt.xlabel('Frequency')
plt.title('Filter functions pre-optimization')
plt.legend()
plt.xlim(-5, 5)
plt.ylim(0, 1.2*np.max(FF))

# plot5 = plt.figure(5)
# plt.plot(a, label = 'a')
# plt.plot(b, label = 'b')
# plt.legend()
# plt.xlabel('Pulse interval')

plt.show()



# run optimization procedure

epsilon = dt # proportional to how much to shift the pulses after each optimization step

# M = np.uint((N+1)/2 + 3) # number of 'Harmonics' to change the pulse timings by
M=25

iterations = 50
chiArray = np.zeros(iterations)

chiArray[0] = np.dot(noise, FF)

for it in range(1, iterations, 1):
    # check the set of possible updates to pulse timing, indexed by n
    print("{}% finished".format(100*it/iterations))
    bestChi = chiArray[it-1]
    updated = 0 # oftentimes the code gets stuck at local minima. Check if it updates after the ith iteration.
    for m in range(1, M):
        # print("m={}".format(m))
        for sgn in [0, 1]:
            # update sequence according to the best harmonic indexed by m
            ffTrial = np.zeros(nx*Nf, dtype = complex)
            pulseTimingTrial = np.zeros(pulseTiming.shape[-1])
            ftTrial = np.zeros(ft.shape[-1])
            for nn in range(0, N, 1):
                pulseTimingTrial[nn] = pulseTiming[nn] - ((-1)**sgn)*(2*pi*m*epsilon/T)*np.sin(pi*m*pulseTiming[nn]/T)*np.cos(pi*m*pulseTiming[nn]/T)
            pulseIndex = 0
            for tt in range(0, Nf, 1):
                if tt*dt<pulseTimingTrial[pulseIndex]:
                    ftTrial[tt] = (-1)**pulseIndex
                elif tt*dt>pulseTimingTrial[-1]:
                    ftTrial[tt] = (-1)**(pulseIndex+1)
                else:
                    pulseIndex += 1
            ffTrial = fft(ftTrial)/(nx*Nf)
            FFTrial = np.abs(ffTrial)**2
            FFTrial = FFTrial/np.sqrt(np.dot(FFTrial, FFTrial))
            chiTrial = np.dot(noise, FFTrial)

            if chiTrial<bestChi:
                pulseTiming = pulseTimingTrial
                ft = ftTrial
                ff = ffTrial
                FF = FFTrial
                bestChi = chiTrial
                updated = 1
    # if there has not been any update, add a random perturbation to try to kick it out of the local minimum.
    if updated == 0 and it!=iterations-1:
        for nn in range(0, N, 1):
            rt = 5*epsilon*np.random.randn()
            if pulseTiming[nn]+rt>0 and pulseTiming[nn]+rt<T:
                pulseTiming[nn] = pulseTiming[nn] + rt
        pulseIndex = 0
        for tt in range(0, Nf, 1):
            if tt*dt<pulseTimingTrial[pulseIndex]:
                ftTrial[tt] = (-1)**pulseIndex
            elif tt*dt>pulseTimingTrial[-1]:
                ftTrial[tt] = (-1)**(pulseIndex+1)
            else:
                pulseIndex += 1
        ffTrial = fft(ftTrial)/(nx*Nf)
        FFTrial = np.abs(ffTrial)**2
        FFTrial = FFTrial/np.sqrt(np.dot(FFTrial, FFTrial))
        chiTrial = np.dot(noise, FFTrial)
        bestChi = chiTrial
        chiArray[it] = bestChi
    else:
        chiArray[it] = bestChi



# plot the results

noise_scaled = (np.max(FF)/np.max(noise))*noise

plot6 = plt.figure(6)
plt.plot(chiArray)
plt.xlabel('Iteration')
plt.ylabel('Chi')
plt.yscale("log")


plot7 = plt.figure(7)
plt.plot(frequency_axis, FF, label = 'Filter function')
plt.plot(frequency_axis, noise_scaled, label = 'Noise spectrum')
plt.xlabel('Frequency')
plt.title('Filter functions post-optimization')
plt.legend()
plt.xlim(-5, 5)
plt.ylim(0, 1.2*np.max(FF))

plot8 = plt.figure(8)
plt.stem(pulseTiming, pulses)
plt.xlabel('Time')
plt.title('Pulse sequence post-optimization')

plt.show()
