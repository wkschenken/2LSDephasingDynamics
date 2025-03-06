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


def LorentzianNoiseComparisonVaryWidth():

    # calculate coherence at the end of a pulse sequence (CPMG or UDD) = C(tf) vs width of the lorentzian noise spectrum
    # using the gaussian phase approximation to calculate C(tf) = exp(-<phi^2>/2)
    N=10**2 #number of widths to calculate
    gammaMax = 175
    gammaArray = np.linspace(0, gammaMax, N)

    n = 10 # number of pulses in the sequence

    A = 10 #arbitrary amplitude of noise spectrum S(omega) = A*L(omega) where L is a normalized Lorentzian
           # chosen to give reasonable coherence values somewhere around 0.5 for the noise parameters used here

    # time/frequency parameters
    tf = 1
    deltat = 10**(-4)
    Nt = np.int(tf/deltat)
    ZP = 6 # zero-padding the time domain sequence to decrease domega in the frequency domain
    timeArray = np.linspace(-ZP*tf, ZP*tf, 2*ZP*Nt+1) #zero-padding the time array for better frequency resolution

    # Arrays to hold the coherence under a CPMG or UDD sequence
    CoherenceArrayUDD = np.zeros(N)
    CoherenceArrayCPMG = np.zeros(N)

    # arrays to be used to visualize sequences...
    pulseUDD = np.ones(n)
    pulseCPMG = np.ones(n)

    PulseTimeUDD = np.zeros(n)
    PulseTimeCPMG = np.zeros(n)

    # ... and time domain filter functions.
    fUDD = np.zeros(2*ZP*Nt+1)
    fCPMG = np.zeros(2*ZP*Nt+1)

    # j = pulse number
    for j in range(0, n):

        # when the jth time interval ends for each sequence
        PulseTimeUDD[j] = tf*(np.sin(3.14*(j+1)/(2*(n+1)))**2)
        PulseTimeCPMG[j] = j*tf/n + tf/(2*n)

        # time intervals between the jth and (j+1)th pulse
        timeUDDmax = tf*(np.sin(3.14*(j+1)/(2*(n+1)))**2)
        timeUDDmin = tf*(np.sin(3.14*j/(2*(n+1)))**2)
        timeCPMGmin = (j-1)*tf/n + tf/(2*n)
        timeCPMGmax = j*tf/n + tf/(2*n)

        #chose the sign of the time domain filter function accordingly
        timeIndex = 0
        for t in timeArray:
            if t>0:
                if t>timeUDDmin and t<timeUDDmax:
                    fUDD[timeIndex]=(-1)**j
                if t>timeCPMGmin and t<timeCPMGmax:
                    fCPMG[timeIndex]=(-1)**j
            timeIndex+=1

    # Frequency domain Filter functions, normalized by Nt/2 by the defn of FFT given by scipy
    FFUDD = 2*fft(fUDD)/(2*Nt*ZP + 1)
    FFCPMG = 2*fft(fCPMG)/(2*Nt*ZP + 1)
    FFUDD = np.abs(FFUDD)**2
    FFCPMG = np.abs(FFCPMG)**2
    frequency_axis = 2*3.14*fftfreq(2*ZP*Nt+1, d=deltat) #angular frequency
    domega = frequency_axis[2]-frequency_axis[1]


    # For each width, calculate the coherence
    gammaIndex = 0

    for gamma in gammaArray:

        # Define the overlap integral <phi^2>
        phi2UDD = 0
        phi2CPMG = 0

        # integrate over omega
        omegaIndex = 0
        for omega in frequency_axis:
            # update the integral <phi^2>;
            # since only summing over positive frequencies here, and the integrand is symmetric, add in a factor of 2 in the final expression for coherence
            # but that factor of 2 is cancelled by the other 2 in C(t)=exp(-<phi^2>/2)
            # phi2UDD+= FFUDD[omegaIndex]*((A/3.14)*(gamma/2)/((omega**2) + (gamma/2)**2))*domega
            # phi2CPMG+= FFCPMG[omegaIndex]*((A/3.14)*(gamma/2)/((omega**2) + (gamma/2)**2))*domega
            phi2UDD+= FFUDD[omegaIndex]*(A/(np.exp((omega-31.4)/gamma)+1))/(gamma*np.log(2))
            phi2CPMG+= FFCPMG[omegaIndex]*(A/(np.exp((omega-31.4)/gamma)+1))/(gamma*np.log(2))
            omegaIndex+=1

        # final coherence values under the Gaussian distributed phase approximation
        coherenceUDD = np.exp(-phi2UDD)
        coherenceCPMG = np.exp(-phi2CPMG)
        CoherenceArrayUDD[gammaIndex] = coherenceUDD
        CoherenceArrayCPMG[gammaIndex] = coherenceCPMG
        gammaIndex+=1
        print('{} percent finished with the gamma loop'.format(np.round(100*gammaIndex/N), decimals=2))

    plot1 = plt.figure(1)
    plt.plot(gammaArray, CoherenceArrayUDD, label = 'UDD-{}'.format(n))
    plt.plot(gammaArray, CoherenceArrayCPMG, label = 'CPMG-{}'.format(n))
    plt.legend()
    plt.xlabel('Lorentzian Width [2pi x Hertz]]')
    plt.ylabel('Coherence @ t_final (Gaussian phase approx)')

    plot2 = plt.figure(2)
    plt.plot(frequency_axis, FFUDD, label = 'UDD-{} Frequency Domain Filter Function'.format(n))
    plt.plot(frequency_axis, FFCPMG, label = 'CPMG-{} Frequency Domain Filter Function'.format(n))
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.xlabel('Angular Frequency [2pi x Hertz]')
    plt.ylabel('Filter Function |F|^2')

    plot3 = plt.figure(3)
    plt.plot(timeArray, fUDD, label = 'UDD-{} Time Domain Filter Function'.format(n))
    plt.plot(timeArray, fCPMG, label = 'CPMG-{} Time Domain Filter Function'.format(n))
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('f(t)')

    plot4 = plt.figure(4)
    plt.stem(PulseTimeUDD, pulseUDD, label = 'UDD-{} Sequence'.format(n),linefmt='blue', markerfmt='o')
    plt.stem(PulseTimeCPMG, pulseCPMG, label = 'CPMG-{} Sequence'.format(n),linefmt='grey', markerfmt='o')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Pulses')

    plt.show()

    return

def LorentzianNoiseComparisonVarySharpCutoff():

    # calculate coherence at the end of a pulse sequence (CPMG or UDD) = C(tf) vs width of the lorentzian noise spectrum
    # using the gaussian phase approximation to calculate C(tf) = exp(-<phi^2>/2)
    N=10**2 #number of widths to calculate
    omegacutMax = 1000
    omegacutArray = np.linspace(0, omegacutMax, N)

    n = 10 # number of pulses in the sequence

    A = .0002 #arbitrary amplitude of noise spectrum S(omega) = A*L(omega) where L is a normalized Lorentzian
           # chosen to give reasonable coherence values somewhere around 0.5 for the noise parameters used here

    # time/frequency parameters
    tf = 1
    deltat = 10**(-4)
    Nt = np.int(tf/deltat)
    ZP = 6 # zero-padding the time domain sequence to decrease domega in the frequency domain
    timeArray = np.linspace(-ZP*tf, ZP*tf, 2*ZP*Nt+1) #zero-padding the time array for better frequency resolution

    # Arrays to hold the coherence under a CPMG or UDD sequence
    CoherenceArrayUDD = np.zeros(N)
    CoherenceArrayCPMG = np.zeros(N)

    # arrays to be used to visualize sequences...
    pulseUDD = np.ones(n)
    pulseCPMG = np.ones(n)

    PulseTimeUDD = np.zeros(n)
    PulseTimeCPMG = np.zeros(n)

    # ... and time domain filter functions.
    fUDD = np.zeros(2*ZP*Nt+1)
    fCPMG = np.zeros(2*ZP*Nt+1)

    # j = pulse number
    for j in range(0, n):

        # when the jth time interval ends for each sequence
        PulseTimeUDD[j] = tf*(np.sin(3.14*(j+1)/(2*(n+1)))**2)
        PulseTimeCPMG[j] = j*tf/n + tf/(2*n)

        # time intervals between the jth and (j+1)th pulse
        timeUDDmax = tf*(np.sin(3.14*(j+1)/(2*(n+1)))**2)
        timeUDDmin = tf*(np.sin(3.14*j/(2*(n+1)))**2)
        timeCPMGmin = (j-1)*tf/n + tf/(2*n)
        timeCPMGmax = j*tf/n + tf/(2*n)

        #chose the sign of the time domain filter function accordingly
        timeIndex = 0
        for t in timeArray:
            if t>0:
                if t>timeUDDmin and t<timeUDDmax:
                    fUDD[timeIndex]=(-1)**j
                if t>timeCPMGmin and t<timeCPMGmax:
                    fCPMG[timeIndex]=(-1)**j
            timeIndex+=1

    # Frequency domain Filter functions, normalized by Nt/2 by the defn of FFT given by scipy
    FFUDD = 2*fft(fUDD)/Nt
    FFCPMG = 2*fft(fCPMG)/Nt
    FFUDD = np.abs(FFUDD)**2
    FFCPMG = np.abs(FFCPMG)**2
    frequency_axis = 2*3.14*fftfreq(2*ZP*Nt+1, d=deltat) #angular frequency
    domega = frequency_axis[2]-frequency_axis[1]


    # For each width, calculate the coherence
    omegacutIndex = 0

    for omegacut in omegacutArray:

        # Define the overlap integral <phi^2>
        phi2UDD = 0
        phi2CPMG = 0

        # integrate over omega
        omegaIndex = 0
        for omega in frequency_axis:
            # update the integral <phi^2>;
            # since only summing over positive frequencies here, and the integrand is symmetric, add in a factor of 2 in the final expression for coherence
            # but that factor of 2 is cancelled by the other 2 in C(t)=exp(-<phi^2>/2)
            if omega<omegacut:
                phi2UDD+= FFUDD[omegaIndex]*A*np.abs(omega)*domega
                phi2CPMG+= FFCPMG[omegaIndex]*A*np.abs(omega)*domega
            omegaIndex+=1

        # final coherence values under the Gaussian distributed phase approximation
        CoherenceArrayUDD[omegacutIndex] = np.exp(-phi2UDD)
        CoherenceArrayCPMG[omegacutIndex] = np.exp(-phi2CPMG)
        omegacutIndex+=1
        print('{} percent finished with the omegacut loop'.format(np.round(100*omegacutIndex/N), decimals=2))

    plot1 = plt.figure(1)
    plt.plot(omegacutArray, CoherenceArrayUDD, label = 'UDD-{}'.format(n))
    plt.plot(omegacutArray, CoherenceArrayCPMG, label = 'CPMG-{}'.format(n))
    plt.legend()
    plt.xlabel('Cutoff Frequency [2pi x Hertz]')
    plt.ylabel('Coherence @ t_final (Gaussian phase approx)')

    plot2 = plt.figure(2)
    plt.plot(frequency_axis, FFUDD, label = 'UDD-{} Frequency Domain Filter Function'.format(n))
    plt.plot(frequency_axis, FFCPMG, label = 'CPMG-{} Frequency Domain Filter Function'.format(n))
    plt.legend()
    plt.xlabel('Angular Frequency [2pi x Hertz]')
    plt.ylabel('Filter Function |F|^2')

    plot3 = plt.figure(3)
    plt.plot(timeArray, fUDD, label = 'UDD-{} Time Domain Filter Function'.format(n))
    plt.plot(timeArray, fCPMG, label = 'CPMG-{} Time Domain Filter Function'.format(n))
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('f(t)')

    plot4 = plt.figure(4)
    plt.stem(PulseTimeUDD, pulseUDD, label = 'UDD-{} Sequence'.format(n),linefmt='blue', markerfmt='o')
    plt.stem(PulseTimeCPMG, pulseCPMG, label = 'CPMG-{} Sequence'.format(n),linefmt='grey', markerfmt='o')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Pulses')

    plt.show()

    return

def LorentzianNoiseComparisonVaryCenter():

    # calculate coherence at the end of a pulse sequence (CPMG or UDD) = C(tf) vs center frequency of the lorentzian noise spectrum
    # using the gaussian phase approximation to calculate C(tf) = exp(-<phi^2>/2)
    N=10**2 #number of widths to calculate
    omega0Max = 150
    omega0Array = np.linspace(0, omega0Max, N)
    gamma = 5 #width of the lorentzian in angular frequency

    n = 30 # number of pulses in the sequence

    A = 200 #arbitrary amplitude of noise spectrum S(omega) = A*L(omega) where L is a normalized Lorentzian
           # chosen to give reasonable coherence values somewhere around 0.5 for the noise parameters used here

    # time/frequency parameters
    tf = 1
    deltat = 10**(-4)
    omegaMax = 3.14/deltat
    ZP = 20 # Multiply time points by ZP and zero-pad
    Nt = ZP*np.int(tf/deltat) # zero-padding the time domain sequence to decrease domega in the frequency domain
    Nomega = Nt
    domega = omegaMax/Nomega
    timeArray = np.linspace(0, ZP*tf, Nt)
    omegaArray = np.linspace(0, omegaMax, Nomega)
    print(omegaMax)


    # Arrays to hold the coherence under a CPMG or UDD sequence
    CoherenceArrayUDD = np.zeros(N)
    CoherenceArrayCPMG = np.zeros(N)

    # arrays to be used to visualize sequences...
    pulseUDD = np.ones(n)
    pulseCPMG = np.ones(n)

    PulseTimeUDD = np.zeros(n)
    PulseTimeCPMG = np.zeros(n)

    # ... and time domain filter functions.
    fUDD = np.zeros(Nt)
    fCPMG = np.zeros(Nt)

    # j = pulse number
    for j in range(0, n):

        # when the jth time interval ends for each sequence
        PulseTimeUDD[j] = tf*(np.sin(3.14*(j+1)/(2*(n+1)))**2)
        PulseTimeCPMG[j] = j*tf/n + tf/(2*n)

        # time intervals between the jth and (j+1)th pulse
        timeUDDmax = tf*(np.sin(3.14*(j+1)/(2*(n+1)))**2)
        timeUDDmin = tf*(np.sin(3.14*j/(2*(n+1)))**2)
        timeCPMGmin = (j-1)*tf/n + tf/(2*n)
        timeCPMGmax = j*tf/n + tf/(2*n)

        #chose the sign of the time domain filter function accordingly
        for t in range(0, Nt-1):
            if t*tf*ZP/Nt>timeUDDmin and t*tf*ZP/Nt<timeUDDmax:
                fUDD[t]=(-1)**j
            if t*tf*ZP/Nt>timeCPMGmin and t*tf*ZP/Nt<timeCPMGmax:
                fCPMG[t]=(-1)**j

    # Frequency domain Filter functions, normalized by Nt/2 by the defn of FFT given by scipy
    FFUDD = 2*fft(fUDD)/Nt
    FFCPMG = 2*fft(fCPMG)/Nt
    FFUDD = np.abs(FFUDD)**2
    FFCPMG = np.abs(FFCPMG)**2
    frequency_axis = 2*3.14*fftfreq(Nt, d=deltat) #angular frequency


    # For each width, calculate the coherence
    omega0Index = 0

    for omega0 in omega0Array:

        # Define the overlap integral <phi^2>
        phi2UDD = 0
        phi2CPMG = 0

        # integrate over omega
        omegaIndex = 0
        for omega in omegaArray:
            # update the integral <phi^2>;
            # since only summing over positive frequencies here, and the integrand is symmetric, add in a factor of 2 in the final expression for coherence
            # but that factor of 2 is cancelled by the other 2 in C(t)=exp(-<phi^2>/2)
            phi2UDD+= FFUDD[omegaIndex]*((A/3.14)*(gamma/2)/(((omega-omega0)**2) + (gamma/2)**2))*domega
            phi2CPMG+= FFCPMG[omegaIndex]*((A/3.14)*(gamma/2)/(((omega-omega0)**2) + (gamma/2)**2))*domega
            omegaIndex+=1

        # final coherence values under the Gaussian distributed phase approximation
        coherenceUDD = np.exp(-phi2UDD)
        coherenceCPMG = np.exp(-phi2CPMG)
        CoherenceArrayUDD[omega0Index] = coherenceUDD
        CoherenceArrayCPMG[omega0Index] = coherenceCPMG
        omega0Index+=1
        print('{} percent finished with the omega0 loop'.format(np.round(100*omega0Index/N), decimals=2))

    plot1 = plt.figure(1)
    plt.plot(omega0Array, CoherenceArrayUDD, label = 'UDD-{}'.format(n))
    plt.plot(omega0Array, CoherenceArrayCPMG, label = 'CPMG-{}'.format(n))
    plt.legend()
    plt.xlabel('Lorentzian Center Frequency, \Gamma={} [2pi x Hertz]'.format(gamma))
    plt.ylabel('Coherence @ t_final (Gaussian phase approx)')

    plot2 = plt.figure(2)
    plt.plot(frequency_axis, FFUDD, label = 'UDD-{} Frequency Domain Filter Function'.format(n))
    plt.plot(frequency_axis, FFCPMG, label = 'CPMG-{} Frequency Domain Filter Function'.format(n))
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.xlabel('Angular Frequency [2pi x Hertz]')
    plt.ylabel('Filter Function |F|^2')

    plot3 = plt.figure(3)
    plt.plot(timeArray, fUDD, label = 'UDD-{} Time Domain Filter Function'.format(n))
    plt.plot(timeArray, fCPMG, label = 'CPMG-{} Time Domain Filter Function'.format(n))
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('f(t)')

    plot4 = plt.figure(4)
    plt.stem(PulseTimeUDD, pulseUDD, label = 'UDD-{} Sequence'.format(n),linefmt='blue', markerfmt='o')
    plt.stem(PulseTimeCPMG, pulseCPMG, label = 'CPMG-{} Sequence'.format(n),linefmt='grey', markerfmt='o')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Pulses')

    plt.show()

    return

def LorentzianNoiseComparisonVaryNumPulsesUDD():

    # calculate coherence at the end of a UDD pulse sequence = C(tf) for different numbers of pulses with a
    # lorentzian noise spectrum using the gaussian phase approximation to calculate C(tf) = exp(-<phi^2>/2)
    #number of sequences with different numbers of pulses to calculate
    Nmin=5
    Nmax = 10
    dN = 1
    N = np.int((Nmax-Nmin)/dN)
    nPulsesArray = np.arange(Nmin, Nmax, dN)
    gamma = 10 #width of the lorentzian in angular frequency
    omega0 = 0 # center frequency of Lorentzian

    A = 1000 #arbitrary amplitude of noise spectrum S(omega) = A*L(omega) where L is a normalized Lorentzian
           # chosen to give reasonable coherence values somewhere around 0.5 for the noise parameters used here

    # time/frequency parameters
    tf = 1
    deltat = 10**(-4)
    omegaMax = 3.14/deltat
    ZP = 20 # Multiply time points by ZP and zero-pad
    Nt = np.int(tf/deltat) # zero-padding the time domain sequence to decrease domega in the frequency domain
    ZP = 6 # zero-padding the time domain sequence to decrease domega in the frequency domain
    timeArray = np.linspace(-ZP*tf, ZP*tf, 2*ZP*Nt+1) #zero-padding the time array for better frequency resolution

    # Arrays to hold the coherence for an n-pulse UDD sequence
    CoherenceArrayUDD = np.zeros(N)


    # time domain filter function
    fUDD = np.zeros((N, 2*ZP*Nt+1))
    # Frequency domain filter function
    FFUDD = np.zeros((N, 2*ZP*Nt+1), dtype = complex)

    sequenceIndex = 0
    for n in nPulsesArray:

        nn=np.int(n)
        # j = pulse number; calculating time domain filter functions
        for j in range(0, nn):

            # time intervals between the jth and (j+1)th pulse
            timeUDDmax = tf*(np.sin(3.14*(j+1)/(2*(n+1)))**2)
            timeUDDmin = tf*(np.sin(3.14*j/(2*(n+1)))**2)

            #chose the sign of the time domain filter function accordingly
            for t in range(0, Nt-1):
                if t*tf*ZP/Nt>timeUDDmin and t*tf*ZP/Nt<timeUDDmax:
                    fUDD[sequenceIndex, t]=(-1)**j


        # Frequency domain Filter functions, normalized by Nt/2 by the defn of FFT given by scipy
        FFUDD[sequenceIndex, :] = 2*fft(fUDD[sequenceIndex, :])/(2*Nt*ZP + 1)
        FFUDD = np.abs(FFUDD)**2
        frequency_axis = 2*3.14*fftfreq(2*ZP*Nt+1, d=deltat) #angular frequency
        domega = frequency_axis[2]-frequency_axis[1]

        # perform overlap integrals for each sequence
        phi2UDD=0
        omegaIndex=0
        for omega in frequency_axis:

            # update the integral <phi^2>;
            # since only summing over positive frequencies here, and the integrand is symmetric, add in a factor of 2 in the final expression for coherence
            # but that factor of 2 is cancelled by the other 2 in C(t)=exp(-<phi^2>/2)
            phi2UDD+= FFUDD[sequenceIndex, omegaIndex]*((A/3.14)*(gamma/2)/(((omega-omega0)**2) + (gamma/2)**2))*domega
            omegaIndex+=1

        # final coherence values under the Gaussian distributed phase approximation
        coherenceUDD = np.exp(-phi2UDD)
        CoherenceArrayUDD[sequenceIndex] = coherenceUDD

        sequenceIndex += 1
        print('{} percent finished with the loop'.format(np.round(100*sequenceIndex/N), decimals=2))



    frequency_axis = 2*3.14*fftfreq(2*ZP*Nt+1, d=deltat) #angular frequency

    plot1 = plt.figure(1)
    plt.plot(nPulsesArray, CoherenceArrayUDD)
    plt.legend()
    plt.xlabel('Number of pulses in UDD sequence')
    plt.ylabel('Coherence @ t_final (Gaussian phase approx)')

    plot2 = plt.figure(2)
    plotFFIndex = 0
    for n in nPulsesArray:
        plt.plot(frequency_axis, FFUDD[plotFFIndex, :], label = 'UDD-{} Frequency Domain Filter Function'.format(n))
        plotFFIndex+=1
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.xlabel('Angular Frequency [2pi x Hertz]')
    plt.ylabel('Filter Function |F|^2')

    plot3 = plt.figure(3)
    plotfIndex = 0
    for n in nPulsesArray:
        plt.plot(timeArray, fUDD[plotfIndex, :], label = 'UDD-{} Time Domain Filter Function'.format(n))
        plotfIndex+=1
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('f(t)')

    plt.show()

    return


LorentzianNoiseComparisonVaryWidth()
# LorentzianNoiseComparisonVarySharpCutoff()
# LorentzianNoiseComparisonVaryCenter()
# LorentzianNoiseComparisonVaryNumPulsesUDD()
