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








#Calculating the probability distribution of the magnetic field at a given
# lattice site that comes from a random distribution of spins throughout the lattice
# Doing for 1, 2, and 3 dim, but starting with 1d



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#The following 2 sets of code will plot <B^2> versus occupation probability
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def RandomB_1D_vs_OccProb():
    #How many occupation probabilities to look at
    OCP = 40
    occ_prob = np.zeros(OCP)
    for mm in range(OCP):
        occ_prob[mm] = mm/(1*OCP)
    #Array to hold <B^2> for that particular occ_prob
    AvgBSqArray = np.zeros(OCP)

    #Dimension of the 2x2 array square
    N = 200
    #How many times to average over a randomly occupied lattice
    M = 1000
    for prob in occ_prob:
        B_fields = np.zeros(M)
        B_fields_sq = np.zeros(M)
        for kk in range(0, M):
            lattice = np.zeros(N)
            updown = [-1, 1]
            for ii in range(0, N-1):
                occ = random.uniform(0, 1)
                if occ<prob:
                    spin = random.choice(updown)
                    lattice[ii] = spin

            lattice[np.int(N/2)] = 0
            B = 0
            for jj in range(0, np.int(N/2 - 1)):
                B = lattice[jj]/((np.abs(np.int(N/2)-jj))**3) + B
            for jj in range(np.int(N/2 + 1), N-1):
                B = lattice[jj]/((np.abs(np.int(N/2)-jj))**3) + B
            B_fields[kk] = B
            B_fields_sq[kk] = B**2
        AvgB = np.round_(np.average(B_fields), decimals = 2)
        AvgBSq = np.round_(np.average(B_fields_sq), decimals = 2)
        StdB = np.round_(np.std(B_fields), decimals = 2)
        index = np.int(prob*OCP*1)
        AvgBSqArray[index] = AvgBSq
        print(prob)

    plt.plot(AvgBSqArray)
    plt.show()

    return

def RandomB_2D_vs_OccProb():

    #How many occupation probabilities to look at
    OCP = 50
    occ_prob = np.zeros(OCP)
    for mm in range(OCP):
        occ_prob[mm] = mm/(1*OCP)
    #Array to hold <B^2> for that particular occ_prob
    AvgBSqArray = np.zeros(OCP)

    #Dimension of the 2x2 array square
    N = 30
    #How many times to average over a randomly occupied lattice
    M = 1000
    for prob in occ_prob:
        # Calculating <B^2> for a given occupation probability
        #######################################################################
        B_fields = np.zeros(M)
        B_fields_sq = np.zeros(M)
        for kk in range(0, M):
            lattice = np.zeros((N, N))
            updown = [-1, 1]
            for ii in range(0, N-1):
                for jj in range(0, N-1):
                    occ = random.uniform(0, 1)
                    if occ<prob:
                        spin = random.choice(updown)
                        lattice[ii, jj] = spin

            lattice[np.int(N/2), np.int(N/2)] = 0
            B = 0
            for ii in range(0, N):
                for jj in range(0, N-1):
                    if jj != np.int(N/2) and ii != np.int(N/2):
                        B = lattice[ii, jj]/(((np.abs(np.int(N/2)-jj))**2 + (np.abs(np.int(N/2)-jj))**2)**(3/2)) + B
            B_fields[kk] = B
            B_fields_sq[kk] = B**2

        AvgB = np.round_(np.average(B_fields), decimals = 3)
        AvgBSq = np.round_(np.average(B_fields_sq), decimals = 3)
        StdB = np.round_(np.std(B_fields), decimals = 3)
        #######################################################################
        index = np.int(prob*OCP*1)
        AvgBSqArray[index] = AvgBSq
        print(prob)

#    plt.hist(B_fields, bins=50)
#    plt.title('<B> = {}, <B^2> = {}, Std Dev = {}'.format(AvgB, AvgBSq, StdB))
#    plt.xlabel('Strength of B (a.u.)')
#    plt.ylabel('Number of occurences')

    plt.plot(occ_prob,AvgBSqArray)
    plt.title('<B^2> vs Occupation Probability')
    plt.xlabel('Occupation Probability')
    plt.ylabel('<B^2>')

    plt.show()

    return



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#These will show a histogram of <B> and <B^2> for a give occ_prob
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def RandomB_1D():

    #Number of 1d lattice sites
    N = 200
    #Number of times to average
    M = 1000
    #Occupation probability
    occ_prob = 0.2

    B_fields = np.zeros(M)
    B_fields_sq = np.zeros(M)
    for kk in range(0, M-1):
        lattice = np.zeros(N)
        updown = [-1, 1]
        for ii in range(0, N-1):
            occ = random.uniform(0, 1)
            if occ<occ_prob:
                spin = random.choice(updown)
                lattice[ii] = spin

        lattice[np.int(N/2)] = 0
        B = 0
        for jj in range(0, N-1):
            if jj != np.int(N/2):
                B = lattice[jj]/((np.abs(np.int(N/2)-jj))**3) + B
        B_fields[kk] = B
        B_fields_sq[kk] = B**2

    plt.hist(B_fields_sq, bins=50)
    plt.xlabel('<B^2> (a.u.)')
    plt.ylabel('Number of occurences')
    plt.show()

    return

def RandomB_2D():

    #Dimension of the 2x2 array square
    N = 35
    #How many times to average over a randomly occupied lattice
    M = 1000
    #Occupation probability
    occ_prob = 0.01
    # Calculating <B^2> for a given occupation probability
    #######################################################################
    B_fields = np.zeros(M)
    B_fields_sq = np.zeros(M)
    for kk in range(0, M):
        lattice = np.zeros((N, N))
        updown = [-1, 1]
        for ii in range(0, N-1):
            for jj in range(0, N-1):
                occ = random.uniform(0, 1)
                if occ<occ_prob:
                    spin = random.choice(updown)
                    lattice[ii, jj] = spin
        if kk == 200 or kk==300 or kk == 400:
            imshow(lattice)
            plt.show()
        lattice[np.int(N/2), np.int(N/2)] = 0
        B = 0
        for ii in range(0, N):
            for jj in range(0, N-1):
                if jj != np.int(N/2) and ii != np.int(N/2):
                    B = lattice[ii, jj]/(((np.abs(np.int(N/2)-jj))**2 + (np.abs(np.int(N/2)-jj))**2)**(3/2)) + B
        B_fields[kk] = B
        B_fields_sq[kk] = B**2
        print(kk)

    AvgB = np.round_(np.average(B_fields), decimals = 2)
    AvgBSq = np.round_(np.average(B_fields_sq), decimals = 2)
    StdB = np.round_(np.std(B_fields), decimals = 2)

    plt.hist(B_fields_sq, bins=80)
    plt.hist(B_fields, bins=80)
    plt.title('<B> = {}, <B^2> = {}, Std Dev = {}'.format(AvgB, AvgBSq, StdB))
    plt.xlabel('<B> (a.u.)')
    plt.ylabel('Number of occurences')
    plt.show()


    plt.show()

    return




#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Trying to run some simulations that include spin dynamics.
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------




def RandomB_1D_vs_t(occ_prob, F0):

    #Number of 1d lattice sites
    N = 200
    #Number of time steps in the dynamics for a given, initial, random spin configuration
    M = 1000
    # Define a time array (just for plotting at the end)
    time = np.zeros(M)
    for ii in range(0,M-1):
        time[ii] = ii/M
    time[M-1]=time[M-2]
    B_fields = np.zeros(M)
    B_fields_sq = np.zeros(M)
    # Populate the lattice
    lattice = np.zeros(N)
    updown = [-1, 1]
    for ii in range(0, N-1):
        occ = random.uniform(0, 1)
        if occ<occ_prob:
            spin = random.choice(updown)
            lattice[ii] = spin
    # Define the central spin as the one at N/2; label it as 2 to be distinct from +/-1 or 0
    lattice[np.int(N/2)] = 2
    # kk defines the time step in the dynamics
    for kk in range(0, M-1):
        # Calculate B and B**2 for this given configuration at the kk_th time step
        B = 0
        for jj in range(0, N-1):
            if jj != np.int(N/2):
                B = lattice[jj]/((np.abs(np.int(N/2)-jj))**3) + B
        B_fields[kk] = B
        B_fields_sq[kk] = B**2
        # Induce random flip flops... Define flip-flop prob per unit time step as F0/r**3
        # where r is the distance between the spins
        print('{}% finished simulating dynamics'.format(100*kk/M))
        for jj in range(0,N-2):
            for ii in range(jj+1,N-1):
                # If the spins have opposite m_s, induce flip-flops with some probability
                if lattice[ii]==1 and lattice[jj]==-1:
                    ff = random.uniform(0, 1)
                    if ff<np.abs(F0/(ii-jj)**3):
                        lattice[ii]=-1
                        lattice[jj]=1
                if lattice[ii]==-1 and lattice[jj]==1:
                    ff = random.uniform(0, 1)
                    if ff<np.abs(F0/(ii-jj)**3):
                        lattice[ii]=1
                        lattice[jj]=-1
        B_fields[M-1] = B_fields[M-2]

    freq, Pxx_den = signal.periodogram(B_fields)

    # Fit the spectral density function above to various trial functions
    def lorentzian(x, a, g, x0):
        return a*((g/2)/((g/2)**2+(x-x0)**2))
    def inverse(t, b, c, t0):
        return b*t*((c/2)/((c/2)**2+(t-t0)**2))
    max_Pxx = np.max(Pxx_den)
    pars_Lorentzian, cov_Lorentzian = curve_fit(lorentzian, freq, Pxx_den, p0=[max_Pxx,0.1,0.01], bounds=(0, np.inf))
    pars_Inverse, cov_Inverse = curve_fit(inverse, freq, Pxx_den, p0=[max_Pxx,0.1,0.01], bounds=(0, np.inf))

    # Integrate the PSD against the filter function for a spin echo
    def Echo_FF(w, t):
        return (np.sin(w*t)**4)/w**2

    # tau steps
    NN = 10**3
    # Final Tau
    tau_f = 30
    Tau_ = np.zeros(NN)
    for ii in range(0, NN-1):
        Tau_[ii] = tau_f*ii/NN
    Tau_[NN-1] = Tau_[NN-2]
    # freq steps
    MM = np.size(freq)
    dw = (freq[2] - freq[1])
    # Define the integral as chi(tau)
    chi = np.zeros(NN)
    # perform the integral as, for a given tau...
    for tau in range(0,NN-1):
        print('{}% finished with the Chi integration'.format(100*tau/NN))
        #... perform the integral over omega
        for w in range(1,MM-1):
            chi[tau] = chi[tau] + dw*Echo_FF(w*dw, tau_f*tau/NN)*Pxx_den[w]
    chi[NN-1] = chi[NN-2]
    Echo_Decay = np.zeros(NN)
    for ii in range(0, NN-1):
        Echo_Decay[ii] = np.exp(-chi[ii])
    Echo_Decay[NN-1] = Echo_Decay[NN-2]
    # Fit chi to a power law, equivalent to fitting the decay to a stretched exponential
    def Chi_Power_Law(tau, T2, n):
        return (tau/T2)**n
    pars_Chi, cov_Chi = curve_fit(Chi_Power_Law, Tau_, chi, p0=[1,3], bounds=(0, np.inf))
    n_fit = str(np.round_(pars_Chi[1],decimals = 2))
    T2_fit= str(np.round_(pars_Chi[0],decimals = 5))


    plot1 = plt.figure(1)
    plt.plot(time, B_fields)
    plt.xlabel('Time (a.u.)')
    plt.ylabel('B(t) (a.u.)')

    plot2 = plt.figure(2)
    plt.semilogy(freq, lorentzian(freq, *pars_Lorentzian), linestyle='--', linewidth=2, color='blue',label='Lorentzian Fit')
    plt.semilogy(freq, inverse(freq, *pars_Inverse), linestyle='--',linewidth=2, color='red',label='1/f Fit')
    plt.semilogy(freq, Pxx_den,linestyle='-', linewidth=2, color='black',label='Spectral Density')
    plt.xlabel('frequency')
    plt.ylabel('Power Spectral Density')
    plt.legend()


    plot3 = plt.figure(3)
    plt.plot(Tau_, Echo_Decay,linestyle='-' , linewidth=2, color='black',label='Echo')
    plt.plot(Tau_, np.exp(-Chi_Power_Law(Tau_, *pars_Chi)),linestyle='--' , linewidth=2, color='red',label='Echo Fit' )
    plt.xlabel('Tau')
    plt.ylabel('Echo Signal')

    plot4 = plt.figure(4)
    plt.plot(Tau_, chi,linestyle='-' , linewidth=2, color='black',label='Chi')
    plt.plot(Tau_, Chi_Power_Law(Tau_, *pars_Chi), linestyle='--',linewidth=2, color='red',label='Chi Fit to Power Law')
    plt.xlabel('Tau')
    plt.ylabel('Chi(Tau)')
    plt.title('Fit to ~(tau/T2)**n, Fitted with T2 = {} and n = {}'.format(T2_fit, n_fit))

    plot5 = plt.figure(5)
    lattice = np.expand_dims(lattice, axis=10)  # or axis=1
    cmap = colors.ListedColormap(['blue', 'white', 'orange', 'red'])
    bounds=[-1.5,-0.5,0.5,1.5,2.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(lattice, interpolation='nearest', origin='lower', cmap=cmap, norm=norm)

    plt.show()

    return



def spin_decay_2d(occ_prob):

    #Dimension of the 2x2 array square
    N = 35
    #Number of time steps in the dynamics for a given, initial, random spin configuration
    M = 1000
    # Define a time array for plotting B(t)
    time = np.zeros(M)
    for ii in range(0,M-1):
        time[ii] = ii/M
    time[M-1]=time[M-2]
    # Array to hold B and B**2 at each time step
    B_fields = np.zeros(M)
    B_fields_sq = np.zeros(M)
    lattice = np.zeros((N, N))
    # Define spin up (1) and spin down (-1) and populate the lattice randomly with spins accordingg to the occupation probability
    updown = [-1, 1]
    for ii in range(0, N-1):
        for jj in range(0, N-1):
            occ = random.uniform(0, 1)
            if occ<occ_prob:
                lattice[ii, jj] = random.choice(updown)
    # Define the ``central spin''
    lattice[np.int(N/2), np.int(N/2)] = 2

    def RandomB_2D_vs_t(F0, tau_f):

        # ------------------------- Simulate the dynamics of the spin bath by inducing random flip-flops between spins --------------------------------------- #
        for kk in range(0, M):
            #print('{}% finished simulating the dynamics'.format(100*kk/M))
            B = 0
            #Calculate B for the current configuration
            for ii in range(0, N):
                for jj in range(0, N-1):
                    if jj != np.int(N/2) and ii != np.int(N/2):
                        B = lattice[ii, jj]/(((np.abs(np.int(N/2)-jj))**2 + (np.abs(np.int(N/2)-jj))**2)**(3/2)) + B
            B_fields[kk] = B
            B_fields_sq[kk] = B**2
            # Induce random flip flops in the bath spins
            # brute force for now.... try making this more efficient later
            # ii and jj are looking for spin up in the first round, and for spin down in the second
            for ii in range(0, N-1):
                for jj in range(0, N-1):
                    if lattice[ii,jj]==1:
                        # nn and mm are looking for a spin down to flip flop with
                        for nn in range(0, N-1):
                            for mm in range(0, N-1):
                                if lattice[nn,mm]==-1:
                                    ff = random.uniform(0, 1)
                                    if ((nn-ii)**2+(mm-jj)**2)**(3/2) !=0:
                                        if ff<np.abs(F0/((nn-ii)**2+(mm-jj)**2)**(3/2)):
                                            lattice[ii,jj]=-1
                                            lattice[nn,mm]=1
                        # Break the loop if the spin has flipped i.e. only allow one flip-flop per timestep
                        if lattice[ii,jj]==-1:
                            break
                    elif lattice[ii,jj]==-1:
                        # nn and mm are looking for a spin down to flip flop with
                        for nn in range(0, N-1):
                            for mm in range(0, N-1):
                                if lattice[nn,mm]==1:
                                    ff = random.uniform(0, 1)
                                    if ((nn-ii)**2+(mm-jj)**2)**(3/2) !=0:
                                        if ff<np.abs(F0/((nn-ii)**2+(mm-jj)**2)**(3/2)):
                                            lattice[ii,jj]=1
                                            lattice[nn,mm]=-1
                        if lattice[ii,jj]==1:
                            break
                    else:
                        lattice[ii,jj]==0

        #--------------------------------- Simulate random flip flops by a T1-type process with random matrices --------------------------------#
        # for kk in range(0, M):
        #     #print('{}% finished simulating the dynamics'.format(100*kk/M))
        #     B = 0
        #     #Calculate B for the current configuration
        #     for ii in range(0, N):
        #         for jj in range(0, N-1):
        #             if jj != np.int(N/2) and ii != np.int(N/2):
        #                 B = lattice[ii, jj]/(((np.abs(np.int(N/2)-jj))**2 + (np.abs(np.int(N/2)-jj))**2)**(3/2)) + B
        #     B_fields[kk] = B
        #     B_fields_sq[kk] = B**2
        #     # define a diagonal matrix f_mat where the enrtries are 1 (no spin flip) or -1 (spin flip), where the probability of flipping is pf
        #     pf = 0.01
        #     flip_mat = np.zeros((N,N))
        #     flop_mat = np.zeros((N,N))
        #     for ii in range(0, N-1):
        #         flip = random.uniform(0, 1)
        #         flop = random.uniform(0, 1)
        #         if flip<pf:
        #             flip_mat[ii, ii] = -1
        #         else:
        #             flip_mat[ii,ii] = 1
        #         if flop<pf:
        #             flop_mat[ii, ii] = -1
        #         else:
        #             flop_mat[ii,ii] = 1
        #     lattice = np.matmul(lattice, flip_mat)
        #     lattice = np.matmul(flop_mat, lattice)

        B_fields[M-1] = B_fields[M-2]

        B_corr = np.zeros(M)
        for tau in range(0,M-1):
            for t in range(0, M-tau-1):
                B_corr[tau] = ((B_fields[t]*B_fields[t+tau])/(M-tau))+B_corr[tau]



        # integrate B(t) to get phi(t) for various pulse sequences; here Ramsey and Echo

        phi_Ramsey = np.zeros(M)
        phi_Echo = np.zeros(M)
        for t in range(0, M-1):
            phi_Ramsey[t] = phi_Ramsey[t-1]+(t/M)*B_fields[t]
            # need to define dummy variable of integration tt to integrate phi for echo
            for tt in range(0, t-1):
                if tt<(t/2):
                    phi_Echo[t] = phi_Echo[t]+(tt/M)*B_fields[tt]
                else:
                    phi_Echo[t] = phi_Echo[t]-(tt/M)*B_fields[tt]
                #print('{}% done integrating B(t)'.format(100*t/M))

        # Calculate the correlation function of B(t)


            #-----------------The following analysis/fits should be valid for gaussian noise, not necessarily for discrete/telegraph noise -----------------------#
            #-----------------In any case it is always an approximation when we have a finite sample size --------------------------------------------------------#


        # Obtain the spectral density of B(t)
        # freq, Pxx_den = signal.periodogram(B_fields)

        # Fit the spectral density function above to various trial functions
        # def lorentzian(x, a, g, x0):
        #     return a*((g/2)/((g/2)**2+(x-x0)**2))
        # def inverse(t, b, c, t0):
        #     return b*t*((c/2)/((c/2)**2+(t-t0)**2))
        # max_Pxx = np.max(Pxx_den)
        # pars_Lorentzian, cov_Lorentzian = curve_fit(lorentzian, freq, Pxx_den, p0=[max_Pxx,0.4,0.1], bounds=(0, np.inf))
        # pars_Inverse, cov_Inverse = curve_fit(inverse, freq, Pxx_den, p0=[max_Pxx,0.4,0.01], bounds=(0, np.inf))

        # Integrate the PSD against the filter function for a spin echo and Ramsey
        # def Echo_FF(w, t):
        #     return 8*(((np.sin(w*t/4))**4)/(w**2))
        # def Ramsey_FF(f, y):
        #     return 2*(((np.sin(f*y/2))**2)/(f**2))
        #
        # # tau steps
        # NN = 10**3
        # Tau_ = np.zeros(NN)
        # for ii in range(0, NN-1):
        #     Tau_[ii] = tau_f*ii/NN
        # Tau_[NN-1] = Tau_[NN-2]
        # # freq steps
        # MM = np.size(freq)
        # dw = (freq[2] - freq[1])
        # # Define the integral as chi(tau)
        # chi_Echo = np.zeros(NN)
        # chi_Ramsey = np.zeros(NN)
        # # perform the integral as, for a given tau...
        # for tau in range(0,NN-1):
        #     print('{}% finished with the Chi integration'.format(100*tau/NN))
        #     #... perform the integral over omega
        #     for w in range(1,MM-1):
        #         chi_Echo[tau] = chi_Echo[tau] + dw*Echo_FF(w*dw, tau_f*tau/NN)*Pxx_den[w]
        #         chi_Ramsey[tau] = chi_Ramsey[tau] + dw*Ramsey_FF(w*dw, tau_f*tau/NN)*Pxx_den[w]
        #
        # chi_Echo[NN-1] = chi_Echo[NN-2]
        # chi_Ramsey[NN-1] = chi_Ramsey[NN-2]
        # Echo_Decay = np.exp(-chi_Echo)
        # Ramsey_Decay = np.exp(-chi_Ramsey)
        # # Fit chi to a power law, equivalent to fitting the decay to a stretched exponential
        # def Chi_Echo_Power_Law(tau, T2, n_Echo):
        #     return (tau/T2)**n_Echo
        # def Chi_Ramsey_Power_Law(tt, T2s, n_Ramsey):
        #     return (tt/T2s)**n_Ramsey
        # pars_Chi_Echo, cov_Chi_Echo = curve_fit(Chi_Echo_Power_Law, Tau_, chi_Echo, p0=[20,4], bounds=(0, np.inf))
        # n_Echo_fit = str(np.round_(pars_Chi_Echo[1],decimals = 2))
        # T2_Echo_fit= str(np.round_(pars_Chi_Echo[0],decimals = 2))
        # pars_Chi_Ramsey, cov_Chi_Ramsey = curve_fit(Chi_Ramsey_Power_Law, Tau_, chi_Ramsey, p0=[3,3], bounds=(0, np.inf))
        # n_Ramsey_fit = str(np.round_(pars_Chi_Ramsey[1],decimals = 2))
        # T2_Ramsey_fit= str(np.round_(pars_Chi_Ramsey[0],decimals = 2))

        return np.cos(phi_Ramsey), np.cos(phi_Echo), phi_Ramsey, phi_Echo, B_fields, B_corr

    M = 1000
    time = np.zeros(M)
    for ii in range(0,M-1):
        time[ii] = ii/M
    time[M-1]=time[M-2]

    nn = 10**5 # = number of times to average
    mm = 2 # number of occupation probabilities to look at
    qq = 50 # mm/qq is the maximum occupation probability
    # For a given occupation probability...
    def stretched_Exp1(t, a, T2, nE):
        return a*np.exp(-(t/T2)**nE)
    def stretched_Exp2(s, b, T2s, nR):
        return b*np.exp(-(s/T2s)**nR)
    def correlation_Fit(tt, c, tau):
        return c*np.exp(-(tt/tau))
    def lorentzian(p, h, m, g):
        return (h/3.14159)*(g/2)/((g/2)**2+(p-m)**2)
    for OCP in range(1, mm):
        cos_phi_ramsey_avg = np.zeros(M)
        cos_phi_echo_avg = np.zeros(M)
        phi_ramsey_avg_EarlyTime = []
        phi_echo_avg_EarlyTime =[]
        phi_ramsey_avg_EarlyTime2 = []
        phi_echo_avg_EarlyTime2 =[]
        phi_ramsey_avg_LateTime = []
        phi_echo_avg_LateTime =[]
        B_fields_EarlyTime = []
        B_fields_EarlyTime2 = []
        B_fields_LateTime = []
        B__fields = np.zeros(M)
        B_Correlation = np.zeros(M)
        OCP1= str(np.round_(OCP/qq,decimals = 4))
        #Define "Early" times (t1, t2) and "Late" times (t3)
        t1 = 10
        t2 = 30
        t3 = 600
        #...calculate cos(phi) for n iterations, getting <cos(phi)> for the decay
        for ii in range(0, nn):
            J0 = 0.1 # (1/J0*10**3) = flip flop rate of nearest neighbors in these units
            cos_phi_ramsey, cos_phi_echo, phi_ramsey, phi_echo, B_fields, B_corr = RandomB_2D_vs_t(J0, 15)
            B_Correlation = B_Correlation + B_corr
            cos_phi_ramsey_avg = cos_phi_ramsey + cos_phi_ramsey_avg
            cos_phi_echo_avg = cos_phi_echo + cos_phi_echo_avg
            phi_ramsey_avg_EarlyTime = np.append(phi_ramsey_avg_EarlyTime, phi_ramsey[t1])
            phi_echo_avg_EarlyTime = np.append(phi_echo_avg_EarlyTime, phi_echo[t1])
            phi_ramsey_avg_EarlyTime2 = np.append(phi_ramsey_avg_EarlyTime2, phi_ramsey[t2])
            phi_echo_avg_EarlyTime2 = np.append(phi_echo_avg_EarlyTime2, phi_echo[t2])
            phi_ramsey_avg_LateTime = np.append(phi_ramsey_avg_LateTime, phi_ramsey[t3])
            phi_echo_avg_LateTime = np.append(phi_echo_avg_LateTime, phi_echo[t3])
            B_fields_EarlyTime = np.append(B_fields_EarlyTime, B_fields[t1])
            B_fields_EarlyTime2 = np.append(B_fields_EarlyTime2, B_fields[t2])
            B_fields_LateTime = np.append(B_fields_LateTime, B_fields[t3])
            B__fields = B_fields
            # Plot the results after every 200 iterations
            if np.mod(ii, 200)==1:

                max_ramsey = np.max(cos_phi_ramsey_avg)
                max_echo = np.max(cos_phi_echo_avg)
                max_corr = np.max(B_corr)
                cos_phi_ramsey_avg[M-1] = cos_phi_ramsey_avg[M-2]
                cos_phi_echo_avg[M-1] = cos_phi_echo_avg[M-2]

                pars_ramsey, cov_ramsey = curve_fit(stretched_Exp2, time, cos_phi_ramsey_avg, p0=[max_ramsey,0.4,0.66], bounds=(0, np.inf))
                pars_echo, cov_echo = curve_fit(stretched_Exp1, time, cos_phi_echo_avg, p0=[max_echo,0.4,1], bounds=(0, np.inf))
                pars_corr, cov_corr = curve_fit(correlation_Fit, time, B_Correlation, p0=[max_corr,0.5], bounds=(0, np.inf))

                n_Echo_fit = str(np.round_(pars_echo[2],decimals = 2))
                T2_Echo_fit= str(np.round_(pars_echo[1],decimals = 2))
                n_Ramsey_fit = str(np.round_(pars_ramsey[2],decimals = 2))
                T2_Ramsey_fit= str(np.round_(pars_ramsey[1],decimals = 2))
                B_tau_c = str(np.round_(pars_corr[1],decimals = 2))

                # best fit of data
                (mu_r_1, sigma_r_1) = norm.fit(phi_ramsey_avg_EarlyTime)
                (mu_e_1, sigma_e_1) = norm.fit(phi_echo_avg_EarlyTime)
                (mu_r_12, sigma_r_12) = norm.fit(phi_ramsey_avg_EarlyTime2)
                (mu_e_12, sigma_e_12) = norm.fit(phi_echo_avg_EarlyTime2)
                (mu_r_2, sigma_r_2) = norm.fit(phi_ramsey_avg_LateTime)
                (mu_e_2, sigma_e_2) = norm.fit(phi_echo_avg_LateTime)
                (mu_b_1, sigma_b_1) = norm.fit(B_fields_EarlyTime)
                (mu_b_12, sigma_b_12) = norm.fit(B_fields_EarlyTime2)
                (mu_b_2, sigma_b_2) = norm.fit(B_fields_LateTime)

                (muC_r_1, sigmaC_r_1) = stats.cauchy.fit(phi_ramsey_avg_EarlyTime)
                (muC_e_1, sigmaC_e_1) = stats.cauchy.fit(phi_echo_avg_EarlyTime)
                (muC_r_12, sigmaC_r_12) = stats.cauchy.fit(phi_ramsey_avg_EarlyTime2)
                (muC_e_12, sigmaC_e_12) = stats.cauchy.fit(phi_echo_avg_EarlyTime2)
                (muC_r_2, sigmaC_r_2) = stats.cauchy.fit(phi_ramsey_avg_LateTime)
                (muC_e_2, sigmaC_e_2) = stats.cauchy.fit(phi_echo_avg_LateTime)

                plot1 = plt.figure(1)
                n_r_1, bins_r_1, patches_r_1 = plt.hist(phi_ramsey_avg_EarlyTime, 60, normed=1, facecolor='red', alpha=0.75)
                y_r_1 = mlab.normpdf( bins_r_1, mu_r_1, sigma_r_1)
                # pars_Lor_r_1, cov_Lor_r_1 = curve_fit(lorentzian, bins_r_1, phi_ramsey_avg_EarlyTime, p0=[0.5,0, 2], bounds=(0, np.inf))
                # plt.plot(phase, lorentzian(bins_r_1, *pars_Lor_r_1), linestyle='--' , linewidth=1.5, color='red')
                pdf_cauchy_r_1 = stats.cauchy.pdf(bins_r_1, muC_r_1, sigmaC_r_1)
                plt.plot(bins_r_1, pdf_cauchy_r_1,'b--',  linewidth=2, label="Lorentzian Fit")
                plt.plot(bins_r_1, y_r_1, 'g--', linewidth=2,label='Gaussian Fit')
                plt.xlabel('Phi Ramsey Early Time')
                plt.ylabel('Occurences')
                plt.title('Ramsey Early Time Phase, t={}'.format(t1/1000))
                plt.grid(True)
                plt.legend(prop={"size":16})

                plot2 = plt.figure(2)
                n_e_1, bins_e_1, patches_e_1 = plt.hist(phi_echo_avg_EarlyTime, 60, normed=1, facecolor='blue', alpha=0.75)
                y_e_1 = mlab.normpdf( bins_e_1, mu_e_1, sigma_e_1)
                # pars_Lor_e_1, cov_Lor_e_1 = curve_fit(lorentzian, bins_e_1, phi_echo_avg_EarlyTime, p0=[0.5,0,2], bounds=(0, np.inf))
                # plt.plot(phase, lorentzian(bins_e_1, *pars_Lor_e_1), linestyle='--' , linewidth=1.5, color='blue')
                pdf_cauchy_e_1 = stats.cauchy.pdf(bins_e_1, muC_e_1, sigmaC_e_1)
                plt.plot(bins_e_1, pdf_cauchy_e_1,'b--',  linewidth=2,label="Lorentzian Fit")
                plt.plot(bins_e_1, y_e_1, 'g--', linewidth=2,label='Gaussian Fit')
                plt.xlabel('Phi Echo Early Time')
                plt.ylabel('Occurences')
                plt.title('Echo Early Time Phase, t={}'.format(t1/1000))
                plt.grid(True)
                plt.legend(prop={"size":16})


                plot3 = plt.figure(3)
                n_r_12, bins_r_12, patches_r_12 = plt.hist(phi_ramsey_avg_EarlyTime2, 60, normed=1, facecolor='red', alpha=0.75)
                y_r_12 = mlab.normpdf( bins_r_12, mu_r_12, sigma_r_12)
                # pars_Lor_r_1, cov_Lor_r_1 = curve_fit(lorentzian, bins_r_1, phi_ramsey_avg_EarlyTime, p0=[0.5,0, 2], bounds=(0, np.inf))
                # plt.plot(phase, lorentzian(bins_r_1, *pars_Lor_r_1), linestyle='--' , linewidth=1.5, color='red')
                pdf_cauchy_r_12 = stats.cauchy.pdf(bins_r_12, muC_r_12, sigmaC_r_12)
                plt.plot(bins_r_12, pdf_cauchy_r_12,'b--',  linewidth=2, label="Lorentzian Fit")
                plt.plot(bins_r_12, y_r_12, 'g--', linewidth=2,label='Gaussian Fit')
                plt.xlabel('Phi Ramsey Early Time')
                plt.ylabel('Occurences')
                plt.title('Ramsey Early Time Phase, t={}'.format(t2/1000))
                plt.grid(True)
                plt.legend(prop={"size":16})

                plot4 = plt.figure(4)
                n_e_12, bins_e_12, patches_e_12 = plt.hist(phi_echo_avg_EarlyTime2, 60, normed=1, facecolor='blue', alpha=0.75)
                y_e_12 = mlab.normpdf( bins_e_12, mu_e_12, sigma_e_12)
                # pars_Lor_e_1, cov_Lor_e_1 = curve_fit(lorentzian, bins_e_1, phi_echo_avg_EarlyTime, p0=[0.5,0,2], bounds=(0, np.inf))
                # plt.plot(phase, lorentzian(bins_e_1, *pars_Lor_e_1), linestyle='--' , linewidth=1.5, color='blue')
                pdf_cauchy_e_12 = stats.cauchy.pdf(bins_e_12, muC_e_12, sigmaC_e_12)
                plt.plot(bins_e_12, pdf_cauchy_e_12,'b--',  linewidth=2,label="Lorentzian Fit")
                plt.plot(bins_e_12, y_e_12, 'g--', linewidth=2,label='Gaussian Fit')
                plt.xlabel('Phi Echo Early Time')
                plt.ylabel('Occurences')
                plt.title('Echo Early Time Phase, t={}'.format(t2/1000))
                plt.grid(True)
                plt.legend(prop={"size":16})



                plot5 = plt.figure(5)
                n_r_2, bins_r_2, patches_r_2 = plt.hist(phi_ramsey_avg_LateTime, 60, normed=1, facecolor='red', alpha=0.75)
                y_r_2 = mlab.normpdf( bins_r_2, mu_r_2, sigma_r_2)
                pdf_cauchy_r_2 = stats.cauchy.pdf(bins_r_2, muC_r_2, sigmaC_r_2)
                plt.plot(bins_r_2, pdf_cauchy_r_2,'b--',  linewidth=2,label="Lorentzian Fit")
                plt.plot(bins_r_2, y_r_2, 'g--', linewidth=2,label='Gaussian Fit')
                plt.xlabel('Phi Ramsey Late Time')
                plt.ylabel('Probability')
                plt.title('Ramsey late Time Phase, t={}'.format(t3/1000))
                plt.grid(True)
                plt.legend(prop={"size":16})

                plot6 = plt.figure(6)
                n_e_2, bins_e_2, patches_e_2 = plt.hist(phi_echo_avg_LateTime, 60, normed=1, facecolor='blue', alpha=0.75)
                y_e_2 = mlab.normpdf( bins_e_2, mu_e_2, sigma_e_2)
                # pars_Lor_e_2, cov_Lor_e_2 = curve_fit(lorentzian, bins_e_2, phi_echo_avg_LateTime, p0=[0.5,0,10], bounds=(0, np.inf))
                # plt.plot(phase, lorentzian(bins_e_2, *pars_Lor_e_2), linestyle='--' , linewidth=1.5, color='blue')
                pdf_cauchy_e_2 = stats.cauchy.pdf(bins_e_2, muC_e_2, sigmaC_e_2)
                plt.plot(bins_e_2, pdf_cauchy_e_2,'b--',  linewidth=2,label="Lorentzian Fit")
                plt.plot(bins_e_2, y_e_2, 'g--', linewidth=2,label='Gaussian Fit')
                plt.xlabel('Phi Echo Late Time')
                plt.ylabel('Probability')
                plt.title('Echo late Time Phase, t={}'.format(t3/1000))
                plt.grid(True)
                plt.legend(prop={"size":16})

                plot7 = plt.figure(7)
                n_b_1, bins_b_1, patches_b_1 = plt.hist(B_fields_EarlyTime, 60, normed=1, facecolor='green', alpha=0.75)
                y_b_1 = mlab.normpdf( bins_b_1, mu_b_1, sigma_b_1)
                l_b_1 = plt.plot(bins_b_1, y_b_1, 'g--', linewidth=2,label='B Fields at early time')
                plt.xlabel('B early Time')
                plt.ylabel('Probability')
                plt.title('B Early Time t={}, avg = {}, Std. Dev. = {}'.format(t1/1000, mu_b_1,sigma_b_1))
                plt.grid(True)
                plt.legend(prop={"size":16})

                plot8 = plt.figure(8)
                n_b_2, bins_b_2, patches_b_2 = plt.hist(B_fields_LateTime, 60, normed=1, facecolor='green', alpha=0.75)
                y_b_2 = mlab.normpdf( bins_b_2, mu_b_2, sigma_b_2)
                l_b_2 = plt.plot(bins_b_2, y_b_2, 'g--', linewidth=2,label='B Fields at late time')
                plt.xlabel('B Late Time')
                plt.ylabel('Probability')
                plt.title('B Late Time t={}, avg = {}, Std. Dev. = {}'.format(t3/1000, mu_b_2,sigma_b_2))
                plt.grid(True)
                plt.legend(prop={"size":16})

                plot9 = plt.figure(9)
                plt.plot(time, cos_phi_ramsey_avg, linestyle='-' , linewidth=2, color='red',label='Ramsey')
                plt.plot(time, cos_phi_echo_avg, linestyle='-' , linewidth=2, color='blue',label='Echo')
                plt.plot(time, stretched_Exp2(time, *pars_ramsey), linestyle='--' , linewidth=1.5, color='red',label='Ramsey Fit')
                plt.plot(time, stretched_Exp1(time, *pars_echo), linestyle='--' , linewidth=1.5, color='blue',label='Echo Fit')
                plt.xlabel(r'$\tau$',fontsize=20)
                plt.ylabel('Signal (a.u.)',fontsize=20)
                plt.title(r'$T_2 = {}, n_E = {}, T_2* = {}, n_R = {}, Occ. Prob.={}$'.format(T2_Echo_fit, n_Echo_fit, T2_Ramsey_fit, n_Ramsey_fit, OCP/qq),fontsize=20)
                plt.legend(prop={"size":16})

                plot10 = plt.figure(10)
                plt.plot(time, B_Correlation, linestyle='-' , linewidth=2, color='blue',label=r'$<B(t+\tau)B(t)>$')
                plt.plot(time, correlation_Fit(time, *pars_corr), linestyle='--' , linewidth=1.5, color='red',label='Fit')
                plt.xlabel(r'$\tau$',fontsize=20)
                plt.ylabel(r'$<B(t-\tau)B(t)> (a.u.)$',fontsize=20)
                plt.title(r'$\tau_c = {}$'.format(B_tau_c),fontsize=20)
                plt.legend(prop={"size":16})

                plot11 = plt.figure(11)
                plt.plot(time, B__fields, linestyle='-' , linewidth=2, color='blue',label=r'$B(t)$')
                plt.xlabel(r'$\tau$',fontsize=20)
                plt.ylabel('B(t) (a.u.)$',fontsize=20)
                plt.title(r'$\tau_c = {}$'.format(B_tau_c),fontsize=20)
                plt.legend(prop={"size":16})

                plot12 = plt.figure(12)
                cmap = colors.ListedColormap(['blue', 'white', 'orange', 'red'])
                bounds=[-1.5,-0.5,0.5,1.5,2.5]
                normm = colors.BoundaryNorm(bounds, cmap.N)
                imshow(lattice, interpolation='nearest', origin='lower', cmap=cmap, norm=normm)
                plt.title(r'$\tau_c = {}, 1/J_0 = {}$'.format(B_tau_c, (1/J0)*10**(-3)),fontsize=20)

                plt.show()

            print('Iteration #{} out of {} completed.'.format(ii+1, nn))


    return

# spin_decay_2d(0.04)














def RandomB_2D_vs_t(occ_prob, F0, tau_f):

    #Dimension of the 2x2 array square
    N = 30
    #Number of time steps in the dynamics for a given, initial, random spin configuration
    M = 1000
    # Define a time array for plotting B(t)
    time = np.zeros(M)
    for ii in range(0,M-1):
        time[ii] = ii/M
    time[M-1]=time[M-2]
    # Array to hold B and B**2 at each time step
    B_fields = np.zeros(M)
    B_fields_sq = np.zeros(M)
    lattice = np.zeros((N, N))
    # Define spin up (1) and spin down (-1) and populate the lattice randomly with spins accordingg to the occupation probability
    updown = [-1, 1]
    for ii in range(0, N-1):
        for jj in range(0, N-1):
            occ = random.uniform(0, 1)
            if occ<occ_prob:
                lattice[ii, jj] = random.choice(updown)
    # Define the ``central spin''
    lattice[np.int(N/2), np.int(N/2)] = 2


    # ------------------------- Simulate the dynamics of the spin bath by inducing random flip-flops between spins --------------------------------------- #
    for kk in range(0, M):
        print('{}% finished simulating the dynamics'.format(100*kk/M))
        B = 0
        #Calculate B for the current configuration
        for ii in range(0, N):
            for jj in range(0, N-1):
                if jj != np.int(N/2) and ii != np.int(N/2):
                    B = lattice[ii, jj]/(((np.abs(np.int(N/2)-jj))**2 + (np.abs(np.int(N/2)-jj))**2)**(3/2)) + B
        B_fields[kk] = B
        B_fields_sq[kk] = B**2
        # Induce random flip flops in the bath spins
        #brute force for now.... try making this more efficient later
        #ii and jj are looking for spin up in the first round, and for spin down in the second
        for ii in range(0, N-1):
            for jj in range(0, N-1):
                if lattice[ii,jj]==1:
                    # nn and mm are looking for a spin down to flip flop with
                    for nn in range(0, N-1):
                        for mm in range(0, N-1):
                            if lattice[nn,mm]==-1:
                                ff = random.uniform(0, 1)
                                if ((nn-ii)**2+(mm-jj)**2)**(3/2) !=0:
                                    if ff<np.abs(F0/((nn-ii)**2+(mm-jj)**2)**(3/2)):
                                        lattice[ii,jj]=-1
                                        lattice[nn,mm]=1
                    # Break the loop if the spin has flipped i.e. only allow one flip-flop per timestep
                    if lattice[ii,jj]==-1:
                        break
                elif lattice[ii,jj]==-1:
                    # nn and mm are looking for a spin down to flip flop with
                    for nn in range(0, N-1):
                        for mm in range(0, N-1):
                            if lattice[nn,mm]==1:
                                ff = random.uniform(0, 1)
                                if ((nn-ii)**2+(mm-jj)**2)**(3/2) !=0:
                                    if ff<np.abs(F0/((nn-ii)**2+(mm-jj)**2)**(3/2)):
                                        lattice[ii,jj]=1
                                        lattice[nn,mm]=-1
                    if lattice[ii,jj]==1:
                        break
                else:
                    lattice[ii,jj]==0

    #--------------------------------- Simulate random flip flops by a T1-type process with random matrices --------------------------------#
    # for kk in range(0, M):
    #     #print('{}% finished simulating the dynamics'.format(100*kk/M))
    #     B = 0
    #     #Calculate B for the current configuration
    #     for ii in range(0, N):
    #         for jj in range(0, N-1):
    #             if jj != np.int(N/2) and ii != np.int(N/2):
    #                 B = lattice[ii, jj]/(((np.abs(np.int(N/2)-jj))**2 + (np.abs(np.int(N/2)-jj))**2)**(3/2)) + B
    #     B_fields[kk] = B
    #     B_fields_sq[kk] = B**2
    #     # define a diagonal matrix f_mat where the enrtries are 1 (no spin flip) or -1 (spin flip), where the probability of flipping is pf
    #     pf = 0.01
    #     flip_mat = np.zeros((N,N))
    #     flop_mat = np.zeros((N,N))
    #     for ii in range(0, N-1):
    #         flip = random.uniform(0, 1)
    #         flop = random.uniform(0, 1)
    #         if flip<pf:
    #             flip_mat[ii, ii] = -1
    #         else:
    #             flip_mat[ii,ii] = 1
    #         if flop<pf:
    #             flop_mat[ii, ii] = -1
    #         else:
    #             flop_mat[ii,ii] = 1
    #     lattice = np.matmul(lattice, flip_mat)
    #     lattice = np.matmul(flop_mat, lattice)
    #
    # B_fields[M-1] = B_fields[M-2]



    # integrate B(t) to get phi(t) for various pulse sequences; here Ramsey and Echo

    phi_Ramsey = np.zeros(M)
    phi_Echo = np.zeros(M)
    phi_CPMG_2 = np.zeros(M)
    for t in range(0, M-1):
        phi_Ramsey[t] = phi_Ramsey[t-1]+(t/M)*B_fields[t]
        # need to define dummy variable of integration tt to integrate phi for echo
        for tt in range(0, t-1):
            if tt<(t/2):
                phi_Echo[t] = phi_Echo[t]+(tt/M)*B_fields[tt]
            else:
                phi_Echo[t] = phi_Echo[t]-(tt/M)*B_fields[tt]
        for ttt in range(0, t-1):
            if ttt<(t/3) or ttt>(2*t/3):
                phi_CPMG_2[t] = phi_CPMG_2[t]+(ttt/M)*B_fields[ttt]
            else:
                phi_CPMG_2[t] = phi_CPMG_2[t] - (ttt/M)*B_fields[ttt]
            #print('{}% done integrating B(t)'.format(100*t/M))


        #-----------------The following analysis/fits should be valid for gaussian noise, not necessarily for discrete/telegraph noise -----------------------#
        #-----------------In any case it is always an approximation when we have a finite sample size --------------------------------------------------------#


    # Obtain the spectral density of B(t)
    freq, Pxx_den = signal.periodogram(B_fields)
    #
    # # Fit the spectral density function above to various trial functions
    # def lorentzian(x, a, g, x0):
    #     return a*((g/2)/((g/2)**2+(x-x0)**2))
    # def inverse(t, b, c, t0):
    #     return b*t*((c/2)/((c/2)**2+(t-t0)**2))
    # max_Pxx = np.max(Pxx_den)
    # pars_Lorentzian, cov_Lorentzian = curve_fit(lorentzian, freq, Pxx_den, p0=[max_Pxx,0.4,0.1], bounds=(0, np.inf))
    # pars_Inverse, cov_Inverse = curve_fit(inverse, freq, Pxx_den, p0=[max_Pxx,0.4,0.01], bounds=(0, np.inf))

    #Integrate the PSD against the filter function for a spin echo and Ramsey
    def Echo_FF(w, t):
        return 8*(((np.sin(w*t/4))**4)/(w**2))
    def Ramsey_FF(f, y):
        return 2*(((np.sin(f*y/2))**2)/(f**2))

    # tau steps
    # NN = 10**3
    # Tau_ = np.zeros(NN)
    # for ii in range(0, NN-1):
    #     Tau_[ii] = tau_f*ii/NN
    # Tau_[NN-1] = Tau_[NN-2]
    # # freq steps
    # MM = np.size(freq)
    # dw = (freq[2] - freq[1])
    # # Define the integral as chi(tau)
    # chi_Echo = np.zeros(NN)
    # chi_Ramsey = np.zeros(NN)
    # # perform the integral as, for a given tau...
    # for tau in range(0,NN-1):
    #     print('{}% finished with the Chi integration'.format(100*tau/NN))
    #     #... perform the integral over omega
    #     for w in range(1,MM-1):
    #         chi_Echo[tau] = chi_Echo[tau] + dw*Echo_FF(w*dw, tau_f*tau/NN)*Pxx_den[w]
    #         chi_Ramsey[tau] = chi_Ramsey[tau] + dw*Ramsey_FF(w*dw, tau_f*tau/NN)*Pxx_den[w]

    # chi_Echo[NN-1] = chi_Echo[NN-2]
    # chi_Ramsey[NN-1] = chi_Ramsey[NN-2]
    # Echo_Decay = np.exp(-chi_Echo)
    # Ramsey_Decay = np.exp(-chi_Ramsey)
    # # Fit chi to a power law, equivalent to fitting the decay to a stretched exponential
    # def Chi_Echo_Power_Law(tau, T2, n_Echo):
    #     return (tau/T2)**n_Echo
    # def Chi_Ramsey_Power_Law(tt, T2s, n_Ramsey):
    #     return (tt/T2s)**n_Ramsey
    # pars_Chi_Echo, cov_Chi_Echo = curve_fit(Chi_Echo_Power_Law, Tau_, chi_Echo, p0=[20,4], bounds=(0, np.inf))
    # n_Echo_fit = str(np.round_(pars_Chi_Echo[1],decimals = 2))
    # T2_Echo_fit= str(np.round_(pars_Chi_Echo[0],decimals = 2))
    # pars_Chi_Ramsey, cov_Chi_Ramsey = curve_fit(Chi_Ramsey_Power_Law, Tau_, chi_Ramsey, p0=[3,3], bounds=(0, np.inf))
    # n_Ramsey_fit = str(np.round_(pars_Chi_Ramsey[1],decimals = 2))
    # T2_Ramsey_fit= str(np.round_(pars_Chi_Ramsey[0],decimals = 2))




    plot1 = plt.figure(1)
    plt.plot(time, B_fields)
    plt.xlabel('Time (a.u.)',fontsize=20)
    plt.ylabel('B(t) (a.u.)',fontsize=20)

    plot2 = plt.figure(2)
    cmap = colors.ListedColormap(['blue', 'white', 'orange', 'red'])
    bounds=[-1.5,-0.5,0.5,1.5,2.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    imshow(lattice, interpolation='nearest', origin='lower', cmap=cmap, norm=norm)

    # plot3 = plt.figure(3)
    # plt.semilogy(freq, lorentzian(freq, *pars_Lorentzian), linestyle='--', linewidth=2, color='purple',label='Lorentzian Fit')
    # plt.semilogy(freq, inverse(freq, *pars_Inverse), linestyle='--',linewidth=2, color='olive',label='1/f Fit')
    # plt.semilogy(freq, Pxx_den,linestyle='-', linewidth=2, color='black',label='Spectral Density')
    # plt.xlabel('frequency',fontsize=20)
    # plt.ylabel('Power Spectral Density',fontsize=20)
    # plt.xlim([0, 0.25])
    # plt.legend(prop={"size":16})


    # plot4 = plt.figure(4)
    # plt.plot(Tau_, Echo_Decay,linestyle='-' , linewidth=2, color='black',label='Echo')
    # plt.plot(Tau_, np.exp(-Chi_Echo_Power_Law(Tau_, *pars_Chi_Echo)),linestyle='--' , linewidth=2, color='red',label='Echo Fit' )
    # plt.plot(Tau_, Ramsey_Decay,linestyle='-' , linewidth=2, color='black',label='Ramsey')
    # plt.plot(Tau_, np.exp(-Chi_Ramsey_Power_Law(Tau_, *pars_Chi_Ramsey)), linestyle='--' , linewidth=2, color='blue',label='Ramsey Fit' )
    # plt.xlabel(r'$\tau$',fontsize=20)
    # plt.ylabel('Signal',fontsize=20)
    # plt.title(r'$T_2 = {}, n_E = {}, T_2* = {}, n_R = {}$'.format(T2_Echo_fit, n_Echo_fit, T2_Ramsey_fit, n_Ramsey_fit),fontsize=20)
    # plt.legend(prop={"size":16})

#    plot5 = plt.figure(5)
#    plt.plot(Tau_, chi_Echo,linestyle='-' , linewidth=2, color='black',label='Chi Echo')
#    plt.plot(Tau_, chi_Ramsey,linestyle='-' , linewidth=2, color='black',label='Chi Ramsey')
#    plt.plot(Tau_, Chi_Echo_Power_Law(Tau_, *pars_Chi_Echo), linestyle='--',linewidth=2, color='red',label='Chi Echo Fit to Power Law')
#    plt.plot(Tau_, Chi_Ramsey_Power_Law(Tau_, *pars_Chi_Ramsey), linestyle='--',linewidth=2, color='blue',label='Chi Ramsey Fit to Power Law')
#    plt.xlabel('Tau')
#    plt.ylabel('Chi(Tau)')
#    plt.title('T2_Echo = {}, n_Echo = {}, T2* = {}, n_Ramsey = {}'.format(T2_Echo_fit, n_Echo_fit, T2_Ramsey_fit, n_Ramsey_fit))
#    plt.legend()

    # plot5 = plt.figure(5)
    # plt.loglog(Tau_, chi_Echo, linestyle='-' , linewidth=2, color='black',label='Chi Echo')
    # plt.loglog(Tau_, chi_Ramsey, linestyle='-' , linewidth=2, color='black',label='Chi Ramsey')
    # plt.loglog(Tau_, Chi_Echo_Power_Law(Tau_, *pars_Chi_Echo), linestyle='--',linewidth=2, color='red',label='Chi Echo Fit to Power Law')
    # plt.loglog(Tau_, Chi_Ramsey_Power_Law(Tau_, *pars_Chi_Ramsey), linestyle='--',linewidth=2, color='blue',label='Chi Ramsey Fit to Power Law')
    # plt.xlabel(r'$\tau$',fontsize=20)
    # plt.ylabel(r'$\chi(\tau)$',fontsize=20)
    # plt.title(r'$T_2 = {}, n_E = {}, T_2* = {}, n_R = {}$'.format(T2_Echo_fit, n_Echo_fit, T2_Ramsey_fit, n_Ramsey_fit),fontsize=20)
    # plt.legend(prop={"size":16})

    # plot6 = plt.figure(6)
    # plt.plot(freq, Echo_FF(freq,tau_f), linestyle='-' , linewidth=2, color='magenta',label='Echo FF')
    # plt.plot(freq, Ramsey_FF(freq,tau_f), linestyle='-' , linewidth=2, color='cyan',label='Ramsey FF')
    # plt.xlabel('frequency',fontsize=20)
    # plt.ylabel('Filter function',fontsize=20)
    # plt.legend(prop={"size":16})

    plot3 = plt.figure(3)
    plt.plot(time, phi_Echo, linestyle='-' , linewidth=2, color='blue',label='Echo')
    plt.plot(time, phi_Ramsey, linestyle='-' , linewidth=2, color='red',label='Ramsey')
    plt.plot(time, phi_CPMG_2, linestyle='-' , linewidth=2, color='green',label='2 Pi Pulses')
    plt.xlabel(r'$\tau$',fontsize=20)
    plt.ylabel(r'$\phi(\tau)$',fontsize=20)
    plt.legend(prop={"size":16})

    plot4 = plt.figure(4)
    plt.hist(B_fields, bins=50)
    plt.xlabel('B (a.u.)',fontsize=20)
    plt.ylabel('# of Occurences in time',fontsize=20)

    plot5 = plt.figure(5)
    plt.hist(phi_Ramsey, bins=50, color='red')
    plt.xlabel(r'$\phi Ramsey (a.u.)$',fontsize=20)
    plt.ylabel('# of Occurences in time',fontsize=20)

    plot6 = plt.figure(6)
    plt.hist(phi_Echo, bins=50, color='blue')
    plt.xlabel(r'$\phi Echo (a.u.)$',fontsize=20)
    plt.ylabel('# of Occurences in time',fontsize=20)

    plot7 = plt.figure(7)
    plt.hist(phi_CPMG_2, bins=50, color='green')
    plt.xlabel(r'$\phi 2 Pi Pulses (a.u.)$',fontsize=20)
    plt.ylabel('# of Occurences in time',fontsize=20)

    plt.show()

    return


# RandomB_2D_vs_t(0.05, 1, 8)





def spin_diffusion_2d():


    def dissusion_length(occ_prob, F0):

        #Dimension of the 2x2 array square
        N = 35
        #Number of time steps in the dynamics for a given, initial, random spin configuration
        M = 200
        # Define a time array for plotting B(t)
        time = np.zeros(M)
        for ii in range(0,M-1):
            time[ii] = ii/M
        time[M-1]=time[M-2]
        # Array to hold B and B**2 at each time step
        B_fields = np.zeros(M)
        B_fields_sq = np.zeros(M)
        lattice = np.zeros((N, N))
        # Populate the lattice randomly with spins according to the occupation probability, but begin with them all polarized for a starting toy model
        for ii in range(0, N-1):
            for jj in range(0, N-1):
                occ = random.uniform(0, 1)
                if occ<occ_prob:
                    lattice[ii, jj] = -1
        # Define the site to which spin is to diffuse from
        lattice[np.int(N/2), np.int(N/2)] = 1


        # ------------------------- Simulate the dynamics of the spin bath by inducing random flip-flops between spins --------------------------------------- #
        # ------------------------- This is the only type of interaction to be studied here, since this is the only mechanism for spin transport I'm aware of in these systems -------#
        # ------------------------- e.g. T1 type processes do not conserve spin ---------------------------------------------------------------------------------#
        for kk in range(0, M):
#            print('{}% finished simulating the dynamics'.format(100*kk/M))
            # Induce random flip flops in the bath spins
            #brute force for now.... try making this more efficient later
            #ii and jj are looking for spin up in the first round, and for spin down in the second
            for ii in range(0, N-1):
                for jj in range(0, N-1):
                    if lattice[ii,jj]==1:
                        # nn and mm are looking for a spin down to flip flop with
                        for nn in range(0, N-1):
                            for mm in range(0, N-1):
                                if lattice[nn,mm]==-1:
                                    ff = random.uniform(0, 1)
                                    if ((nn-ii)**2+(mm-jj)**2)**(3/2) !=0:
                                        if ff<np.abs(F0/((nn-ii)**2+(mm-jj)**2)**(3/2)):
                                            lattice[ii,jj]=-1
                                            lattice[nn,mm]=1
                        # Break the loop if the spin has flipped i.e. only allow one flip-flop per timestep
                        if lattice[ii,jj]==-1:
                            break
                    elif lattice[ii,jj]==-1:
                        # nn and mm are looking for a spin down to flip flop with
                        for nn in range(0, N-1):
                            for mm in range(0, N-1):
                                if lattice[nn,mm]==1:
                                    ff = random.uniform(0, 1)
                                    if ((nn-ii)**2+(mm-jj)**2)**(3/2) !=0:
                                        if ff<np.abs(F0/((nn-ii)**2+(mm-jj)**2)**(3/2)):
                                            lattice[ii,jj]=1
                                            lattice[nn,mm]=-1
                        if lattice[ii,jj]==1:
                            break
                    else:
                        lattice[ii,jj]==0


        r = 0
        for ii in range(0, N-1):
            for jj in range(0, N-1):
                if lattice[ii, jj] == 1:
                    r = ((ii-np.int(N/2))**2 + (jj-np.int(N/2))**2)**(1/2)

        return r


    QQ = 400 # number of times to calculate r for a given occupation probability
    Occ_Prob = [0.00002, 0.0001, 0.0003, 0.0005, 0.001, 0.0015, 0.002,0.0025, 0.003, 0.01, 0.02, 0.1]
    ll = 12
    rr = np.zeros((ll, QQ))

    for ii in range(0, ll):
        for jj in range(0, QQ):
            rr[ii, jj] = dissusion_length(Occ_Prob[ii],1)
            print('{}% Finshed with occupation probability {}'.format(100*jj/QQ, Occ_Prob[ii]))

    r_avg = np.mean(rr, axis=1)
    r_stdev = np.std(rr, axis=1)
    print(r_avg)
    print(r_stdev)

#    # create figure and axis objects with subplots()
#    fig,ax = plt.subplots()
#    # make a plot
#    ax.plot(Occ_Prob, r_avg, color="red", marker="o")
#    # set x-axis label
#    ax.set_xlabel("Occupation Probability",fontsize=14)
#    # set y-axis label
#    ax.set_ylabel("Avg. Distance",color="red",fontsize=14)
#
#    # twin object for two different y-axis on the sample plot
#    ax2=ax.twinx()
#    # make a plot with different y-axis using second axis object
#    ax2.plot(Occ_Prob, r_stdev,color="blue",marker="o")
#    ax2.set_ylabel("Std. Dev.",color="blue",fontsize=14)
#    plt.show()

    plt.plot(Occ_Prob, r_avg, linestyle='-', linewidth=2, color='red', label='Avg. diffusion length')
    plt.plot(Occ_Prob, r_stdev, linestyle='-', linewidth=2, color='blue', label='Std. Dev.')
    plt.xlabel('Occupation Probability',fontsize=20)
    plt.ylabel('Distance',fontsize=20)
    plt.legend(prop={"size":16})
    plt.show()




    return

spin_diffusion_2d()
