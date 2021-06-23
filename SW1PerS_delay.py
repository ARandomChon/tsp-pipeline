"""
SW1PerS for time delay (tau).
=======================================================================

This function implements SW1PerS for the selection of the delay (tau) for permutation entropy. 
This method uses persistent homology over a rnage of embedding parameters to estimate the optimal 
tau from the lowest periodicity score. The periodicity score will decrease until the attractor has 
completely unfolded, which is in assocaition with an appropriate delay (tau).

Author:       Audun Myers

Date:         10/19/2018

Purpose:      Calculate the periodicity score using swipers and locate the best results based 
              lowest score
              
Input:        nT: number of time arrays per point cloud (or number of points in circle for ripser), should be 200 by default
              M: the emedding dimension (should be chosen as 2 times the number of fourier terms needed)
              signal: 1-D signal to be analyzed for periodicity
              fs: sample frequency should be provided if delay and window size are to be determined. (function won't work 
              without fs)
              if you dont have fs just make one up so that the function will still operate. won;t make a difference for 
              the plot.
              
Output:       Plot of L vs peridoicity score
              suggested window size and delay if sampling frequency is provided.
                
References:   Modified version of Dr. Firas Khasawneh's SW1PerS script for Python
              SW1PerS: Sliding windows and 1-persistence scoring; discovering periodicity in gene expression time series data
              SLIDING WINDOWS AND PERSISTENCE: AN APPLICATION OF TOPOLOGICAL METHODS TO SIGNAL ANALYSIS
"""


def MaxSigFreq(xf, yf, cutoff): 
    yf = yf/max(yf) #normalize series
    N = 0
    for i in range(len(yf)):
        if yf[i] > cutoff:
            N = xf[i]
    return N

def data_dimension(data): #finds the dimension of the data given
    shape = np.shape(data)
    dimension = min(shape)
    if len(shape) == 1:
        dimension = 1
    return dimension

def correct_data_format(data, dimension): #reformats the data for being analyzed
    data = data.reshape(dimension, max(np.shape(data)))
    if dimension == 1:
        data = data[0]
    return data

def LMS_for_offset(z_unsorted):
    import numpy as np
    n = len(z_unsorted) #length of data
    h = int((n/2)+1) #poisition of median value rounded up

    #sorts data in increasing order
    z = np.sort(z_unsorted)

    #Finding smallest difference
    z0 = np.split(z,[h-1])[0] #breaks z into two arrays for finding minimum difference
    z1 = np.split(z,[h-1])[1]
    l0 = len(z0)
    l1 = len(z1)

    if l0 != l1: #verifies the two arrays are the same length
        z1 = z1[:-1] #removes last element from x1 to make l1 and l0 the same size

    z_diff = z1-z0 #array of the difference between two halves of original ordered array
    z_diff_min_index = np.argmin(z_diff) #index of the minimum difference
    z_diff_min = z_diff[z_diff_min_index]
    
    #finding the best fit value ignoring outliers
    b = 0.5*z0[z_diff_min_index]+0.5*z1[z_diff_min_index] #finds best fit 1D value of noise floor
    return b, z_diff_min**2

def LMS(data): #only works for 1-d and 2-d currently
    dimension = data_dimension(data) #gets data dimension
    data = correct_data_format(data, dimension) #formats data properly
    b, Mm2 = LMS_for_offset(data) #calculates the best offset
    m = 0 #no slope for 1-D data
    return m,b

def AbsFFT(ts, fs):
    from scipy.fftpack import fft
    #time series must be one dimensional array
    fs = fs/2
    ts = ts.reshape(len(ts,))
    t = 1/fs #calculates the time between each data point
    N = len(ts)
    
    xf = np.split(np.linspace(0.0, 1.0/t, N//2),[1])[1] 
    #array of time data points to appropriate frequency range and removes first term

    yf = fft(ts) #computes fast fourier transform on time series
    yf = (np.abs(yf[0:N//2])) #converts fourier transform to spectral density scale
    yf = np.split(yf,[1])[1] #removes first term (tends to be infinity or overly large)
    yf = yf/max(yf) #Normalizes fourier transform based on maximum density
    
    return(xf,yf)

def FindLmax(ts, fs):
    xf, yf = AbsFFT(ts, fs)
    m, b = LMS(yf)
    cutoff = 5*b+0.001
    maxfreq = MaxSigFreq(xf, yf, cutoff) #max frequency in Hz
    if maxfreq == 0:
        maxfreq = 1
    #print(maxfreq)
    period = 1/maxfreq
    t_total = len(ts)/fs
    Lmax = t_total/period # "Lmax is the maximum number of expected periods"
    return Lmax

def sw1pers(signal, L, M, nT, coeffs = 11):
    #L is number of windows
    # M is embedding dimension
    #nT is number of points in point cloud
    # signal is just the time series
    
    import numpy as np
    from ripser import ripser
    from scipy import interpolate
    
    w = 2*np.pi/(L) #The theory behind SW1PerS implies that a good window size (w) ~ 2piM/(L*(M+1))
    tau = w/M               #time delay ~ w/M from theory behind SW1PerS
    print('tau: ', tau)
    nS = len(signal)        #number of data points in signal
    t = np.linspace(0, 2*np.pi, nS) #maps the time into [0,2pi]
    T = (2*np.pi - w)*np.linspace(0,1,nT) 
                            #window values of time
    
    Y = (np.ones(M+1)*T.reshape(len(T),1))
    Z = (tau*(np.arange(0,M+1))*np.ones(nT).reshape(len(np.ones(nT)),1))
    tt = (Y + Z).T 
    print('window size: ', w)
    
                            #array of time series for each window
    spline = interpolate.splrep(t, signal, s=0) 
    
                            #curve fits data from signal to cubic spline over desired length in time
    cloud = interpolate.splev(tt, spline, der=0) 
                            #uses curved fit response to find data points (interpolated) for tt
    plt.plot(t, signal)
    plt.plot(tt.T[0], cloud.T[0], 'r.')
    plt.show()
    # pointwise center of cloud
    cloud = cloud - np.ones(M+1).reshape(M+1,1)*np.mean(cloud, axis = 0)
    
    # pointwise normalize of cloud (rowsise)
    d = np.ones(M+1).reshape(M+1,1)*(np.sum(cloud**2, axis=0))**0.5
    cloud = cloud/d
    
    # transpose and apply square form in preparation to give cloud to Ripser
    plt.plot(cloud[0], cloud[1], 'k.')
    plt.show()
    cloud = cloud.T
    cloud = scipy.spatial.distance.pdist(cloud)
    dist_matrix = scipy.spatial.distance.squareform(cloud)
    #calls for ripser to calculate persistence homology
    dgms = ripser(dist_matrix, maxdim=1, distance_matrix=True, metric='euclidean')['dgms']
    printing = False
    if printing == True:
        
        MS = 10
        plt.title('(E)', loc='left')
        plt.plot(dgms[0][:-1].T[0],dgms[0][:-1].T[1] ,'rX', markersize = MS)
        plt.plot([0,2],[0,2],'k--')
        plt.plot(dgms[1].T[0],dgms[1].T[1] ,'g^', markersize = MS)
        plt.xlabel('Birth', size = TextSize)
        plt.ylabel('Death', size = TextSize)
        plt.show()
        pd = dgms[1].T
        lifetimes = pd[1]-pd[0]
        max_lifetime = max(lifetimes)
        print(max_lifetime)
        
    if len(dgms[1]) == 0:
        score = 1
    else:
        pd = dgms[1].T
        lifetimes = pd[1]-pd[0]
        max_lifetime = max(lifetimes)
        score = 1-max_lifetime/(3**0.5)
    return score


# In[ ]:


if __name__ == '__main__':
    import numpy as np
    import scipy
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy.integrate import odeint
    
    fs = 10
    t = np.linspace(0, 100, fs*100) 
    ts = np.sin(t)

    fs = 1/(t[5]-t[4])
    t = np.linspace(0,(max(t)-min(t)),len(ts))
  
    #_Set parameters for SW1PerS__
    nT = 200 #number of points in point cloud
    M = 2 #embedding dimension
    
    Lmax = (FindLmax(ts, fs)) # Lmax is the number of expected periods
    score = []
    L = []
    #calculates scores/run sw1pers
    for i in np.linspace(0.1*Lmax, 2*Lmax, 20):
        L.append(i)
        score.append(sw1pers(ts, i , M, nT, coeffs = 11))
    score = np.array(score)
    L = np.array(L)
    I = np.argmax(-score)  
    P = (M/(M+1))*(max(t)/L[I])
    a = 3
    tau = P*fs/(a)
        
    print('window size:           '+str(P))
    print('embedding delay:       '+str(round(tau,0)))
    print('embedding delay lower: '+str(round(a*tau/4,0)))
    print('embedding delay upper: '+str(round(a*tau/2,0)))
        
        
    # ______________Plotting____________________
    TextSize = 15
    gs = gridspec.GridSpec(1, 2) 
    plt.figure(1) 
    plt.figure(figsize=(12,4))
        
        
    ax = plt.subplot(gs[0, 0]) #______________PLOT a________
    plt.plot(t,ts)
    plt.xlim(min(t),max(t))
    plt.xticks(size = TextSize)
    plt.yticks(size = TextSize)
    plt.ylabel('Signal', size = TextSize)
    plt.xlabel('Time (seconds)', size = TextSize)
    plt.plot([0,P],[np.average(ts),np.average(ts)],'r')
        
    ax = plt.subplot(gs[0, 1]) #______________PLOT b_______
    plt.plot(L,score, color = 'black', marker = '.')
    plt.plot(L[I],score[I],'ro', label = 'Min, score at L = '+ str(L[I]))
    plt.ylim(0,1)
    plt.xlim(min(L),max(L))
    periods = int(Lmax/1.25)
    #plt.plot([periods,periods],[0,1],'g--', label = 'Max. number of periods: '+str(round(periods,2)))
    plt.xticks(size = TextSize)
    plt.yticks(size = TextSize)
    plt.ylabel('Periodicity Score', size = TextSize)
    plt.xlabel('L parameter', size = TextSize)
    plt.legend( loc=0, borderaxespad=0.)
        
    plt.subplots_adjust(hspace=0.25)
    plt.subplots_adjust(wspace=0.25)
    plt.savefig('C:\\Users\\myersau3.EGR\\Desktop\\python_png\\sw1pers_fig.png', bbox_inches='tight',dpi = 400)
    plt.show()
    
    