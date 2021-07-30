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

def LMS(data): #only works for 1-d and 2-d currently
    dimension = data_dimension(data) #gets data dimension
    data = correct_data_format(data, dimension) #formats data properly
    b, Mm2 = LMS_for_offset(data) #calculates the best offset
    m = 0 #no slope for 1-D data
    return m,b

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



def AbsFFT(ts, fs):
    from scipy.fftpack import fft
    #time series must be one dimensional array
    fs = fs/2
    t = 1/fs #calculates the time between each data point
    ts = ts.reshape(len(ts,))
    N = len(ts)
    
    xf = np.split(np.linspace(0.0, 1.0/t, N//2),[1])[1] 
    #array of time data points to appropriate frequency range and removes first term

    yf = fft(ts) #computes fast fourier transform on time series
    yf = (np.abs(yf[0:N//2])) #converts fourier transform to spectral density scale
    yf = np.split(yf,[1])[1] #removes first term (tends to be infinity or overly large)
    yf = yf/max(yf) #Normalizes fourier transform based on maximum density
    
    return(xf,yf)

def FindLmax(ts, fs, plotting = False):
    xf, yf = AbsFFT(ts, fs)
    m, b = LMS(yf)
    cutoff = 5*b+0.01
    maxfreq = MaxSigFreq(xf, yf, cutoff) #max frequency in Hz
    if maxfreq == 0:
        maxfreq = 1
    period = 1/maxfreq
    t_total = len(ts)/fs
    
    if plotting == True:
        plt.plot(np.linspace(0,t_total, len(ts)), ts)
        plt.plot([0, period],[0,0], 'r--')
        plt.show()
        
        plt.plot(xf, yf)
        plt.plot([maxfreq, maxfreq], [0,1], 'r--')
        plt.show()
    Lmax = t_total/period # "Lmax is the maximum number of expected periods"
    return Lmax

def sw1pers(signal, w, M, nT, coeffs = 11, plotting = False):
    
    import numpy as np
    from ripser import ripser
    from scipy import interpolate
    
    #w = 2*np.pi*M/(L*(M+1)) #The theory behind SW1PerS implies that a good window size (w) ~ 2piM/(L*(M+1))
    tau = w/M               #time delay ~ w/M from theory behind SW1PerS
    nS = len(signal)        #number of data points in signal
    t = np.linspace(0, 2*np.pi, nS) #maps the time into [0,2pi]
    T = (2*np.pi - M*tau)*np.linspace(0,1,nT) 
                            #window values of time
    Y = (np.ones(M+1)*T.reshape(len(T),1))
    Z = (tau*(np.arange(0,M+1))*np.ones(nT).reshape(len(np.ones(nT)),1))
    tt = (Y + Z).T 
                            #array of time series for each window
    spline = interpolate.splrep(t, signal, s=0) 
                            #curve fits data from signal to cubic spline over desired length in time
    cloud = interpolate.splev(tt, spline, der=0) 
                            #uses curved fit response to find data points (interpolated) for tt
    # pointwise center of cloud
    cloud = cloud - np.ones(M+1).reshape(M+1,1)*np.mean(cloud, axis = 0)
    
    # pointwise normalize of cloud (rowsise)
    d = np.ones(M+1).reshape(M+1,1)*(np.sum(cloud**2, axis=0))**0.5
    cloud = cloud/d
    x_cloud = cloud[0]
    y_cloud = cloud[1] 
    # transpose and apply square form in preparation to give cloud to Ripser
    cloud = cloud.T
    cloud = scipy.spatial.distance.pdist(cloud)
    dist_matrix = scipy.spatial.distance.squareform(cloud)
    #calls for ripser to calculate persistence homology
    dgms = ripser(dist_matrix, maxdim=1, coeff=coeffs, distance_matrix=True, metric='euclidean')['dgms']
    
    if plotting == True:
        gs = gridspec.GridSpec(1, 3) 
        plt.figure(1) 
        MS = 10
        plt.figure(figsize=(14,4))
           
        ax = plt.subplot(gs[0, 0])
        ax.plot(t, signal, 'k')
        ax.plot([0,w],[0,0],'r')
        
        ax = plt.subplot(gs[0, 1])
        ax.plot(x_cloud, y_cloud, 'ko')
        
        ax = plt.subplot(gs[0, 2])
        ax.plot(dgms[0][:-1].T[0],dgms[0][:-1].T[1] ,'rX', markersize = MS)
        ax.plot([0,2],[0,2],'k--')
        ax.plot(dgms[1].T[0],dgms[1].T[1] ,'g^', markersize = MS)
        plt.xlabel('Birth', size = TextSize)
        plt.ylabel('Death', size = TextSize)
        plt.show()
    
    if len(dgms[1]) == 0:
        score = 1
    else:
        pd = dgms[1].T
        lifetimes = pd[1]-pd[0]
        max_lifetime = max(lifetimes)
        score = 1-max_lifetime/(3**0.5)
    return score

def N_SW1PerS(ts, cutoff = 0.1, plotting = False):
    # this function uses a fourier reconstruction to determine number of terms to get a low error.
    import numpy as np
    y = ts 
    error = cutoff+1 #initially set error to be greater than cutoff
    i = 0
    y = y-np.mean(y)
    y = y/max(y)
    n = len(y)
    while error>cutoff:
        i = i+1
        Y = np.fft.fft(y) #take FFT
        I = np.argsort(Y)
        I_insig = I[0:n-i] #indices of len(ts) - n insignificant fourier terms.
        np.put(Y, I_insig, 0.0)
        ifft = np.fft.ifft(Y) #take inverse FFT
        ifft = ifft-np.mean(np.real(ifft)) #center around zero
        ifft = ifft/max(np.real(ifft)) #normalize
        
        error = sum(abs((y-np.real(ifft))))/sum(abs(y)) #find error between two
        if plotting == True: 
            plt.plot(ifft, label = str(i)+'-term reconstruction, Err. = '+str(round(100*error,2))+'\%')
        N = i #store number of terms needed for error less than cutoff (10%) as N
        
    if plotting == True: 
        print('k/n: ', N/n)
        plt.plot(y, 'k--', label = 'Actual Time Series (normalized)')
        plt.legend(loc = 'upper right', bbox_to_anchor=(1.65, 1.0))
        plt.show()
    return N


# In[ ]:


if __name__ == '__main__':
    import numpy as np
    import scipy
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    

    t= np.linspace(0,10,500)
    ts = np.sin(2*np.pi*t)
    ts = ts
    t = np.linspace(0,(max(t)-min(t)),len(ts))
    
    
    
    
    
    
    #_Set parameters for SW1PerS__
    fs = 1/(t[1]-t[0]) #sampling frequency
    nT = 200
    N = N_SW1PerS(ts, cutoff = 0.25, plotting = False)
    M = 2*N
    Lmax = FindLmax(ts, fs, plotting = False) # Lmax is the maximum number of expected periods
    w_max = 2*np.pi*M/(Lmax*(M+1))
    st = 2
    scores = []
    windows = np.linspace(0.001*w_max,2.0*w_max, 30)
    #calculates scores/run sw1pers
    for  w in windows:
        score = sw1pers(ts, w , M, nT, coeffs = 11, plotting  = False)
        scores.append(score)
    I = np.argmax(-np.array(scores))
    window_opt = windows[I]
    print(window_opt)
    tau = int(window_opt*fs/(M))+1

    print('')
    print('Number of significant frequencies (N): ', N)
    print('------Parameters For Embedding-------')
    print('embedding delay (tau):    ', tau)
    print('embedding dimension (2N): ', M)
        
        
    # ______________Plotting____________________
    TextSize = 15
    gs = gridspec.GridSpec(1, 2) 
    plt.figure(1) 
    plt.figure(figsize=(12,4))
        
    ax = plt.subplot(gs[0, 0]) #______________PLOT a________
    plt.plot(t,ts, label = 'time series')
    plt.xlim(min(t),max(t))
    plt.xticks(size = TextSize)
    plt.yticks(size = TextSize)
    plt.ylabel('Signal', size = TextSize)
    plt.xlabel('Scaled Time (seconds)', size = TextSize)
    plt.plot([0,window_opt],[np.average(ts),np.average(ts)],'r', label = 'window size')
    plt.legend(loc = 'upper right')
        
    ax = plt.subplot(gs[0, 1]) #______________PLOT b_______
    plt.plot(windows,scores, color = 'black', marker = '.')
    plt.plot(windows[I],scores[I],'ro', label = 'Min score at w = '+ str(round(windows[I],3)))
    plt.ylim(0,1)
    plt.xlim(min(windows),max(windows))
    plt.xticks(size = TextSize)
    plt.yticks(size = TextSize)
    plt.ylabel('Periodicity Score', size = TextSize)
    plt.xlabel('Window Size', size = TextSize)
    plt.legend( loc=0, borderaxespad=0.)
        
    plt.subplots_adjust(hspace=0.25)
    plt.subplots_adjust(wspace=0.25)
    plt.show()
    
    