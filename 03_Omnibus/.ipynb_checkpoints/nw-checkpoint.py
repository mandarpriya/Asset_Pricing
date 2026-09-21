# nw.py
# Newey-West estimate of covariance matrix with
# possible automatic lag selection
#
#    h: a Txr matrix of data
#    lag: number of lags (when lag is an empty matrix 
#    or lag<0, it is for automatic lag selection)
#    prewhite: an indicator of whether to do pre-whitening
#              or not (default is no), selection by 0 (false) or 1 (true)
#    V: covariance matrix of h
#

import numpy as np

def nw_func(h, lag, prewhite=0):
    [T, r] = h.shape
    # Demean h
    h = h - np.matlib.repmat(np.mean(h, axis = 0),T,1)
    
    # Number of lags is defined by user
    if  lag >= 0:
        V = (h.T @ h) / T
        for i in range(0,lag):
            V1 = (h[i+1:T,:].T @ h[0:(T-i-1),:])/T
            V = V + (1- (i+1)/(lag+1)) * (V1 + V1.T)
        return V
    
    #Automatic lag-selection
    # First step: pre-whitening by fitting a VAR(1)        
    else:
        
        if prewhite == 1:
             h0 = h[0:(T-1),:]
             h1 = h[1:T,:]
             A = np.linalg.lstsq(h0, h1, rcond = None)[0]  # Note that our A is A' in Newey West (1994)
             he = h1 - h0 @ A
        else:
            he = h
            
        T1 = np.size(he, axis=0)
        n = int(np.fix(12*(0.01*T) ** (2/9)))
        
        # Compute autocorrelation coefficients of w'he
        w = np.ones((r,1))
        hw = he @ w
        sigmah = np.zeros((n,1))
        
        for i in range(0,n):
            sigmah[i,:] = (hw[0:(T1-i-1)].T @ hw[i+1:T1]) / T1
        
        sigmah0 = (hw.T @ hw)/T1
        
        # Compute s0 and s1 and set bandwidth parameter
        
        s0 = sigmah0 + 2 * np.sum(sigmah, axis = 0)
        s1 = 2 * (np.array(range(1,n+1)) @ sigmah)[:,None]
        gam = 1.1447 * np.abs(s1/s0) ** (2/3)
        m = int(np.fix(gam * T ** (1/3)))
        V = (he.T @ he)/ T1
        
        for i in range(0,m): 
            V1 = (he[(i+1):T1,:].T @ he[0:(T1-i-1),:])/T1;
            V = V + (1- (i+1)/(m+1)) * (V1+V1.T);
    
        if prewhite == 1:
            IA = np.linalg.inv(np.eye(r) - A.T)
            V = IA @ V @ IA.T
                            
        return V
        

      
    
    