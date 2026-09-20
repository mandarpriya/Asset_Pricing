# Implements a circular block bootstrap for bootstrapping stationary, dependent series
# 
# INPUTS:
#    
#  DATA   - T by 1 vector of data to be bootstrapped
#  B      - Number of bootstraps
#  W      - Block length
#  Seed   - Seed for random numbers
#
# OUTPUTS:
#    
#  BSDATA  - T by B matrix of bootstrapped data
#  INDICES - T by B matrix of locations of the original BSDATA=DATA(indexes);
# 
# COMMENTS:
#    
#   To generate bootstrap sequences for other uses, such as bootstrapping vector processes,
#   set DATA to (0:N)'.
#
# See also stationary_bootstrap
#
# Original code in MATLAB
# Author: Kevin Sheppard
# kevin.sheppard@economics.ox.ac.uk
# Revision: 2    Date: 12/31/2001

import numpy as np

def block_bootstrap_func(data, B, w,  seed):
    
    # Input Checking
    [t,k] = data.shape # Get length of data
    
    if k>1:
        raise ValueError("DATA must be a column vector")
    
    if t<2:
        raise ValueError("DATA must have at least 2 observations")
    
    if not np.isscalar(w) or w<1 or np.floor(w) != w or w>t:
        raise ValueError("W must be a positive scalar integer smaller than T")
    
    if not np.isscalar(B) or B<1 or np.floor(B) != B:
        raise ValueError("B must be a positive scalar integer")
        
    # Compute the numbers of blocks needed
    s = int(np.ceil(t/w))
    
    # Generate the starting points
    #np.random.seed(seed)
    Bs = np.floor(np.random.rand(B,s)*t).T
    indices = np.zeros((s*w,B), dtype = int)
    index = 0
    # Adder is a variable that needs to be added each loop
    adder = np.matlib.repmat(range(0,w),1,B)
    
    for i in range(0,t,w):
        indices[i:(i+w),:] = np.matlib.repmat(Bs[index,:],1,w) + adder
        index += 1
        
    data = np.squeeze(data)
    indices = indices[0:t,:]
    indices[indices > t] = indices[indices > t] - t
    bsdata = data[indices]
    
    return [bsdata, indices]           
        
        
        
        
        
        
        
        
        
        
    