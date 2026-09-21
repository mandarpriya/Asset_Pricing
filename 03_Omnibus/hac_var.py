# This function comes from Craig Burnside, 
# The Cross Section of Foreign Currency Risk Premia and Consumption Growth Risk: Comment
# AMERICAN ECONOMIC REVIEW, VOL. 101, NO. 7, DECEMBER 2011
# https://www.aeaweb.org/articles?id=10.1257/aer.101.7.3456
# 
# compute a HAC or VARHAC estimator of the long-run covariance matrix of GMM errors 
# original code in MATLAB
#
# INPUTS:
# e:  error matrix, T x n
# K1: maximum lag order considered for own lags
# K2: maximum lag order for other lags
# if K1=K2=0, HAC standard errors are computed

# OUTPUT:
# S: the estimated spectral density matrix of the errors

import numpy as np

def hac_var_func(e, K1, K2):
    if K1 < K2:
        raise ValueError("K1 cannot be less than K2")
    else:
        K = max(K1, K2)
        [T, n] = np.shape(e)
        su = np.std(e, axis = 0, ddof = 0)
        u = np.divide(e, (np.ones((T,1)) @ su[:, None].T)) # normalize the std. dev. of the GMM errors to 1 (undo this later)
        
        if K != 0:
            v = u[K:T, :]
            A = np.zeros((n, K*n)) # initialize the A matrix
            
            for j in range(0,n):
                # work in jth equation
                y = u[K:T, j]
                y = y[:, None]
                BIC = np.log(y.T @ y) # initialize BIC
                
                for k in range(0, K1+1):
                    # this is the number of lags of variable j to enter
                    if k == 0:
                        minit = 1;
                    elif k == 1:
                        x1 = u[K-1:T-1, j]                    
                        minit = 0
                    else: 
                        x1 = np.concatenate((x1, u[K-k:T-k,j][:, None]), axis = 1)
                        minit = 0
                        
                    for m in range(minit,K+1):
                        u = np.divide(e, (np.ones((T,1)) @ su[:, None].T))
                        y = u[K:T, j]
                        y = y[:, None]
                        
                        if k == 0 and m == 1:
                            x2 = u[K-1:T-1, np.delete(range(0,n), j)]
                            x = x2
                        elif k == 0 and m > 1:
                            x2 = np.concatenate((x2, u[K-m:T-m, np.delete(range(0,n), j)]), axis = 1)
                            x = x2
                        elif k == 1 and m == 0:
                            x1 = x1[:, None]
                            x = x1
                        elif k == 1 and m == 1:
                            x2 = u[K-1:T-1, np.delete(range(0,n), j)]
                            x = np.concatenate((x1, x2), axis = 1)
                        elif k == 1 and m > 1:
                            x2 = np.concatenate((x2, u[K-m:T-m, np.delete(range(0,n), j)]), axis = 1)
                            x = np.concatenate((x1, x2), axis = 1)
                        elif k > 1 and m == 0:
                            x = x1
                        elif k > 1 and m == 1:
                            x2 = u[K-1:T-1, np.delete(range(0,n), j)]
                            x = np.concatenate((x1, x2), axis = 1)
                        else:
                            x2 = np.concatenate((x2, u[K-m:T-m, np.delete(range(0,n), j)]), axis = 1)
                            x = np.concatenate((x1, x2), axis = 1)
                            
                        if m <= K2:
                            b = np.linalg.solve((x.T @ x), (x.T @ y))
                            if np.shape(b)[0] > 1:
                                vv = y - x @ b
                            else:
                                vv = y - x * b
                                
                            # NumPy >= 2.3: float() on an ndim>0 array is a TypeError
                            # (deprecated in 1.25). vv is (T,1) so vv.T @ vv is (1,1).
                            # np.squeeze makes it 0-d. Only change vs the authors' file.
                            bic = float(np.squeeze(np.log(vv.T @ vv) + (k+m * (n-1)) * np.log(T-K) / (T-K)))
                            
                            if bic < BIC:
                                BIC = bic
                                sizb = max(k,m) * n
                                bb = np.zeros((sizb,1))
                                
                                if k > 0:
                                    bb[j:(k-1)*n+j+1:n, 0][:, None] = b[0:k,:]
                                if m > 0:
                                    for p in range(0,m):
                                        if len(b[k+(n-1)*p:k+(n-1)*p+j,0]) > 0:
                                            bb[p*n:p*n+j,0] = b[k+(n-1)*p:k+(n-1)*p+j,0]
                                            bb[p*n+j+1:p*n,0] = b[k+(n-1)*p+j+1:k+(n-1)*p,0]
                                        else:
                                            bb[p*n+1:(p+1)*n+j,0] = b[k+(n-1)*p:k+(n-1)*(p+1)+j, 0]
                                    
                                A[j, 0:sizb] = bb.T
                                v[:,j] = vv[:,0]
            
            B = np.concatenate((np.eye(n), (-A)), axis = 1)
            Sv = (v.T @ v) / (T-K) 
            sumB = np.diag(np.sum(B, axis = 1))
            S1 = np.linalg.inv(sumB) @ Sv @ np.linalg.inv(sumB).T
            
        else:
            u = np.divide(e, (np.ones((T,1)) @ su[:, None].T))
            S1 = (u.T @ u) / float(T) # no correction for serial correlation
        
        S = S1 * (su[:, None] @ su[:, None].T) # scale the covariance matrix back up
        
        return S
            
            
                
                
                
                
                                