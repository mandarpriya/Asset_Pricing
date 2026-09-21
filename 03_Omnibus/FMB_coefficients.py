import numpy as np

def FMB_coefficients_func(R,f,with_intercept):
    [T,K] = f.shape
    N = np.size(R, axis=1)
    
    # Time Series Regression to determine Beta (Cochrane, 2005, p.230, ff)
    F = np.array(np.concatenate((np.ones((T,1),float),f), axis=1))
    FFi = np.linalg.solve(F.T @ F, np.identity(K+1))             
    B = FFi @ F.T @ R
    alpha = np.array(B[0]).T                                   # Alphas
    beta = np.array(B[1:K+1]).T                                # Betas
    R_bar = np.mean(R, axis=0).T                               # Average returns
    
    # Cross-sectional regression to determine Lambda
    
    if with_intercept == 1:
        
        X = np.concatenate((np.ones((N,1)), beta), axis = 1)
        A = np.linalg.solve(X.T @ X,X.T)
        Theta = A @ R_bar
        Lambda = np.array(Theta[1:K+1])                       
        const = Theta [0]
    
    else:
        A = np.linalg.solve((beta.T @ beta),beta.T)
        Lambda = A @ R_bar                                     
        const = np.nan
        
    return [const, Lambda]