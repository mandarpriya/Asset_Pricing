"""
omnibus_section1.py  --  TRIMMED extract of Omnibus.py (Kroencke & Thimme, 2021).

Contains ONLY:
    Section 0  (betas, cross-sectional Lambda, pricing errors, R2 / R2i /
                R2_gls / R2i_gls, the `ans` dict)
    Section 1  (methods 1.01 - 1.06: tests of whether betas are zero)

Extracted verbatim from Omnibus.py lines 139-413, then closed with `return ans`.
Not my code -- see README.md in this folder. Cite Kroencke & Thimme (2021).

Usage:
    import sys; sys.path.insert(0, '<path to 05_Omnibus>')
    from omnibus_section1 import omnibus_section1
    ans = omnibus_section1(R, f, 1.01)
    ans['Test'], ans['pval']
"""

import numpy as np
import numpy.matlib as npmatlib
import math
import scipy.stats as sps
import scipy.linalg as spl
import scipy as spy

# Import subfunctions
import cdfchic
import nw
import hac_var
import block_bootstrap
import FMB_coefficients
import linchi2


def omnibus_section1(R, f, method, Lambda0 = 0, level = 0.05):
    
    #------------------------------------------------------------------------------
    # SECTION 0) GENERAL 
    #------------------------------------------------------------------------------
    
    lags = 3       # set to -1 to select Andrews optimal bandwith
    excess_ret = 0  # set to 0 if gross returns are used (relevant for Kleibergen et al statistics)
    traded_f = 1    # set to 1 if the factor is traded
    
    [T,K] = f.shape
    N = np.size(R, axis=1)
    
    if Lambda0 == 0 and np.isin(math.floor(method), [1,2,4,6]):
        Lambda0  = np.zeros((K,1))
    
    elif Lambda0 == 0 and np.isin(math.floor(method), [3,5]):
        Lambda0  = np.zeros((K+1,1))             
    
    # Time Series Regression to determine Beta (Cochrane, 2005, p.230, ff)
    F = np.array(np.concatenate((np.ones((T,1),float),f), axis=1))
    FFi = np.linalg.solve(F.T @ F, np.identity(K+1))             
    B = np.matmul(np.matmul(FFi, F.T), R)
    alpha = np.array(B[0]).T                                   # Alphas
    beta = np.array(B[1:K+1]).T                                # Betas
    e = R - F @ B                                              # Residuals
    SSR = e.T @ e                                              # Sum of squared residuals
    R_bar = np.mean(R, axis=0).T                               # Average returns  
    SST = (R - npmatlib.repmat(R_bar.T,T,1)).T @ (R - npmatlib.repmat(R_bar.T,T,1))  # Sum of squared demeaned returns      
    S = (1/T) * SSR                                            # Covariance matrix of residuals under i.i.d.-assumption
    
    # Cross-sectional regression to determine Lambda (Cochrane, 2005, p.235 ff & Burnside, 2011, AER)
    
    if np.isin(math.floor(method), [1,2,4]): # cross-sectional regression without intercept
    
         A = np.linalg.solve((beta.T @ beta),beta.T)
         Lambda = A @ R_bar                                     # OLS cross-sectional 
         Lambda_gls = np.linalg.solve(np.linalg.solve(np.cov(R, rowvar = False).T, beta).T @ beta, np.linalg.solve(np.cov(R, rowvar = False).T, beta).T @ R_bar) # GLS cross-sectional regression
         const = np.nan
         const_gls = np.nan
          
    else : # cross-sectional regression with intercept (constant IS NOT part of the pricing errors)
    
         X = np.concatenate((np.ones((N,1), float), beta), axis = 1)
         A = np.linalg.solve(X.T @ X,X.T)
         Theta = A @ R_bar
         Lambda = np.array(Theta[1:K+1])                       # OLS cross-sectional
         Theta_gls = np.linalg.solve(np.linalg.solve(np.cov(R, rowvar = False).T, X).T @ X, np.linalg.solve(np.cov(R, rowvar = False).T,X).T @ R_bar) 
         Lambda_gls = np.array(Theta_gls[1:K+1])               # GLS cross-sectional regression
         const = Theta [0]
         const_gls = Theta_gls [0]
    
    # OLS and GLS pricing errors and cross-sectional R^2
    
    PE = R_bar - beta @ Lambda                                                      # pricing errors
    R2 = 1- PE.T @ PE / ((R_bar - np.mean(R_bar)).T @ (R_bar - np.mean(R_bar)))     # cross-sectional R^2
    PEi = R_bar - beta @ Lambda - const                                             # pricing errors without imposing zero intercept
    R2i = 1- PEi.T @ PEi / ((R_bar - np.mean(R_bar)).T @ (R_bar - np.mean(R_bar)))  # cross-sectional R^2 without imposing zero intercept
     
    PE_gls = R_bar - beta @ Lambda_gls
    mu = np.linalg.solve(np.linalg.solve(np.cov(R, rowvar = False).T, np.ones((N,1), float)).T @ np.ones((N,1), float), np.linalg.solve(np.cov(R, rowvar = False).T, R_bar[:, None]).T @ np.ones((N,1), float))[:,0]
    R2_gls = np.squeeze(1 - np.linalg.solve(((np.linalg.solve(np.cov(R, rowvar = False).T, R_bar - np.ones(N) * mu).T @ (R_bar - np.ones(N) * mu)[:, None])[:, None]).T, (np.linalg.solve(np.cov(R, rowvar = False).T, PE_gls[:, None]).T @ PE_gls[:, None]).T).T[:,0])
    PEi_gls = R_bar - beta @ Lambda_gls - const_gls
    R2i_gls = np.squeeze(1 - np.linalg.solve(((np.linalg.solve(np.cov(R, rowvar = False).T, R_bar - np.ones(N) * mu).T @ (R_bar - np.ones(N) * mu)[:, None])[:, None]).T, (np.linalg.solve(np.cov(R, rowvar = False).T, PEi_gls[:, None]).T @ PEi_gls[:, None]).T).T[:,0])                   
    
    # Prepare output
    
    ans = dict([
        
            ('beta', beta),
            ('alpha', alpha),
            ('Lambda', Lambda),
            ('Lambda_gls', Lambda_gls),
            ('const', const),
            ('const_gls', const_gls),
            ('PE', PE),
            ('PEi', PEi),
            ('PE_gls', PE_gls),
            ('PEi_gls', PEi_gls),
            ('R2', R2),
            ('R2i', R2i),
            ('R2_gls', R2_gls),
            ('R2i_gls', R2i_gls),
         
        ])
    
    
    if len(Lambda0) == 0 and np.isin(math.floor(method), [1,2,4,6]) and traded_f == 0:
        Lambda0 = Lambda[:,None]
    elif len(Lambda0) == 0 and np.isin(math.floor(method), [3,5]) and traded_f == 0:
        Lambda0 = Theta[:,None]
    elif len(Lambda0) == 0 and np.isin(math.floor(method), [1,2,4,6]) and traded_f == 1:
        Lambda0 = np.mean(f, axis=0)[:,None]
    elif len(Lambda0) == 0 and np.isin(math.floor(method), [3,5]) and traded_f == 1:
        Lambda0 = np.concatenate((np.zeros(1), np.mean(f, axis=0)), axis=0)[:,None]
    
    
    
    #------------------------------------------------------------------------------
    # SECTION 1) PRELIMINARY TESTS
    #------------------------------------------------------------------------------
    #@ToDo Calculation of cA and if- Blocks in 2.09/3.09 (error in MATLAB)
    #@ToDo Dimension Error in 6.03/6.04 (error in MATLAB)
    
    
    #------------------------------------------------------------------------------
    # Method 1.01) Test if betas are all equal to zero 
    #               Assume residuals are iid
    #------------------------------------------------------------------------------
    
    if method == 1.01: 
        
        AV = np.kron(S, FFi)  # Covariance matrix of intercepts and betas
        Test = np.full((K,1), np.nan)
        pval = np.full((K,1), np.nan)
        b_vec = np.reshape(B.T, (1+K)*N) # Reshape the factor betas
        
        for k in range(0,K):
            idx = range(k+1, N*(K+1), K+1)
            b = b_vec[idx]
            Test[k,:] = np.linalg.solve(AV[np.ix_(idx, idx)].T, b[:, None]).T @ b
            pval[k,:] = cdfchic.cdfchic_func(Test[k,:], N)
                
        ans['Test'] = Test
        ans['pval'] = pval
    
        
    #------------------------------------------------------------------------------    
    # Method 1.02)  Test if betas are all equal to zero                
    #               Estimate covariance using Newey West
    #------------------------------------------------------------------------------    
        
    if method == 1.02:    
        for i in range(0, np.size(e, axis=1)):
            if i == 0:
                g = np.matlib.repmat(e[:, i], K+1, 1).T * F 
            else:   
                g = np.append(g, np.matlib.repmat(e[:, i], K+1, 1).T * F, axis = 1) 
        
        S_hac = nw.nw_func(g,lags) # Call Newey-West function
        Di = np.kron(np.eye(N), FFi)
        AV_hac = Di @ S_hac @ Di * T # Covariance matrix of intercepts and betas
        Test = np.full((K,1), np.nan)
        pval = np.full((K,1), np.nan)
        b_vec = np.reshape(B.T, (1+K)*N) # Reshape the factors beta
        
        for k in range(0,K):
            idx = range(k+1, N*(K+1), K+1)
            b = b_vec[idx]
            Test[k,:] = np.linalg.solve(AV_hac[np.ix_(idx, idx)].T, b[:, None]).T @ b
            pval[k,:] = cdfchic.cdfchic_func(Test[k,:], N)
                
        ans['Test'] = Test
        ans['pval'] = pval
        
        
    #------------------------------------------------------------------------------
    # Method 1.03)  Test if betas are all equal to zero 
    #               Estimate covariance matrix using varhac
    #------------------------------------------------------------------------------
                      
    if method == 1.03:
        for i in range(0, np.size(e, axis=1)):
            if i == 0:
                g = np.matlib.repmat(e[:, i], K+1, 1).T * F 
            else:   
                g = np.append(g, np.matlib.repmat(e[:, i], K+1, 1).T * F, axis = 1)
                
        S_hac = hac_var.hac_var_func(g,1,0) # Call VARHAC function from Burnside
        Di = np.kron(np.eye(N), FFi)
        AV_hac = Di @ S_hac @ Di * T # Covariance matrix of intercepts and betas
        Test = np.full((K,1), np.nan)
        pval = np.full((K,1), np.nan)
        b_vec = np.reshape(B.T, (1+K)*N) # Reshape the factors beta
        
        for k in range(0,K):
            idx = range(k+1, N*(K+1), K+1)
            b = b_vec[idx]
            Test[k,:] = np.linalg.solve(AV_hac[np.ix_(idx, idx)].T, b[:, None]).T @ b
            pval[k,:] = cdfchic.cdfchic_func(Test[k,:], N)
                
        ans['Test'] = Test
        ans['pval'] = pval
    
    #------------------------------------------------------------------------------
    # Method 1.04) Test if betas are all the same               
    #              Assume residuals are iid 
    #------------------------------------------------------------------------------
    
    if method == 1.04:
        AV = np.kron(S, FFi)
        Test = np.full((K,1), np.nan)
        pval = np.full((K,1), np.nan)
        b_vec = np.reshape(B.T, (1+K)*N) # Reshape the factors beta
        RR = np.concatenate((np.ones((N-1,1)), -np.eye(N-1)), axis = 1)
        
        for k in range(0,K):
            idx = range(k+1, N*(K+1), K+1)
            b = b_vec[idx]
            RRb = RR @ b
            Test[k,:] = np.linalg.solve((RR @ AV[np.ix_(idx, idx)] @ RR.T).T, RRb).T @ RRb
            pval[k,:] = cdfchic.cdfchic_func(Test[k,:], N-1)
                
        ans['Test'] = Test
        ans['pval'] = pval
        
    #------------------------------------------------------------------------------
    # Method 1.05) Test if betas are all the same
    #              Estimate covariance using Newey West
    #------------------------------------------------------------------------------    
    
    if method == 1.05:
        for i in range(0, np.size(e, axis=1)):
            if i == 0:
                g = np.matlib.repmat(e[:, i], K+1, 1).T * F 
            else:   
                g = np.append(g, np.matlib.repmat(e[:, i], K+1, 1).T * F, axis = 1)
                
        S_hac = nw.nw_func(g, lags) # Call neweywest function from Burnside
        Di = np.kron(np.eye(N), FFi)
        AV_hac = Di @ S_hac @ Di * T # Covariance matrix of intercepts and betas
        Test = np.full((K,1), np.nan)
        pval = np.full((K,1), np.nan)
        b_vec = np.reshape(B.T, (1+K)*N) # Reshape the factors beta
        RR = np.concatenate((np.ones((N-1,1)), -np.eye(N-1)), axis = 1)
        
        for k in range(0,K):
            idx = range(k+1, N*(K+1), K+1)
            b = b_vec[idx]
            RRb = RR @ b
            Test[k,:] = np.linalg.solve((RR @ AV_hac[np.ix_(idx, idx)] @ RR.T).T, RRb).T @ RRb
            pval[k,:] = cdfchic.cdfchic_func(Test[k,:], N-1)
                
        ans['Test'] = Test
        ans['pval'] = pval
    
    #------------------------------------------------------------------------------
    # Method 1.06) Test if betas are all the same
    #              Estimate covariance using varhac
    #------------------------------------------------------------------------------
    
    if method == 1.06:
        for i in range(0, np.size(e, axis=1)):
            if i == 0:
                g = np.matlib.repmat(e[:, i], K+1, 1).T * F 
            else:   
                g = np.append(g, np.matlib.repmat(e[:, i], K+1, 1).T * F, axis = 1)
                
        S_hac = hac_var.hac_var_func(g, 1, 0) # Call VARHAC function from Burnside
        Di = np.kron(np.eye(N), FFi)
        AV_hac = Di @ S_hac @ Di * T # Covariance matrix of intercepts and betas
        Test = np.full((K,1), np.nan)
        pval = np.full((K,1), np.nan)
        b_vec = np.reshape(B.T, (1+K)*N) # Reshape the factors beta
        RR = np.concatenate((np.ones((N-1,1)), -np.eye(N-1)), axis = 1)
        
        for k in range(0,K):
            idx = range(k+1, N*(K+1), K+1)
            b = b_vec[idx]
            RRb = RR @ b
            Test[k,:] = np.linalg.solve((RR @ AV_hac[np.ix_(idx, idx)] @ RR.T).T, RRb).T @ RRb
            pval[k,:] = cdfchic.cdfchic_func(Test[k,:], N-1)
                
        ans['Test'] = Test
        ans['pval'] = pval
        
    #------------------------------------------------------------------------------
    # SECTION 2)    SIGNIFICANCE TESTS FOR LAMBDA (MARKET PRICE OF RISK) IN A
    #               CROSS-SECTIONAL REGRESSION WITHOUT INTERCEPT
    #------------------------------------------------------------------------------
    
    #------------------------------------------------------------------------------
    # Method 2.01) Fama/MacBeth standard errors
    #------------------------------------------------------------------------------
    

    return ans

import numpy as np
import numpy.matlib as npmatlib
import math
import scipy.stats as sps
import scipy.linalg as spl
import scipy as spy

# Import subfunctions
import cdfchic
import nw
import hac_var
import block_bootstrap
import FMB_coefficients
import linchi2


def omnibus_section1(R, f, method, Lambda0 = 0, level = 0.05):
    
    #------------------------------------------------------------------------------
    # SECTION 0) GENERAL 
    #------------------------------------------------------------------------------
    
    lags = 3       # set to -1 to select Andrews optimal bandwith
    excess_ret = 0  # set to 0 if gross returns are used (relevant for Kleibergen et al statistics)
    traded_f = 1    # set to 1 if the factor is traded
    
    [T,K] = f.shape
    N = np.size(R, axis=1)
    
    if Lambda0 == 0 and np.isin(math.floor(method), [1,2,4,6]):
        Lambda0  = np.zeros((K,1))
    
    elif Lambda0 == 0 and np.isin(math.floor(method), [3,5]):
        Lambda0  = np.zeros((K+1,1))             
    
    # Time Series Regression to determine Beta (Cochrane, 2005, p.230, ff)
    F = np.array(np.concatenate((np.ones((T,1),float),f), axis=1))
    FFi = np.linalg.solve(F.T @ F, np.identity(K+1))             
    B = np.matmul(np.matmul(FFi, F.T), R)
    alpha = np.array(B[0]).T                                   # Alphas
    beta = np.array(B[1:K+1]).T                                # Betas
    e = R - F @ B                                              # Residuals
    SSR = e.T @ e                                              # Sum of squared residuals
    R_bar = np.mean(R, axis=0).T                               # Average returns  
    SST = (R - npmatlib.repmat(R_bar.T,T,1)).T @ (R - npmatlib.repmat(R_bar.T,T,1))  # Sum of squared demeaned returns      
    S = (1/T) * SSR                                            # Covariance matrix of residuals under i.i.d.-assumption
    
    # Cross-sectional regression to determine Lambda (Cochrane, 2005, p.235 ff & Burnside, 2011, AER)
    
    if np.isin(math.floor(method), [1,2,4]): # cross-sectional regression without intercept
    
         A = np.linalg.solve((beta.T @ beta),beta.T)
         Lambda = A @ R_bar                                     # OLS cross-sectional 
         Lambda_gls = np.linalg.solve(np.linalg.solve(np.cov(R, rowvar = False).T, beta).T @ beta, np.linalg.solve(np.cov(R, rowvar = False).T, beta).T @ R_bar) # GLS cross-sectional regression
         const = np.nan
         const_gls = np.nan
          
    else : # cross-sectional regression with intercept (constant IS NOT part of the pricing errors)
    
         X = np.concatenate((np.ones((N,1), float), beta), axis = 1)
         A = np.linalg.solve(X.T @ X,X.T)
         Theta = A @ R_bar
         Lambda = np.array(Theta[1:K+1])                       # OLS cross-sectional
         Theta_gls = np.linalg.solve(np.linalg.solve(np.cov(R, rowvar = False).T, X).T @ X, np.linalg.solve(np.cov(R, rowvar = False).T,X).T @ R_bar) 
         Lambda_gls = np.array(Theta_gls[1:K+1])               # GLS cross-sectional regression
         const = Theta [0]
         const_gls = Theta_gls [0]
    
    # OLS and GLS pricing errors and cross-sectional R^2
    
    PE = R_bar - beta @ Lambda                                                      # pricing errors
    R2 = 1- PE.T @ PE / ((R_bar - np.mean(R_bar)).T @ (R_bar - np.mean(R_bar)))     # cross-sectional R^2
    PEi = R_bar - beta @ Lambda - const                                             # pricing errors without imposing zero intercept
    R2i = 1- PEi.T @ PEi / ((R_bar - np.mean(R_bar)).T @ (R_bar - np.mean(R_bar)))  # cross-sectional R^2 without imposing zero intercept
     
    PE_gls = R_bar - beta @ Lambda_gls
    mu = np.linalg.solve(np.linalg.solve(np.cov(R, rowvar = False).T, np.ones((N,1), float)).T @ np.ones((N,1), float), np.linalg.solve(np.cov(R, rowvar = False).T, R_bar[:, None]).T @ np.ones((N,1), float))[:,0]
    R2_gls = np.squeeze(1 - np.linalg.solve(((np.linalg.solve(np.cov(R, rowvar = False).T, R_bar - np.ones(N) * mu).T @ (R_bar - np.ones(N) * mu)[:, None])[:, None]).T, (np.linalg.solve(np.cov(R, rowvar = False).T, PE_gls[:, None]).T @ PE_gls[:, None]).T).T[:,0])
    PEi_gls = R_bar - beta @ Lambda_gls - const_gls
    R2i_gls = np.squeeze(1 - np.linalg.solve(((np.linalg.solve(np.cov(R, rowvar = False).T, R_bar - np.ones(N) * mu).T @ (R_bar - np.ones(N) * mu)[:, None])[:, None]).T, (np.linalg.solve(np.cov(R, rowvar = False).T, PEi_gls[:, None]).T @ PEi_gls[:, None]).T).T[:,0])                   
    
    # Prepare output
    
    ans = dict([
        
            ('beta', beta),
            ('alpha', alpha),
            ('Lambda', Lambda),
            ('Lambda_gls', Lambda_gls),
            ('const', const),
            ('const_gls', const_gls),
            ('PE', PE),
            ('PEi', PEi),
            ('PE_gls', PE_gls),
            ('PEi_gls', PEi_gls),
            ('R2', R2),
            ('R2i', R2i),
            ('R2_gls', R2_gls),
            ('R2i_gls', R2i_gls),
         
        ])
    
    
    if len(Lambda0) == 0 and np.isin(math.floor(method), [1,2,4,6]) and traded_f == 0:
        Lambda0 = Lambda[:,None]
    elif len(Lambda0) == 0 and np.isin(math.floor(method), [3,5]) and traded_f == 0:
        Lambda0 = Theta[:,None]
    elif len(Lambda0) == 0 and np.isin(math.floor(method), [1,2,4,6]) and traded_f == 1:
        Lambda0 = np.mean(f, axis=0)[:,None]
    elif len(Lambda0) == 0 and np.isin(math.floor(method), [3,5]) and traded_f == 1:
        Lambda0 = np.concatenate((np.zeros(1), np.mean(f, axis=0)), axis=0)[:,None]
    
    
    
    #------------------------------------------------------------------------------
    # SECTION 1) PRELIMINARY TESTS
    #------------------------------------------------------------------------------
    #@ToDo Calculation of cA and if- Blocks in 2.09/3.09 (error in MATLAB)
    #@ToDo Dimension Error in 6.03/6.04 (error in MATLAB)
    
    
    #------------------------------------------------------------------------------
    # Method 1.01) Test if betas are all equal to zero 
    #               Assume residuals are iid
    #------------------------------------------------------------------------------
    
    if method == 1.01: 
        
        AV = np.kron(S, FFi)  # Covariance matrix of intercepts and betas
        Test = np.full((K,1), np.nan)
        pval = np.full((K,1), np.nan)
        b_vec = np.reshape(B.T, (1+K)*N) # Reshape the factor betas
        
        for k in range(0,K):
            idx = range(k+1, N*(K+1), K+1)
            b = b_vec[idx]
            Test[k,:] = np.linalg.solve(AV[np.ix_(idx, idx)].T, b[:, None]).T @ b
            pval[k,:] = cdfchic.cdfchic_func(Test[k,:], N)
                
        ans['Test'] = Test
        ans['pval'] = pval
    
        
    #------------------------------------------------------------------------------    
    # Method 1.02)  Test if betas are all equal to zero                
    #               Estimate covariance using Newey West
    #------------------------------------------------------------------------------    
        
    if method == 1.02:    
        for i in range(0, np.size(e, axis=1)):
            if i == 0:
                g = np.matlib.repmat(e[:, i], K+1, 1).T * F 
            else:   
                g = np.append(g, np.matlib.repmat(e[:, i], K+1, 1).T * F, axis = 1) 
        
        S_hac = nw.nw_func(g,lags) # Call Newey-West function
        Di = np.kron(np.eye(N), FFi)
        AV_hac = Di @ S_hac @ Di * T # Covariance matrix of intercepts and betas
        Test = np.full((K,1), np.nan)
        pval = np.full((K,1), np.nan)
        b_vec = np.reshape(B.T, (1+K)*N) # Reshape the factors beta
        
        for k in range(0,K):
            idx = range(k+1, N*(K+1), K+1)
            b = b_vec[idx]
            Test[k,:] = np.linalg.solve(AV_hac[np.ix_(idx, idx)].T, b[:, None]).T @ b
            pval[k,:] = cdfchic.cdfchic_func(Test[k,:], N)
                
        ans['Test'] = Test
        ans['pval'] = pval
        
        
    #------------------------------------------------------------------------------
    # Method 1.03)  Test if betas are all equal to zero 
    #               Estimate covariance matrix using varhac
    #------------------------------------------------------------------------------
                      
    if method == 1.03:
        for i in range(0, np.size(e, axis=1)):
            if i == 0:
                g = np.matlib.repmat(e[:, i], K+1, 1).T * F 
            else:   
                g = np.append(g, np.matlib.repmat(e[:, i], K+1, 1).T * F, axis = 1)
                
        S_hac = hac_var.hac_var_func(g,1,0) # Call VARHAC function from Burnside
        Di = np.kron(np.eye(N), FFi)
        AV_hac = Di @ S_hac @ Di * T # Covariance matrix of intercepts and betas
        Test = np.full((K,1), np.nan)
        pval = np.full((K,1), np.nan)
        b_vec = np.reshape(B.T, (1+K)*N) # Reshape the factors beta
        
        for k in range(0,K):
            idx = range(k+1, N*(K+1), K+1)
            b = b_vec[idx]
            Test[k,:] = np.linalg.solve(AV_hac[np.ix_(idx, idx)].T, b[:, None]).T @ b
            pval[k,:] = cdfchic.cdfchic_func(Test[k,:], N)
                
        ans['Test'] = Test
        ans['pval'] = pval
    
    #------------------------------------------------------------------------------
    # Method 1.04) Test if betas are all the same               
    #              Assume residuals are iid 
    #------------------------------------------------------------------------------
    
    if method == 1.04:
        AV = np.kron(S, FFi)
        Test = np.full((K,1), np.nan)
        pval = np.full((K,1), np.nan)
        b_vec = np.reshape(B.T, (1+K)*N) # Reshape the factors beta
        RR = np.concatenate((np.ones((N-1,1)), -np.eye(N-1)), axis = 1)
        
        for k in range(0,K):
            idx = range(k+1, N*(K+1), K+1)
            b = b_vec[idx]
            RRb = RR @ b
            Test[k,:] = np.linalg.solve((RR @ AV[np.ix_(idx, idx)] @ RR.T).T, RRb).T @ RRb
            pval[k,:] = cdfchic.cdfchic_func(Test[k,:], N-1)
                
        ans['Test'] = Test
        ans['pval'] = pval
        
    #------------------------------------------------------------------------------
    # Method 1.05) Test if betas are all the same
    #              Estimate covariance using Newey West
    #------------------------------------------------------------------------------    
    
    if method == 1.05:
        for i in range(0, np.size(e, axis=1)):
            if i == 0:
                g = np.matlib.repmat(e[:, i], K+1, 1).T * F 
            else:   
                g = np.append(g, np.matlib.repmat(e[:, i], K+1, 1).T * F, axis = 1)
                
        S_hac = nw.nw_func(g, lags) # Call neweywest function from Burnside
        Di = np.kron(np.eye(N), FFi)
        AV_hac = Di @ S_hac @ Di * T # Covariance matrix of intercepts and betas
        Test = np.full((K,1), np.nan)
        pval = np.full((K,1), np.nan)
        b_vec = np.reshape(B.T, (1+K)*N) # Reshape the factors beta
        RR = np.concatenate((np.ones((N-1,1)), -np.eye(N-1)), axis = 1)
        
        for k in range(0,K):
            idx = range(k+1, N*(K+1), K+1)
            b = b_vec[idx]
            RRb = RR @ b
            Test[k,:] = np.linalg.solve((RR @ AV_hac[np.ix_(idx, idx)] @ RR.T).T, RRb).T @ RRb
            pval[k,:] = cdfchic.cdfchic_func(Test[k,:], N-1)
                
        ans['Test'] = Test
        ans['pval'] = pval
    
    #------------------------------------------------------------------------------
    # Method 1.06) Test if betas are all the same
    #              Estimate covariance using varhac
    #------------------------------------------------------------------------------
    
    if method == 1.06:
        for i in range(0, np.size(e, axis=1)):
            if i == 0:
                g = np.matlib.repmat(e[:, i], K+1, 1).T * F 
            else:   
                g = np.append(g, np.matlib.repmat(e[:, i], K+1, 1).T * F, axis = 1)
                
        S_hac = hac_var.hac_var_func(g, 1, 0) # Call VARHAC function from Burnside
        Di = np.kron(np.eye(N), FFi)
        AV_hac = Di @ S_hac @ Di * T # Covariance matrix of intercepts and betas
        Test = np.full((K,1), np.nan)
        pval = np.full((K,1), np.nan)
        b_vec = np.reshape(B.T, (1+K)*N) # Reshape the factors beta
        RR = np.concatenate((np.ones((N-1,1)), -np.eye(N-1)), axis = 1)
        
        for k in range(0,K):
            idx = range(k+1, N*(K+1), K+1)
            b = b_vec[idx]
            RRb = RR @ b
            Test[k,:] = np.linalg.solve((RR @ AV_hac[np.ix_(idx, idx)] @ RR.T).T, RRb).T @ RRb
            pval[k,:] = cdfchic.cdfchic_func(Test[k,:], N-1)
                
        ans['Test'] = Test

    return ans

import numpy as np
import numpy.matlib as npmatlib
import math
import scipy.stats as sps
import scipy.linalg as spl
import scipy as spy

# Import subfunctions
import cdfchic
import nw
import hac_var
import block_bootstrap
import FMB_coefficients
import linchi2


def omnibus_section1(R, f, method, Lambda0 = 0, level = 0.05):
    
    #------------------------------------------------------------------------------
    # SECTION 0) GENERAL 
    #------------------------------------------------------------------------------
    
    lags = 3       # set to -1 to select Andrews optimal bandwith
    excess_ret = 0  # set to 0 if gross returns are used (relevant for Kleibergen et al statistics)
    traded_f = 1    # set to 1 if the factor is traded
    
    [T,K] = f.shape
    N = np.size(R, axis=1)
    
    if Lambda0 == 0 and np.isin(math.floor(method), [1,2,4,6]):
        Lambda0  = np.zeros((K,1))
    
    elif Lambda0 == 0 and np.isin(math.floor(method), [3,5]):
        Lambda0  = np.zeros((K+1,1))             
    
    # Time Series Regression to determine Beta (Cochrane, 2005, p.230, ff)
    F = np.array(np.concatenate((np.ones((T,1),float),f), axis=1))
    FFi = np.linalg.solve(F.T @ F, np.identity(K+1))             
    B = np.matmul(np.matmul(FFi, F.T), R)
    alpha = np.array(B[0]).T                                   # Alphas
    beta = np.array(B[1:K+1]).T                                # Betas
    e = R - F @ B                                              # Residuals
    SSR = e.T @ e                                              # Sum of squared residuals
    R_bar = np.mean(R, axis=0).T                               # Average returns  
    SST = (R - npmatlib.repmat(R_bar.T,T,1)).T @ (R - npmatlib.repmat(R_bar.T,T,1))  # Sum of squared demeaned returns      
    S = (1/T) * SSR                                            # Covariance matrix of residuals under i.i.d.-assumption
    
    # Cross-sectional regression to determine Lambda (Cochrane, 2005, p.235 ff & Burnside, 2011, AER)
    
    if np.isin(math.floor(method), [1,2,4]): # cross-sectional regression without intercept
    
         A = np.linalg.solve((beta.T @ beta),beta.T)
         Lambda = A @ R_bar                                     # OLS cross-sectional 
         Lambda_gls = np.linalg.solve(np.linalg.solve(np.cov(R, rowvar = False).T, beta).T @ beta, np.linalg.solve(np.cov(R, rowvar = False).T, beta).T @ R_bar) # GLS cross-sectional regression
         const = np.nan
         const_gls = np.nan
          
    else : # cross-sectional regression with intercept (constant IS NOT part of the pricing errors)
    
         X = np.concatenate((np.ones((N,1), float), beta), axis = 1)
         A = np.linalg.solve(X.T @ X,X.T)
         Theta = A @ R_bar
         Lambda = np.array(Theta[1:K+1])                       # OLS cross-sectional
         Theta_gls = np.linalg.solve(np.linalg.solve(np.cov(R, rowvar = False).T, X).T @ X, np.linalg.solve(np.cov(R, rowvar = False).T,X).T @ R_bar) 
         Lambda_gls = np.array(Theta_gls[1:K+1])               # GLS cross-sectional regression
         const = Theta [0]
         const_gls = Theta_gls [0]
    
    # OLS and GLS pricing errors and cross-sectional R^2
    
    PE = R_bar - beta @ Lambda                                                      # pricing errors
    R2 = 1- PE.T @ PE / ((R_bar - np.mean(R_bar)).T @ (R_bar - np.mean(R_bar)))     # cross-sectional R^2
    PEi = R_bar - beta @ Lambda - const                                             # pricing errors without imposing zero intercept
    R2i = 1- PEi.T @ PEi / ((R_bar - np.mean(R_bar)).T @ (R_bar - np.mean(R_bar)))  # cross-sectional R^2 without imposing zero intercept
     
    PE_gls = R_bar - beta @ Lambda_gls
    mu = np.linalg.solve(np.linalg.solve(np.cov(R, rowvar = False).T, np.ones((N,1), float)).T @ np.ones((N,1), float), np.linalg.solve(np.cov(R, rowvar = False).T, R_bar[:, None]).T @ np.ones((N,1), float))[:,0]
    R2_gls = np.squeeze(1 - np.linalg.solve(((np.linalg.solve(np.cov(R, rowvar = False).T, R_bar - np.ones(N) * mu).T @ (R_bar - np.ones(N) * mu)[:, None])[:, None]).T, (np.linalg.solve(np.cov(R, rowvar = False).T, PE_gls[:, None]).T @ PE_gls[:, None]).T).T[:,0])
    PEi_gls = R_bar - beta @ Lambda_gls - const_gls
    R2i_gls = np.squeeze(1 - np.linalg.solve(((np.linalg.solve(np.cov(R, rowvar = False).T, R_bar - np.ones(N) * mu).T @ (R_bar - np.ones(N) * mu)[:, None])[:, None]).T, (np.linalg.solve(np.cov(R, rowvar = False).T, PEi_gls[:, None]).T @ PEi_gls[:, None]).T).T[:,0])                   
    
    # Prepare output
    
    ans = dict([
        
            ('beta', beta),
            ('alpha', alpha),
            ('Lambda', Lambda),
            ('Lambda_gls', Lambda_gls),
            ('const', const),
            ('const_gls', const_gls),
            ('PE', PE),
            ('PEi', PEi),
            ('PE_gls', PE_gls),
            ('PEi_gls', PEi_gls),
            ('R2', R2),
            ('R2i', R2i),
            ('R2_gls', R2_gls),
            ('R2i_gls', R2i_gls),
         
        ])
    
    
    if len(Lambda0) == 0 and np.isin(math.floor(method), [1,2,4,6]) and traded_f == 0:
        Lambda0 = Lambda[:,None]
    elif len(Lambda0) == 0 and np.isin(math.floor(method), [3,5]) and traded_f == 0:
        Lambda0 = Theta[:,None]
    elif len(Lambda0) == 0 and np.isin(math.floor(method), [1,2,4,6]) and traded_f == 1:
        Lambda0 = np.mean(f, axis=0)[:,None]
    elif len(Lambda0) == 0 and np.isin(math.floor(method), [3,5]) and traded_f == 1:
        Lambda0 = np.concatenate((np.zeros(1), np.mean(f, axis=0)), axis=0)[:,None]
    
    
    
    #------------------------------------------------------------------------------
    # SECTION 1) PRELIMINARY TESTS
    #------------------------------------------------------------------------------
    #@ToDo Calculation of cA and if- Blocks in 2.09/3.09 (error in MATLAB)
    #@ToDo Dimension Error in 6.03/6.04 (error in MATLAB)
    
    
    #------------------------------------------------------------------------------
    # Method 1.01) Test if betas are all equal to zero 
    #               Assume residuals are iid
    #------------------------------------------------------------------------------
    
    if method == 1.01: 
        
        AV = np.kron(S, FFi)  # Covariance matrix of intercepts and betas
        Test = np.full((K,1), np.nan)
        pval = np.full((K,1), np.nan)
        b_vec = np.reshape(B.T, (1+K)*N) # Reshape the factor betas
        
        for k in range(0,K):
            idx = range(k+1, N*(K+1), K+1)
            b = b_vec[idx]
            Test[k,:] = np.linalg.solve(AV[np.ix_(idx, idx)].T, b[:, None]).T @ b
            pval[k,:] = cdfchic.cdfchic_func(Test[k,:], N)
                
        ans['Test'] = Test
        ans['pval'] = pval
    
        
    #------------------------------------------------------------------------------    
    # Method 1.02)  Test if betas are all equal to zero                
    #               Estimate covariance using Newey West
    #------------------------------------------------------------------------------    
        
    if method == 1.02:    
        for i in range(0, np.size(e, axis=1)):
            if i == 0:
                g = np.matlib.repmat(e[:, i], K+1, 1).T * F 
            else:   
                g = np.append(g, np.matlib.repmat(e[:, i], K+1, 1).T * F, axis = 1) 
        
        S_hac = nw.nw_func(g,lags) # Call Newey-West function
        Di = np.kron(np.eye(N), FFi)
        AV_hac = Di @ S_hac @ Di * T # Covariance matrix of intercepts and betas
        Test = np.full((K,1), np.nan)
        pval = np.full((K,1), np.nan)
        b_vec = np.reshape(B.T, (1+K)*N) # Reshape the factors beta
        
        for k in range(0,K):
            idx = range(k+1, N*(K+1), K+1)
            b = b_vec[idx]
            Test[k,:] = np.linalg.solve(AV_hac[np.ix_(idx, idx)].T, b[:, None]).T @ b
            pval[k,:] = cdfchic.cdfchic_func(Test[k,:], N)
                
        ans['Test'] = Test
        ans['pval'] = pval
        
        
    #------------------------------------------------------------------------------
    # Method 1.03)  Test if betas are all equal to zero 
    #               Estimate covariance matrix using varhac
    #------------------------------------------------------------------------------
                      
    if method == 1.03:
        for i in range(0, np.size(e, axis=1)):
            if i == 0:
                g = np.matlib.repmat(e[:, i], K+1, 1).T * F 
            else:   
                g = np.append(g, np.matlib.repmat(e[:, i], K+1, 1).T * F, axis = 1)
                
        S_hac = hac_var.hac_var_func(g,1,0) # Call VARHAC function from Burnside
        Di = np.kron(np.eye(N), FFi)
        AV_hac = Di @ S_hac @ Di * T # Covariance matrix of intercepts and betas
        Test = np.full((K,1), np.nan)
        pval = np.full((K,1), np.nan)
        b_vec = np.reshape(B.T, (1+K)*N) # Reshape the factors beta
        
        for k in range(0,K):
            idx = range(k+1, N*(K+1), K+1)
            b = b_vec[idx]
            Test[k,:] = np.linalg.solve(AV_hac[np.ix_(idx, idx)].T, b[:, None]).T @ b
            pval[k,:] = cdfchic.cdfchic_func(Test[k,:], N)
                
        ans['Test'] = Test
        ans['pval'] = pval
    
    #------------------------------------------------------------------------------
    # Method 1.04) Test if betas are all the same               
    #              Assume residuals are iid 
    #------------------------------------------------------------------------------
    
    if method == 1.04:
        AV = np.kron(S, FFi)
        Test = np.full((K,1), np.nan)
        pval = np.full((K,1), np.nan)
        b_vec = np.reshape(B.T, (1+K)*N) # Reshape the factors beta
        RR = np.concatenate((np.ones((N-1,1)), -np.eye(N-1)), axis = 1)
        
        for k in range(0,K):
            idx = range(k+1, N*(K+1), K+1)
            b = b_vec[idx]
            RRb = RR @ b
            Test[k,:] = np.linalg.solve((RR @ AV[np.ix_(idx, idx)] @ RR.T).T, RRb).T @ RRb
            pval[k,:] = cdfchic.cdfchic_func(Test[k,:], N-1)
                
        ans['Test'] = Test
        ans['pval'] = pval
        
    #------------------------------------------------------------------------------
    # Method 1.05) Test if betas are all the same
    #              Estimate covariance using Newey West
    #------------------------------------------------------------------------------    
    
    if method == 1.05:
        for i in range(0, np.size(e, axis=1)):
            if i == 0:
                g = np.matlib.repmat(e[:, i], K+1, 1).T * F 
            else:   
                g = np.append(g, np.matlib.repmat(e[:, i], K+1, 1).T * F, axis = 1)
                
        S_hac = nw.nw_func(g, lags) # Call neweywest function from Burnside
        Di = np.kron(np.eye(N), FFi)
        AV_hac = Di @ S_hac @ Di * T # Covariance matrix of intercepts and betas
        Test = np.full((K,1), np.nan)
        pval = np.full((K,1), np.nan)
        b_vec = np.reshape(B.T, (1+K)*N) # Reshape the factors beta
        RR = np.concatenate((np.ones((N-1,1)), -np.eye(N-1)), axis = 1)
        
        for k in range(0,K):
            idx = range(k+1, N*(K+1), K+1)
            b = b_vec[idx]
            RRb = RR @ b
            Test[k,:] = np.linalg.solve((RR @ AV_hac[np.ix_(idx, idx)] @ RR.T).T, RRb).T @ RRb
            pval[k,:] = cdfchic.cdfchic_func(Test[k,:], N-1)
                
        ans['Test'] = Test
        ans['pval'] = pval
    
    #------------------------------------------------------------------------------
    # Method 1.06) Test if betas are all the same
    #              Estimate covariance using varhac
    #------------------------------------------------------------------------------
    
    if method == 1.06:
        for i in range(0, np.size(e, axis=1)):
            if i == 0:
                g = np.matlib.repmat(e[:, i], K+1, 1).T * F 
            else:   
                g = np.append(g, np.matlib.repmat(e[:, i], K+1, 1).T * F, axis = 1)
                
        S_hac = hac_var.hac_var_func(g, 1, 0) # Call VARHAC function from Burnside
        Di = np.kron(np.eye(N), FFi)
        AV_hac = Di @ S_hac @ Di * T # Covariance matrix of intercepts and betas
        Test = np.full((K,1), np.nan)
        pval = np.full((K,1), np.nan)
        b_vec = np.reshape(B.T, (1+K)*N) # Reshape the factors beta
        RR = np.concatenate((np.ones((N-1,1)), -np.eye(N-1)), axis = 1)
        
        for k in range(0,K):
            idx = range(k+1, N*(K+1), K+1)
            b = b_vec[idx]
            RRb = RR @ b
            Test[k,:] = np.linalg.solve((RR @ AV_hac[np.ix_(idx, idx)] @ RR.T).T, RRb).T @ RRb
            pval[k,:] = cdfchic.cdfchic_func(Test[k,:], N-1)
                
        ans['Test'] = Test
        ans['pval'] = pval

    return ans
