# This program performs different tests of linear factor models for the cross-section of expected returns.
# 
# INPUTS:
#     
# R           matrix with returns or excess returns ($TxN$)\
# f           matrix with factors ($TxK$)\
# method      one of the methods described below
# 

# This program performs different tests of linear factor models for the cross-section of expected returns.
# 
# INPUTS:
#     
# R           matrix with returns or excess returns (TxN)
# f           matrix with factors (TxK)
# method      one of the methods described below
# 
# OPTIONAL INPUT:
# 
# Lambda0     - Hypothetical market price of risk vector (must be Kx1 vector 
#               in regression without intercept and (K+1)x1 with intercept)
#             - when no Lambda0 is entered by the user, Lambda0 is by default
#               set to a vector of zeros
#             - alternatively enter []. In this case Lambda0 is set to the
#               estimated Lambda (from FMB regression) if the factor is not
#               traded and to the factor mean if it is traded.
#               
# level       - Some methods (2.09, 2.10, 3.09, and 3.10) require the level
#               of the test. If not provided by the user, it is set to 0.05.
# 
# OPTIONS:
# These can be specified below the "function"-command.
# lags        - number of Newey-West lags. Enter -1 to select Andrews optimal 
#               number of lags
# excess_ret  - set to 0 if gross returns are used (relevant for Kleibergen 
#               et al statistics) and 1 if excess returns are used
# traded_f    - set to 1 if the factor is traded and 0 otherwise
# 
# Methods:
# SECTION 1)  PRELIMINARY TESTS
# 1.01    Test if betas are all equal to zero, assume residuals are iid
# 1.02    Test if betas are all equal to zero, estimate covariance using Newey West
# 1.03    Test if betas are all equal to zero, estimate covariance matrix using varhac
# 1.04    Test if betas are all the same, assume residuals are iid
# 1.05    Test if betas are all the same, estimate covariance using Newey West
# 1.06    Test if betas are all the same, estimate covariance matrix using varhac
# 
# SECTION 2)  SIGNIFICANCE TESTS FOR LAMBDA (MARKET PRICE OF RISK) IN A
#             CROSS-SECTIONAL REGRESSION WITHOUT INTERCEPT
# 2.01    Fama/MacBeth standard errors
# 2.02    OLS standard errors (no Shanken correction)
# 2.03    Shanken-corrected standard errors
# 2.04    GMM standard errors, Newey-West
# 2.05    GMM standard errors, VARHAC
# 2.08    Bayesian FMB confidence bounds and p-values (Bryzgalova/Huang/Julliard, 2020)
# 2.09    GRS-FAR confidence bounds, iid-normal assumption (Kleibergen/Zhan, 2020)
# 2.10    Bootstrap Confidence Intervals (Burnside, 2011)
# 
# SECTION 3)  SIGNIFICANCE TESTS FOR LAMBDA (MARKET PRICE OF RISK) IN A
#             CROSS-SECTIONAL REGRESSION WITH INTERCEPT
# 3.01    Fama/MacBeth standard errors
# 3.02    OLS standard errors (no Shanken correction)
# 3.03    Shanken-corrected standard errors
# 3.04    GMM standard errors, Newey-West
# 3.05    GMM standard errors, VARHAC
# 3.06    Giglio/Xiu three pass method
# 3.07    Kan/Robotti/Shanken robust standard errors
# 3.08    Bayesian FMB confidence bounds and p-values (Bryzgalova/Huang/Julliard, 2020)
# 3.09    GRS-FAR confidence bounds, iid-normal assumption (Kleibergen/Zhan, 2020)
# 3.10    Bootstrap Confidence Intervals (Burnside, 2011)
# 
# SECTION 4)  MODEL TESTS ASSUMING NO INTERCEPT IN CROSS-SECTIONAL REGRESSION 
#             (OR IT DOES NOT MATTER)
# 4.01    GRS test (Gibbons, Ross, Shanken, 1989, Econometrica)
# 4.02    "Asymptotic GRS test", assume residuals are iid
# 4.03    GRS test, estimate covariance using Newey West
# 4.04    GRS test, estimate covariance matrix using varhac
# 4.05    CHI^2 test on pricing errors (OLS, no Shanken correction)
# 4.06    CHI^2 test on pricing errors including Shanken correction
# 4.07    CHI^2 test on pricing errors using GMM standard errors (Newey-West)
# 4.08    CHI^2 test on pricing errors using GMM standard errors (VARHAC)
# 4.09    FAR test (under normality assumption)
# 4.10    Asymptotic FAR test 
# 4.11    GLSLM test (under normality assumption)
# 4.12    Asymptotic GLSLM test 
# 4.13    JGLS test (under normality assumption)
# 4.14    Asymptotic JGLS test 
# 4.15    FMLM test (under normality assumption)
# 4.16    Asymptotic FMLM test 
# 4.17    JFM test (under normality assumption)
# 4.18    Asymptotic JFM test
# 4.19    Hotelling (H) test (under normality assumption)
# 4.20    Asymptotic Hotelling (H) test
# 4.21    Bryzgalova/Huang/Julliard (2020) test of hypothesis R^2<=0 and confidence interval of R2 
#  
# SECTION 5)  MODEL TESTS ASSUMING AN INTERCEPT IN CROSS-SECTIONAL REGRESSION
# 5.05    CHI^2 test on pricing errors (OLS, no Shanken correction)
# 5.06    CHI^2 test on pricing errors including Shanken correction
# 5.07    CHI^2 test on pricing errors using GMM standard errors (Newey-West)
# 5.08    CHI^2 test on pricing errors using GMM standard errors (VARHAC)
# 5.09    Shanken's (1985) asymptotic CSRT test
# 5.10    Shanken's (1985) finite sample CSRT test
# 5.11    Kan/Robotti/Shanken test of hypothesis R^2=1
# 5.12    Kan/Robotti/Shanken test of hypothesis R^2=0 (imposing H0: Lambda=0_K when estimating covariance matrix)
# 5.13    Kan/Robotti/Shanken test of hypothesis R^2=0 (without imposing H0: Lambda=0_K when estimating covariance matrix)
# 5.14    Standard error of sample R2 (Kan/Robotti/Shanken)
# 5.15    Wald test of H0: Lambda=0_K (imposing H0 when estimating covariance matrix)
# 5.16    Wald test of H0: Lambda=0_K (without imposing H0 when estimating covariance matrix)
# 5.17    Bryzgalova/Huang/Julliard (2020) test of hypothesis R^2<=0 and confidence interval of R2 (constant is not part of the pricing errors)
# 5.18    Bryzgalova/Huang/Julliard (2020) confidence interval of R2i (constant is part of the pricing errors)
# 
# SECTION 6)  SIGNIFICANCE TESTS FOR LINEAR SDF SPECIFICATIONS (SDF LOADINGS)
# 6.01    Test of the SDF:  m=1-b'(f-E(f)), W=1, without common pricing error, Cochrane(2005) Chapter 13.2
# 6.02    Test of the SDF:  m=1-b'(f-E(f)), W=1, with common pricing error, Burnside(2011)
# 6.03    Test of the SDF:  m=1-b'(f-E(f)),  W=S^-1, without common pricing error, Gospodinov, Kan, and Robotti(2014)
# 6.04    Robust test of the SDF:  m=1-b'(f-E(f)),  W=S^-1, without common pricing error, Gospodinov, Kan, and Robotti(2014)


# Import phyton libaries
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

def vec(a):
    return a.flatten('F')[:,None]


def omnibus(R, f, method, Lambda0 = 0, level = 0.05):
    
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
    
    if method == 2.01:
        Lambda_t = np.full((T, K), np.nan)
        
        for t in range(0,T):
            Lambda_t[t, :] = R[t,:] @ A.T
        
        SE_Lambda = np.sqrt(np.mean((Lambda_t - np.ones((T,1)) @ np.mean(Lambda_t, axis = 0)[:, None].T) ** 2, axis = 0) / T).T
        T_Lambda = ((Lambda - Lambda0) / SE_Lambda)[0,:]
        
        ans['SE'] = SE_Lambda
        ans ['T'] = T_Lambda 
        ans['pval'] = 1 - sps.t.cdf(np.abs(T_Lambda), T-K)
    
    #------------------------------------------------------------------------------
    # Method 2.02) OLS standard errors (no Shanken correction)
    #------------------------------------------------------------------------------
    
    if method == 2.02:
        Sf = np.cov(f.T, ddof = 0)
        AV_Lambda = A @ S @ A.T + Sf
        SE_Lambda = np.sqrt(np.diag(AV_Lambda / T))
        T_Lambda = ((Lambda -Lambda0) / SE_Lambda)[0,:]
        
        ans['AV'] = AV_Lambda
        ans['SE'] = SE_Lambda
        ans['T'] = T_Lambda
        ans['pval'] = 1 - sps.t.cdf(np.abs(T_Lambda), T-K) 
    
    #------------------------------------------------------------------------------
    # Method 2.03) Shanken-corrected standard errors
    #------------------------------------------------------------------------------
    
    if method == 2.03:
        Sf = np.cov(f.T, ddof = 0)
        if f.shape[1] == 1:
            Shanken = 1  + Lambda.T / Sf @ Lambda
        else:
            Shanken = 1  + (np.linalg.solve(Sf.T, Lambda).T @ Lambda)
        AV_Lambda = A @ S @ A.T * Shanken + Sf
        SE_Lambda = np.sqrt(np.diag(AV_Lambda / T))
        T_Lambda = ((Lambda -Lambda0) / SE_Lambda)[0,:]
        
        ans['AV'] = AV_Lambda
        ans['SE'] = SE_Lambda
        ans['T'] = T_Lambda
        ans['pval'] = 1 - sps.t.cdf(np.abs(T_Lambda), T-K) 
        
    #------------------------------------------------------------------------------
    # Method 2.04) GMM standard errors, Newey-West
    #------------------------------------------------------------------------------
    
    if method == 2.04:
        for i in range(0, np.size(e, axis=1)):
            if i == 0:
                g = np.matlib.repmat(e[:, i], K+1, 1).T * F 
            else:   
                g = np.append(g, np.matlib.repmat(e[:, i], K+1, 1).T * F, axis = 1)
                
        g_cs  = R - np.matlib.repmat((beta @ Lambda).T, T,1) # time series of cross section sample moments
        g_GMM = np.append(g, g_cs, axis = 1) # T x n*(k+2) -> time series of sample moments
        
        # GMM Standard Errors, Newey West
        Scs_hac = nw.nw_func(g_GMM, lags)
        
        aT = np.append(np.append(np.eye((N*(K+1))), np.zeros((N*(K+1), N)), axis = 1), np.append(np.zeros((K, N*(K+1))), beta.T, axis = 1), axis = 0)
        Mf = F.T @ (F/T)
        dT = np.append(np.append((-1)*np.kron(np.eye(N), Mf),np.zeros((N*(K+1), K)),axis = 1),np.append((-1)*np.kron(np.eye(N),np.insert(Lambda,0,0)),(-1)* beta,axis = 1), axis = 0)
        
        AV_hac = np.linalg.solve((aT @ dT), aT) @ Scs_hac @ np.linalg.solve((aT @ dT), aT).T
        AV_Lambda = AV_hac[N*(K+1):len(AV_hac), N*(K+1):len(AV_hac)]
        SE_Lambda = np.sqrt(np.diag(AV_Lambda/T))
        T_Lambda = ((Lambda - Lambda0)/SE_Lambda)[0,:]
        
        ans['AV'] = AV_Lambda
        ans['SE'] = SE_Lambda
        ans['T'] = T_Lambda
        ans['pval'] = 1 - sps.t.cdf(np.abs(T_Lambda), T-K) 
        
    #------------------------------------------------------------------------------
    # Method 2.05) GMM standard errors, VARHAC
    #------------------------------------------------------------------------------
    
    if method == 2.05:
        for i in range(0, np.size(e, axis=1)):
            if i == 0:
                g = np.matlib.repmat(e[:, i], K+1, 1).T * F 
            else:   
                g = np.append(g, np.matlib.repmat(e[:, i], K+1, 1).T * F, axis = 1)
                
        g_cs  = R - np.matlib.repmat((beta @ Lambda).T, T,1) # time series of cross section sample moments
        g_GMM = np.append(g, g_cs, axis = 1) # T x n*(k+2) -> time series of sample moments
        
        # GMM Standard Errors, VARHAC
        Scs_varhac = hac_var.hac_var_func(g_GMM - np.matlib.repmat(np.mean(g_GMM, axis = 0)[:, None].T,T,1),1,0) # Call VARHAC function from Burnside
        
        aT = np.append(np.append(np.eye((N*(K+1))), np.zeros((N*(K+1), N)), axis = 1), np.append(np.zeros((K, N*(K+1))), beta.T, axis = 1), axis = 0)
        Mf = F.T @ (F/T)
        dT = np.append(np.append((-1)*np.kron(np.eye(N), Mf),np.zeros((N*(K+1), K)),axis = 1),np.append((-1)*np.kron(np.eye(N),np.insert(Lambda,0,0)),(-1)* beta,axis = 1), axis = 0)
        
        AV_varhac = np.linalg.solve((aT @ dT), aT) @ Scs_varhac @ np.linalg.solve((aT @ dT), aT).T
        AV_Lambda = AV_varhac[N*(K+1):len(AV_varhac), N*(K+1):len(AV_varhac)]
        SE_Lambda = np.sqrt(np.diag(AV_Lambda/T))
        T_Lambda = ((Lambda - Lambda0)/SE_Lambda)[0,:]
        
        ans['AV'] = AV_Lambda
        ans['SE'] = SE_Lambda
        ans['T'] = T_Lambda
        ans['pval'] = 1 - sps.t.cdf(np.abs(T_Lambda), T-K)
    
    #------------------------------------------------------------------------------
    # Method 2.08) Bayesian FMB confidence bounds and p-values 
    #              (Bryzgalova/Huang/Julliard, 2020)
    #------------------------------------------------------------------------------
    
    if method == 2.08:
        sim_length = 500;
        CI_bound = 0.95;
        
        f_demean = f - np.mean(f, axis=0);
        Lambda_dist = np.zeros((sim_length,K))
        x = np.append(np.ones((T,1)), f_demean, axis = 1)
        xx_inv = np.linalg.pinv(x.T @ x)
        B_ols = xx_inv @ (x.T @ R)
        e = R - x @ B_ols
        Sigma_ols = (e.T @ e)/T
        
        for i in range(0, sim_length):
            Sigma = sps.invwishart.rvs(T-K-1, T * Sigma_ols)
            Var_B = np.kron(Sigma, xx_inv)
            B_ols_vec = B_ols.flatten('F')
            rng = np.random.default_rng()
            B_vec = rng.multivariate_normal(B_ols_vec.T, Var_B)
            B_i = np.reshape(B_vec, (K+1,N), order = 'F')
            a_i = B_i[0,:]
            beta_i = B_i[1:len(B_i),:].T
            Lambda_dist[i,:] = (np.linalg.pinv(beta_i.T @ beta_i) @ (beta_i.T @ a_i)).T
        
        ans['Lambda_dist'] = Lambda_dist
        ans['CI'] = np.quantile(Lambda_dist,((1-CI_bound)/2, 1-(1-CI_bound)/2), axis=0)
        av_sign = -1+2*(np.mean(Lambda_dist>0, axis = 0) > 0.5)
        ans['pval'] = np.mean((((np.ones((sim_length,1)) @ av_sign[:,None].T) * (Lambda_dist - np.ones((sim_length,1)) @ Lambda0.T)) <0), axis=0)
            
            
    #------------------------------------------------------------------------------
    # Method 2.09) GRS-FAR test (under iid-normality assumption) 
    #              (Kleibergen/Zhan, 2020)
    #------------------------------------------------------------------------------
    
    if method == 2.09:
        F_bar = f.T - np.mean(f, axis=0)[:, None] @ np.ones((1,T))
        Q = F_bar @ F_bar.T / T
        F_exp = np.append(f.T, np.ones((1,T)), axis = 0)
        B_exp = np.linalg.solve((F_exp @ F_exp.T).T,(R.T @ F_exp.T).T).T
        Sigma_tt = 1/T*(R.T - B_exp @ F_exp) @ (R.T - B_exp @ F_exp).T
        Bhat = B_exp[:,:-1]
        
        cTemp = N/(T-N-K) * sps.f.ppf(1-level,N,T-N-K)
        cA = np.linalg.solve(Sigma_tt.T, Bhat).T @ Bhat - cTemp/Q
        cB = -2*np.linalg.solve(Sigma_tt.T, Bhat).T @  np.mean(R, axis=0).T
        cC = np.linalg.solve(Sigma_tt.T, np.mean(R, axis=0)).T @ np.mean(R, axis=0).T - cTemp
        
        #TODO: if-Blocks
        # values for debugging purpose
        ans['pval'] = 0
        ans['reject_L0'] = 0
        
    
    #------------------------------------------------------------------------------
    # Method 2.10) Bootstrap Confidence Intervals (e.g., Burnside, 2011)
    #------------------------------------------------------------------------------
    
    if method == 2.10:
        seed = 1
        boot_length = 500
        Lambda_dist = np.full((boot_length, K), np.NaN)
    
        idx = block_bootstrap.block_bootstrap_func(f[:, :1], boot_length, 1, seed)[1]
    
        for tau in range(0, boot_length):
            f_BOOT = f[idx[:, tau], :]
            R_BOOT = R[idx[:, tau], :]
            Lambda_dist[tau, :] = FMB_coefficients.FMB_coefficients_func(R_BOOT, f_BOOT, 0)[1]
    
        ans['Lambda_dist'] = Lambda_dist
        ans['CI'] = np.quantile(Lambda_dist, (level/2, 1-level/2), axis=0)
        
        av_sign = -1 + 2*(np.mean(Lambda_dist > 0, axis=0) > 0.5)
        ans['pval'] = (np.mean((np.ones((boot_length,1)) @ av_sign[:,None].T * (Lambda_dist - np.ones((boot_length,1)) @ Lambda0.T)) < 0, axis=0))[:,None].T
        
        H_reject_Leql0 = np.full((1, K), np.NaN)
        for k in range(0,K):
            Lambda_CIlow = ans['CI'][0,k]
            Lambda_CIhigh = ans['CI'][1,k]
            H_reject_Leql0[0,k] = Lambda_CIhigh >  0 and Lambda_CIlow > 0 or Lambda_CIhigh < 0 and Lambda_CIlow < 0
            
        ans['H_reject_Leql0'] = H_reject_Leql0  
    
    
    #------------------------------------------------------------------------------
    # SECTION 3)    SIGNIFICANCE TESTS FOR LAMBDA (MARKET PRICE OF RISK) IN A
    #               CROSS-SECTIONAL REGRESSION WITH INTERCEPT
    #------------------------------------------------------------------------------
    
    #------------------------------------------------------------------------------
    # Method 3.01)  Fama/MacBeth standard errors
    #------------------------------------------------------------------------------
    
    if method == 3.01:
        Theta_t = np.full((T, K+1), np.nan)
        
        for t in range(0,T):
            Theta_t[t, :] = R[t,:] @ A.T
        
        SE_Theta = np.sqrt(np.mean((Theta_t - np.ones((T,1)) @ np.mean(Theta_t, axis = 0)[:, None].T) ** 2, axis = 0) / T).T
        T_Theta = ((Theta - Lambda0) / SE_Theta)[0,:]
        
        ans['SE'] = SE_Theta
        ans ['T'] = T_Theta 
        ans['pval'] = 1 - sps.t.cdf(np.abs(T_Theta), T-K)
        
    #------------------------------------------------------------------------------
    # Method 3.02)  OLS standard errors (no Shanken correction)
    #------------------------------------------------------------------------------
    
    if method == 3.02:
        if f.shape[1] == 1:
            Sf = np.append(np.zeros((K+1,1)), np.append(np.zeros((1,K)), np.expand_dims(np.cov(f.T, ddof = 0), axis = (0,1)), axis = 0), axis = 1)
        else:
            Sf = np.append(np.zeros((K+1,1)), np.append(np.zeros((1,K)), np.cov(f.T, ddof = 0), axis = 0), axis = 1)
        AV_Theta = A @ S @ A.T + Sf
        SE_Theta = np.sqrt(np.diag(AV_Theta / T))
        T_Theta = ((Theta -Lambda0) / SE_Theta)[0,:]
        
        ans['AV'] = AV_Theta
        ans['SE'] = SE_Theta
        ans['T'] = T_Theta
        ans['pval'] = 1 - sps.t.cdf(np.abs(T_Theta), T-K)
        
        
        
    #------------------------------------------------------------------------------
    # Method 3.03)  Shanken-corrected standard errors
    #------------------------------------------------------------------------------
    
    if method == 3.03:
        if f.shape[1] == 1:
            Sf = np.append(np.zeros((K+1,1)), np.append(np.zeros((1,K)), np.expand_dims(np.cov(f.T, ddof = 0), axis=(0,1)), axis = 0), axis = 1)
            Shanken = 1  + Lambda.T / np.cov(f.T, ddof = 0) @ Lambda
        else:
            Sf = np.append(np.zeros((K+1,1)), np.append(np.zeros((1,K)), np.cov(f.T, ddof = 0), axis = 0), axis = 1)
            Shanken = 1  + (np.linalg.solve(np.cov(f.T, ddof = 0).T, Lambda).T @ Lambda)
            
        AV_Theta = A @ S @ A.T * Shanken + Sf # Shanken standard errors
        SE_Theta = np.sqrt(np.diag(AV_Theta / T))
        T_Theta = ((Theta -Lambda0) / SE_Theta)[0,:]
        
        ans['AV'] = AV_Theta
        ans['SE'] = SE_Theta
        ans['T'] = T_Theta
        ans['pval'] = 1 - sps.t.cdf(np.abs(T_Theta), T-K) 
    
    #------------------------------------------------------------------------------
    # Method 3.04)  GMM standard errors, Newey-West
    #------------------------------------------------------------------------------
    
    if method == 3.04:
        for i in range(0, np.size(e, axis=1)):
            if i == 0:
                g = np.matlib.repmat(e[:, i], K+1, 1).T * F 
            else:   
                g = np.append(g, np.matlib.repmat(e[:, i], K+1, 1).T * F, axis = 1)
                
        g_cs  = R - np.matlib.repmat((beta @ Lambda).T, T,1) # time series of cross section sample moments
        g_GMM = np.append(g, g_cs, axis = 1) # T x n*(k+2) -> time series of sample moments
        
        # GMM Standard Errors, Newey West
        Scs_hac = nw.nw_func(g_GMM, lags)
        
        aT = np.append(np.append(np.eye((N*(K+1))), np.zeros((N*(K+1), N)), axis = 1), np.append(np.zeros((K+1, N*(K+1))), X.T, axis = 1), axis = 0)
        Mf = F.T @ (F/T)
        dT = np.append(np.append((-1)*np.kron(np.eye(N), Mf),np.zeros((N*(K+1), K+1)),axis = 1),np.append((-1)*np.kron(np.eye(N),np.insert(Lambda,0,0)),(-1)* X,axis = 1), axis = 0)
        
        AV_hac = np.linalg.solve((aT @ dT), aT) @ Scs_hac @ np.linalg.solve((aT @ dT), aT).T
        AV_Theta = AV_hac[N*(K+1):, N*(K+1):]
        SE_Theta = np.sqrt(np.diag(AV_Theta/T))
        T_Theta = ((Theta - Lambda0)/SE_Theta)[0,:]
        
        ans['AV'] = AV_Theta
        ans['SE'] = SE_Theta
        ans['T'] = T_Theta
        ans['pval'] = 1 - sps.t.cdf(np.abs(T_Theta), T-K)
    
    #------------------------------------------------------------------------------
    # Method 3.05)  GMM standard errors, VARHAC
    #------------------------------------------------------------------------------
    
    if method == 3.05:
        for i in range(0, np.size(e, axis=1)):
            if i == 0:
                g = np.matlib.repmat(e[:, i], K+1, 1).T * F 
            else:   
                g = np.append(g, np.matlib.repmat(e[:, i], K+1, 1).T * F, axis = 1)
                
        g_cs  = R - np.matlib.repmat((beta @ Lambda).T, T,1) # time series of cross section sample moments
        g_GMM = np.append(g, g_cs, axis = 1) # T x n*(k+2) -> time series of sample moments
        
        # GMM Standard Errors, Newey West
        Scs_varhac = hac_var.hac_var_func((g_GMM - np.matlib.repmat(np.mean(g_GMM, axis = 0), T, 1)), 1, 0)
        
        aT = np.append(np.append(np.eye((N*(K+1))), np.zeros((N*(K+1), N)), axis = 1), np.append(np.zeros((K+1, N*(K+1))), X.T, axis = 1), axis = 0)
        Mf = F.T @ (F/T)
        dT = np.append(np.append((-1)*np.kron(np.eye(N), Mf),np.zeros((N*(K+1), K+1)),axis = 1),np.append((-1)*np.kron(np.eye(N),np.insert(Lambda,0,0)),(-1)* X,axis = 1), axis = 0)
        
        AV_varhac = np.linalg.solve((aT @ dT), aT) @ Scs_varhac @ np.linalg.solve((aT @ dT), aT).T
        AV_Theta = AV_varhac[N*(K+1):, N*(K+1):]
        SE_Theta = np.sqrt(np.diag(AV_Theta/T))
        T_Theta = ((Theta - Lambda0)/SE_Theta)[0,:]
        
        ans['AV'] = AV_Theta
        ans['SE'] = SE_Theta
        ans['T'] = T_Theta
        ans['pval'] = 1 - sps.t.cdf(np.abs(T_Theta), T-K)
    
    #------------------------------------------------------------------------------
    # Method 3.06) Giglio/Xiu three pass method
    #------------------------------------------------------------------------------
    
    if method == 3.06:
        pmax = np.min((N,7))
        alphahat = np.full((N, pmax), np.nan)
        R2F = np.full((1, pmax), np.nan)
        Gammahat  = np.full((K+1, pmax), np.nan)
        avarhat = np.full((K+1, pmax), np.nan)
        gthat = np.full((K,T,pmax), np.nan)
        rtbar = R.T - np.mean(R, axis = 0)[:,None] @ np.ones((1,T))
        gtbar = f.T - np.mean(f, axis = 0)[:,None] @ np.ones((1,T))
        
        svd = np.linalg.svd(rtbar/np.sqrt(N)/np.sqrt(T), full_matrices = False) #SVD
        block = svd[1][:, None]
        eigvec = svd[2].T
        tempeig = block ** 2 
        
        for phat in range(0, pmax): # different # of principle components
            vhat = T ** 0.5 * eigvec[:, 0:phat+1].T    
            betahat = T ** (-1) * rtbar @ vhat.T
            Sigmavhat =  vhat @ vhat.T / T
            Gammatilde = (np.linalg.solve((np.append(np.ones((N,1)), betahat, axis = 1).T @ np.append(np.ones((N,1)), betahat, axis = 1)), np.append(np.ones((N,1)), betahat, axis = 1).T)  @ np.mean(R, axis = 0).T)[:, None] # FMB with PCs
            etahat = np.linalg.solve((vhat @ vhat.T), (gtbar @ vhat.T).T).T # time series regression
            what = gtbar - etahat @ vhat
            alphahat[:, phat] =  R_bar - np.append(np.ones((N,1)), betahat, axis = 1) @ np.squeeze(Gammatilde) # pricing errors
            gthat[:,:, phat] = etahat @ vhat # cleaned factor proxies
            Gammahat[0, phat] = Gammatilde[0, 0]   # BUGFIX: Gammatilde[0] is shape (1,);
                                                  # NumPy >= 2.3 refuses to put it in a scalar slot.
            Gammahat[1:, phat] = np.squeeze(etahat @ Gammatilde[1:]) # combine CS and TS estimates
            Miota = np.eye(N) - np.ones((N,1)) @ np.ones((1,N)) / N # goodnes of Fit
            R2F[0, phat] = R_bar.T @ Miota @ np.linalg.solve((betahat.T @ Miota @ betahat).T, betahat.T).T @ betahat.T @ Miota @ R_bar/(R_bar.T @ Miota @ R_bar) 
            
            # Newey-West estimation of Avar
            Pi11hat = np.zeros((K*(phat+1), K*(phat+1)))
            Pi12hat = np.zeros((K*(phat+1), phat+1))
            Pi22hat = np.zeros((phat+1, phat+1))
            
            for t in range(0,T):
                Pi11hat = Pi11hat + vec(what[:,t][:,None] @ vhat[:,t][:,None].T) @ vec(what[:,t][:,None] @ vhat[:,t][:,None].T).T / T
                Pi12hat = Pi12hat + vec(what[:,t][:,None] @ vhat[:,t][:,None].T) @ vhat[:,t][:,None].T / T
                Pi22hat = Pi22hat + vhat[:,t][:,None] @ vhat[:,t][:,None].T / T
                
                for s in range(0, np.min((t, lags))):
                    Pi11hat = Pi11hat + 1/T * (1-(s+1)/(lags+1)) * (vec(what[:,t][:,None] @ vhat[:,t][:,None].T) @ vec(what[:,t-s-1][:,None] @ vhat[:,t-s-1][:,None].T).T + vec(what[:,t-s-1][:,None] @ vhat[:,t-s-1][:,None].T) @ vec(what[:,t][:,None] @ vhat[:,t][:,None].T).T)
                    Pi12hat = Pi12hat + 1/T * (1-(s+1)/(lags+1)) * (vec(what[:,t][:,None] @ vhat[:,t][:,None].T) @ vhat[:,t-s-1][:,None].T + vec(what[:,t-s-1][:,None] @ vhat[:,t-s-1][:,None].T) @ vhat[:,t][:,None].T)
                    Pi22hat = Pi22hat + 1/T * (1-(s+1)/(lags+1)) * (vhat[:,t][:,None] @ vhat[:,t-s-1][:,None].T + vhat[:,t-s-1][:,None] @ vhat[:,t][:,None].T)
    
            v1 = (1 - np.linalg.solve((betahat.T @ betahat/N).T, np.mean(betahat, axis = 0).T) @ np.mean(betahat, axis = 0).T) ** (-1) * np.var(alphahat[:,phat], ddof = 1)/N
            v2 = np.diag(np.kron(np.linalg.solve(Sigmavhat.T, Gammatilde[1:]).T, np.eye(K)) @ Pi11hat @ np.kron(np.linalg.solve(Sigmavhat, Gammatilde[1:]), np.eye(K))/T + 
                         np.kron(np.linalg.solve(Sigmavhat.T, Gammatilde[1:]).T, np.eye(K)) @ Pi12hat @ etahat.T/T + (np.kron(np.linalg.solve(Sigmavhat.T, Gammatilde[1:]).T, np.eye(K)) @ Pi12hat @ etahat.T).T/T +
                         etahat @ Pi22hat @ etahat.T / T) + np.diag(np.var(alphahat[:,phat], ddof = 1) * np.linalg.solve((betahat.T @ betahat / N - np.mean(betahat, axis = 0)[:,None] @ np.mean(betahat, axis = 0)[:,None].T).T,  etahat.T).T @ etahat.T) / N
            
            avarhat[:, phat] = np.squeeze(np.append(np.array(v1)[None,None],v2[:,None], axis = 0))
            
        # choose number of latent factors
        ppmax = 20
        obj = tempeig[0:ppmax] + 0.5 * np.array(range(1, ppmax+1))[:,None] * (np.log(T) + np.log(N)) * (N ** (-0.5) + T ** (-0.5)) * np.median(tempeig[0:ppmax])
        phat = np.argmin(obj)-1
        
        ans['Theta_GX'] = Gammahat[:,phat]
        ans['SE'] = np.sqrt(avarhat[:,phat])
        ans['T'] = ((Gammahat[:,phat]-Lambda0) / ans['SE'])[0,:]
        ans['RSQ_cs_GX'] = R2F[0, phat] # constant IS part of the pricing errors
        ans['alpha_GX'] = alphahat[:,phat]
        ans['cleanF'] = gthat[:,:,phat].T
        ans['pval'] = 1 - sps.t.cdf(np.abs(ans['T']), T-K)
    
    
    #------------------------------------------------------------------------------
    # Method 3.07)  Kan/Robotti/Shanken robust standard errors
    #------------------------------------------------------------------------------
    
    if method == 3.07:
        V = np.cov(f.T, ddof = 0)
        f_demean = f - np.ones((T,1)) @ np.mean(f, axis = 0)[:, None].T
        R_demean = R - np.ones((T,1)) @ np.mean(R, axis = 0)[:, None].T
        Theta_t =  np.linalg.solve((X.T @ X).T, (R_demean @ X).T).T 
        phi_t = Theta_t - np.append(np.zeros((T,1)), f_demean, axis = 1)
        ut = R_demean @ PEi
        
        if f.shape[1] == 1:
            zt = np.append(np.zeros((T,1)), np.linalg.solve(np.expand_dims(V.T, axis = (0,1)), f_demean.T).T, axis = 1)
            wt = np.linalg.solve(np.expand_dims(V.T, axis = (0,1)), f_demean.T).T @ Lambda
        else:
            zt = np.append(np.zeros((T,1)), np.linalg.solve(V.T, f_demean.T).T, axis = 1)
            wt = np.linalg.solve(V.T, f_demean.T).T @ Lambda
        ht0 = Theta_t + np.linalg.solve((X.T @ X).T, zt.T).T * (ut[:,None] @ np.ones((1, K+1)))
        ht2 = ht0 - phi_t * (wt[:,None]  @ np.ones((1, K+1)))
        V2 = nw.nw_func(ht2, lags)
        
        ans['SE'] = np.sqrt(np.diag(V2) / T)
        ans['T'] = ((Theta - Lambda0) / ans['SE'])[0,:]
        ans['pval'] = 1 - sps.t.cdf(np.abs(ans['T']), T-K)
        
    
    #------------------------------------------------------------------------------
    # Method 3.08)  Bayesian FMB confidence bounds and p-values 
    #               (Bryzgalova/Huang/Julliard, 2020)
    #------------------------------------------------------------------------------
    
    if method == 3.08:
        sim_length = 500;
        CI_bound = 0.95;
        
        f_demean = f - np.mean(f, axis=0);
        Lambda_dist = np.zeros((sim_length,K+1))
        x = np.append(np.ones((T,1)), f_demean, axis = 1)
        xx_inv = np.linalg.pinv(x.T @ x)
        B_ols = xx_inv @ (x.T @ R)
        e = R - x @ B_ols
        Sigma_ols = (e.T @ e)/T
        
        for i in range(0, sim_length):
            Sigma = sps.invwishart.rvs(T-K-1, T * Sigma_ols)
            Var_B = np.kron(Sigma, xx_inv)
            B_ols_vec = B_ols.flatten('F')
            rng = np.random.default_rng()
            B_vec = rng.multivariate_normal(B_ols_vec.T, Var_B)
            B_i = np.reshape(B_vec, (K+1,N), order ='F')
            a_i = B_i[0,:]
            beta_i = B_i[1:len(B_i),:].T
            H = np.append(np.ones((N,1)), beta_i, axis = 1)
            HH_inv = np.linalg.pinv(H.T @ H)
            Lambda_dist[i,:] = HH_inv @ (H.T @ a_i)
        
        ans['Lambda_dist'] = Lambda_dist
        ans['CI'] = np.quantile(Lambda_dist,((1-CI_bound)/2, 1-(1-CI_bound)/2), axis=0)
        av_sign = -1+2*(np.mean(Lambda_dist>0, axis = 0) > 0.5)
        ans['pval'] = np.mean((((np.ones((sim_length,1)) @ av_sign[:,None].T) * (Lambda_dist - np.ones((sim_length,1)) @ Lambda0.T)) <0), axis=0)
    
    
    #------------------------------------------------------------------------------
    # Method 3.09) GRS-FAR test (under normality assumption) 
    #              (Kleibergen/Zhan, 2020)
    #------------------------------------------------------------------------------
    
    # TODO: method 3.09
    if method == 3.09:
        # values for debugging purpose
        ans['pval'] = 0
        ans['reject_L0'] = 0
    
    #------------------------------------------------------------------------------
    # Method 3.10) Bootstrap Confidence Intervals 
    #              (e.g., Burnside, 2011)
    #------------------------------------------------------------------------------
    
    if method == 3.10:
        seed = 1
        boot_length = 500
        Lambda_dist = np.full((boot_length, K), np.NaN)
        const_dist = np.full((boot_length, 1), np.NaN)
        
        idx = block_bootstrap.block_bootstrap_func(f[:, :1], boot_length, 1, seed)[1]
    
        for tau in range(0, boot_length):
            f_BOOT = f[idx[:, tau], :]
            R_BOOT = R[idx[:, tau], :]
            FMB = FMB_coefficients.FMB_coefficients_func(R_BOOT, f_BOOT, 1)
            const_dist[tau, :] = FMB[0]
            Lambda_dist[tau, :] = FMB[1] 
    
        ans['Lambda_dist'] = np.concatenate((const_dist, Lambda_dist), axis = 1)
        ans['CI'] = np.quantile(ans['Lambda_dist'], (level/2, 1-level/2), axis = 0)
        
        av_sign = -1 + 2*(np.mean(ans['Lambda_dist'] > 0, axis=0) > 0.5)
        ans['pval'] = (np.mean((np.ones((boot_length,1)) @ av_sign[:,None].T * (ans['Lambda_dist'] - np.ones((boot_length,1)) @ Lambda0.T)) < 0, axis=0))[:,None].T
        
        H_reject_Leql0 = np.full((1, K+1), np.NaN)
        for k in range(0,K+1):
            Lambda_CIlow = ans['CI'][0,k]
            Lambda_CIhigh = ans['CI'][1,k]
            H_reject_Leql0[0,k] = Lambda_CIhigh >  0 and Lambda_CIlow > 0 or Lambda_CIhigh < 0 and Lambda_CIlow < 0
            
        ans['H_reject_Leql0'] = H_reject_Leql0  
    
    
    #------------------------------------------------------------------------------
    # SECTION 4)   MODEL TESTS assuming no intercept in cross-sectional
    #              regression (or it does not matter)
    #------------------------------------------------------------------------------
    
    #------------------------------------------------------------------------------
    # Method 4.01) GRS test (Gibbons, Ross, Shanken, 1989, Econometrica)
    #              Assume residuals are iid
    #------------------------------------------------------------------------------
    
    if method == 4.01:
       AV = np.kron(S, FFi) # Covariance matrix of intercepts and betas
       Wald = np.linalg.solve(AV[0::(K+1),0::(K+1)],alpha[:,None]).T @ alpha[:,None]; # Wald Test
       ans['Test'] = (T-N-K) * Wald / (T*N) # GRS Test
       ans['pval'] = 1 - sps.f.cdf(ans['Test'],N,(T-N-K))  # p-value
    
    #------------------------------------------------------------------------------
    # Method 4.02) "Asymptotic GRS test" (Gibbons, Ross, Shanken, 1989, Econometrica)
    #               Assume residuals are iid
    #------------------------------------------------------------------------------
    
    if method == 4.02:
        AV = np.kron(S, FFi) # Covariance matrix of intercepts and betas
        ans['Test'] = np.linalg.solve(AV[0::(K+1),0::(K+1)],alpha[:,None]).T @ alpha[:,None]; # Wald Test
        ans['pval'] = 1 - sps.chi2.cdf(ans['Test'],N)  # p-value
    
    #------------------------------------------------------------------------------
    # Method 4.03) GRS test (Gibbons, Ross, Shanken, 1989, Econometrica)
    #              Estimate covariance using Newey West
    #------------------------------------------------------------------------------
    
    if method == 4.03:
        for i in range(0, np.size(e, axis=1)):
            if i == 0:
                g = np.matlib.repmat(e[:, i], K+1, 1).T * F 
            else:   
                g = np.append(g, np.matlib.repmat(e[:, i], K+1, 1).T * F, axis = 1)
                
        S_hac = nw.nw_func(g,lags) # Call Newey-West function
        Di = np.kron(np.eye(N), FFi)
        AV_hac = Di @ S_hac @ Di * T # Covariance matrix of intercepts and betas
        ans['Test'] = np.linalg.solve(AV_hac[0::(K+1),0::(K+1)],alpha[:,None]).T @ alpha[:,None]; # Wald HAC Test
        ans['pval'] = 1 - sps.chi2.cdf(ans['Test'],N)  # HAC robust GRS Test
        
    #------------------------------------------------------------------------------
    # Method 4.04) GRS test (Gibbons, Ross, Shanken, 1989, Econometrica)
    #              Estimate covariance matrix using varhac
    #------------------------------------------------------------------------------
    
    if method == 4.04:
        for i in range(0, np.size(e, axis=1)):
            if i == 0:
                g = np.matlib.repmat(e[:, i], K+1, 1).T * F 
            else:   
                g = np.append(g, np.matlib.repmat(e[:, i], K+1, 1).T * F, axis = 1)
                
        S_hac = hac_var.hac_var_func(g,1,0) # Call VARHAC function from Burnside
        Di = np.kron(np.eye(N), FFi)
        AV_hac = Di @ S_hac @ Di * T # Covariance matrix of intercepts and betas
        ans['Test'] = np.linalg.solve(AV_hac[0::(K+1),0::(K+1)],alpha[:,None]).T @ alpha[:,None]; # Wald HAC Test
        ans['pval'] = 1 - sps.chi2.cdf(ans['Test'],N)  # HAC robust GRS Test
    
    #------------------------------------------------------------------------------
    # Method 4.05) CHI^2 test on pricing errors (OLS, no Shanken correction)
    #------------------------------------------------------------------------------
    
    if method == 4.05:
        Mx = np.eye(N) - np.linalg.solve((beta.T @ beta).T, beta.T).T @ beta.T
        ans['Test'] = T * PE[:,None].T @ np.linalg.pinv(Mx @ S @ Mx) @ PE[:,None] # Chi square test on pricing errors
        ans['pval'] = 1 - sps.chi2.cdf(ans['Test'],N-K)
        
    #------------------------------------------------------------------------------
    # Method 4.06) CHI^2 test on pricing errors including Shanken correction
    #------------------------------------------------------------------------------
    
    if method == 4.06:
        Shanken = 1  + (np.linalg.solve(np.cov(f.T, ddof = 0).T, Lambda).T @ Lambda)
        Mx = np.eye(N) - np.linalg.solve((beta.T @ beta).T, beta.T).T @ beta.T
        ans['Test'] = T * PE[:,None].T @ np.linalg.pinv(Shanken * Mx @ S @ Mx) @ PE[:,None] # Chi square test on pricing errors (test stat)
        ans['pval'] = 1 - sps.chi2.cdf(ans['Test'],N-K)
        
    #------------------------------------------------------------------------------
    # Method 4.07) CHI^2 test on pricing errors using GMM standard errors (Newey-West)
    #------------------------------------------------------------------------------
    
    if method == 4.07:
        # GMM Moments
        for i in range(0, np.size(e, axis=1)):
            if i == 0:
                g = np.matlib.repmat(e[:, i], K+1, 1).T * F 
            else:   
                g = np.append(g, np.matlib.repmat(e[:, i], K+1, 1).T * F, axis = 1)
                
        g_cs  = R - np.matlib.repmat((beta @ Lambda).T, T,1) # time series of cross section sample moments
        g_GMM = np.append(g, g_cs, axis = 1) # T x n*(k+2) -> time series of sample moments
        
        # GMM Standard Errors, Newey West
        Scs_hac = nw.nw_func(g_GMM, lags)
        
        aT = np.append(np.append(np.eye((N*(K+1))), np.zeros((N*(K+1), N)), axis = 1), np.append(np.zeros((K, N*(K+1))), beta.T, axis = 1), axis = 0)
        Mf = F.T @ (F/T)
        dT = np.append(np.append((-1)*np.kron(np.eye(N), Mf),np.zeros((N*(K+1), K)),axis = 1),np.append((-1)*np.kron(np.eye(N),np.insert(Lambda,0,0)),(-1)* beta,axis = 1), axis = 0)
        
        Vgt_hac = (np.eye(N*(K+2)) - np.linalg.solve((aT @ dT).T, dT.T).T @ aT) @ Scs_hac @ (np.eye(N*(K+2)) - np.linalg.solve((aT @ dT).T, dT.T).T @ aT).T
        ans['Test'] = T * PE[:,None].T @ np.linalg.pinv(Vgt_hac[N*(K+1):,N*(K+1):]) @ PE[:,None]
        ans['pval'] = 1 - sps.chi2.cdf(ans['Test'],N-K)
        
    #------------------------------------------------------------------------------
    # Method 4.08) CHI^2 test on pricing errors using GMM standard errors (VARHAC)
    #------------------------------------------------------------------------------
    
    if method == 4.08:
        # GMM Moments
        for i in range(0, np.size(e, axis=1)):
            if i == 0:
                g = np.matlib.repmat(e[:, i], K+1, 1).T * F 
            else:   
                g = np.append(g, np.matlib.repmat(e[:, i], K+1, 1).T * F, axis = 1)
                
        g_cs  = R - np.matlib.repmat((beta @ Lambda).T, T,1) # time series of cross section sample moments
        g_GMM = np.append(g, g_cs, axis = 1) # T x n*(k+2) -> time series of sample moments
        
        # GMM Standard Errors, VARHAC
        Scs_varhac = hac_var.hac_var_func(g_GMM-np.matlib.repmat(np.mean(g_GMM, axis=0),T,1),1,0) # Call VARHAC function from Burnside
        
        aT = np.append(np.append(np.eye((N*(K+1))), np.zeros((N*(K+1), N)), axis = 1), np.append(np.zeros((K, N*(K+1))), beta.T, axis = 1), axis = 0)
        Mf = F.T @ (F/T)
        dT = np.append(np.append((-1)*np.kron(np.eye(N), Mf),np.zeros((N*(K+1), K)),axis = 1),np.append((-1)*np.kron(np.eye(N),np.insert(Lambda,0,0)),(-1)* beta,axis = 1), axis = 0)
        
        Vgt_hac = (np.eye(N*(K+2)) - np.linalg.solve((aT @ dT).T, dT.T).T @ aT) @ Scs_varhac @ (np.eye(N*(K+2)) - np.linalg.solve((aT @ dT).T, dT.T).T @ aT).T
        ans['Test'] = T * PE[:,None].T @ np.linalg.pinv(Vgt_hac[N*(K+1):,N*(K+1):]) @ PE[:,None]
        ans['pval'] = 1 - sps.chi2.cdf(ans['Test'],N-K)
    
    #------------------------------------------------------------------------------
    # Method 4.09) FAR test (under normality assumption) - WITH CONSTANT
    #------------------------------------------------------------------------------
    
    if method == 4.09:
        if excess_ret == 0:
            R = R @ np.append(np.eye(N-1), (-1)*np.ones((N-1,1)), axis = 1).T # subtract nth asset return from other returns to come up with excess returns
            N = N-1
        
        R_demean = R - np.ones((T,1)) @ np.mean(R, axis = 0)[:, None].T
        f_demean = f - np.ones((T,1)) @ np.mean(f, axis = 0)[:, None].T
        f_pluslambda = f_demean + np.ones((T,1)) @ Lambda0.T
        f_pluslambda_2 = f_pluslambda.T @ f_pluslambda
        B_tilde = R.T @ np.linalg.solve(f_pluslambda_2.T, f_pluslambda.T).T
        B_hat = R_demean.T @ np.linalg.solve((f_demean.T @ f_demean).T, f_demean.T).T
        Sigma_hat = (R_demean - f_demean @ B_hat.T).T @ (R_demean - f_demean @ B_hat.T) / (T-K-1)
        Rmean_betalambda = np.mean(R, axis = 0)[:,None] - B_tilde @ Lambda0
        c_coef = T / (1 - np.linalg.solve((f_pluslambda_2/T).T, Lambda0).T @ Lambda0)
        FAR = c_coef * (np.linalg.solve(Sigma_hat.T, Rmean_betalambda).T @ Rmean_betalambda)
        
        ans['Test'] = (T-K-N)/N/(T-K-1) * FAR
        ans['pval'] = 1 - sps.f.cdf(ans['Test'],N,T-K-N) # p-value
        
        
    #------------------------------------------------------------------------------
    # Method 4.10) Asymptotic FAR test
    #------------------------------------------------------------------------------
    
    if method == 4.10:
        if excess_ret == 0:
            R = R @ np.append(np.eye(N-1), (-1)*np.ones((N-1,1)), axis = 1).T # subtract nth asset return from other returns to come up with excess returns
            N = N-1
        
        R_demean = R - np.ones((T,1)) @ np.mean(R, axis = 0)[:, None].T
        f_demean = f - np.ones((T,1)) @ np.mean(f, axis = 0)[:, None].T
        f_pluslambda = f_demean + np.ones((T,1)) @ Lambda0.T
        f_pluslambda_2 = f_pluslambda.T @ f_pluslambda
        B_tilde = R.T @ np.linalg.solve(f_pluslambda_2.T, f_pluslambda.T).T
        B_hat = R_demean.T @ np.linalg.solve((f_demean.T @ f_demean).T, f_demean.T).T
        Sigma_hat = (R_demean - f_demean @ B_hat.T).T @ (R_demean - f_demean @ B_hat.T) / (T-K-1)
        Rmean_betalambda = np.mean(R, axis = 0)[:,None] - B_tilde @ Lambda0
        c_coef = T / (1 - np.linalg.solve((f_pluslambda_2/T).T, Lambda0).T @ Lambda0)
        FAR = c_coef * (np.linalg.solve(Sigma_hat.T, Rmean_betalambda).T @ Rmean_betalambda)
        
        ans['Test'] = FAR
        ans['pval'] = 1 - sps.chi2.cdf(ans['Test'],N) # p-value 
    
    #------------------------------------------------------------------------------
    # Method 4.11) GLSLM test (under normality assumption)
    #------------------------------------------------------------------------------
    
    if method == 4.11:
        if excess_ret == 0:
            R = R @ np.append(np.eye(N-1), (-1)*np.ones((N-1,1)), axis = 1).T # subtract nth asset return from other returns to come up with excess returns
            N = N-1
        
        R_demean = R - np.ones((T,1)) @ np.mean(R, axis = 0)[:, None].T
        f_demean = f - np.ones((T,1)) @ np.mean(f, axis = 0)[:, None].T
        f_pluslambda = f_demean + np.ones((T,1)) @ Lambda0.T
        f_pluslambda_2 = f_pluslambda.T @ f_pluslambda
        B_tilde = R.T @ np.linalg.solve(f_pluslambda_2.T, f_pluslambda.T).T
        B_hat = R_demean.T @ np.linalg.solve((f_demean.T @ f_demean).T, f_demean.T).T
        Sigma_hat = (R_demean - f_demean @ B_hat.T).T @ (R_demean - f_demean @ B_hat.T) / (T-K-1)
        Rmean_betalambda = np.mean(R, axis = 0)[:,None] - B_tilde @ Lambda0
        c_coef = T / (1 - np.linalg.solve((f_pluslambda_2/T).T, Lambda0).T @ Lambda0)
        GLSLM = c_coef * np.linalg.solve(Sigma_hat.T, Rmean_betalambda).T @ np.linalg.solve((np.linalg.solve(Sigma_hat.T, B_tilde).T @ B_tilde).T,B_tilde.T).T @ np.linalg.solve(Sigma_hat.T, B_tilde).T @ Rmean_betalambda
        
        # Finite sample distribution of GLSLM
        m = 20000
        v_GLSLM = np.full((m,1), np.NaN)
        C = np.eye(N,N-K)
        rng = np.random.default_rng(seed = 1)
        
        for i in range(0,m):
            psi = rng.multivariate_normal(np.zeros(N), np.eye(N),1)
            WZ_i = rng.multivariate_normal(np.zeros(N), np.eye(N),T-K-1)
            W = WZ_i.T @ WZ_i / (T-K-1)
            PSI = np.reshape(psi.T,N,'F')[:,None]
            v_GLSLM[i] = np.linalg.solve(W.T, PSI).T @ PSI - PSI.T @ np.linalg.solve((C.T @ W @ C).T, C.T).T @ C.T @ PSI
        
        v_GLSLM = np.sort(v_GLSLM)
        index_pos = np.argwhere(v_GLSLM >= GLSLM)
        
        if np.size(index_pos, axis = 0) == 0:
            index_pos = m
        else:
            index_pos = index_pos[0,1]
            
        ans['Test'] = GLSLM
        ans['pval'] = 1-index_pos/m # p-value
        
        
    #------------------------------------------------------------------------------
    # Method 4.12) Asymptotic GLSLM test 
    #------------------------------------------------------------------------------
    
    if method == 4.12:
        if excess_ret == 0:
            R = R @ np.append(np.eye(N-1), (-1)*np.ones((N-1,1)), axis = 1).T # subtract nth asset return from other returns to come up with excess returns
            N = N-1
        
        R_demean = R - np.ones((T,1)) @ np.mean(R, axis = 0)[:, None].T
        f_demean = f - np.ones((T,1)) @ np.mean(f, axis = 0)[:, None].T
        f_pluslambda = f_demean + np.ones((T,1)) @ Lambda0.T
        f_pluslambda_2 = f_pluslambda.T @ f_pluslambda
        B_tilde = R.T @ np.linalg.solve(f_pluslambda_2.T, f_pluslambda.T).T
        B_hat = R_demean.T @ np.linalg.solve((f_demean.T @ f_demean).T, f_demean.T).T
        Sigma_hat = (R_demean - f_demean @ B_hat.T).T @ (R_demean - f_demean @ B_hat.T) / (T-K-1)
        Rmean_betalambda = np.mean(R, axis = 0)[:,None] - B_tilde @ Lambda0
        c_coef = T / (1 - np.linalg.solve((f_pluslambda_2/T).T, Lambda0).T @ Lambda0)
        GLSLM = c_coef * np.linalg.solve(Sigma_hat.T, Rmean_betalambda).T @ np.linalg.solve((np.linalg.solve(Sigma_hat.T, B_tilde).T @ B_tilde).T,B_tilde.T).T @ np.linalg.solve(Sigma_hat.T, B_tilde).T @ Rmean_betalambda
        
        ans['Test'] = GLSLM
        ans['pval'] = 1 - sps.chi2.cdf(ans['Test'], K) # p-value
        
    #------------------------------------------------------------------------------
    # Method 4.13) JGLS test (under normality assumption)
    #------------------------------------------------------------------------------
    
    if method == 4.13:
        if excess_ret == 0:
            R = R @ np.append(np.eye(N-1), (-1)*np.ones((N-1,1)), axis = 1).T # subtract nth asset return from other returns to come up with excess returns
            N = N-1
        
        R_demean = R - np.ones((T,1)) @ np.mean(R, axis = 0)[:, None].T
        f_demean = f - np.ones((T,1)) @ np.mean(f, axis = 0)[:, None].T
        f_pluslambda = f_demean + np.ones((T,1)) @ Lambda0.T
        f_pluslambda_2 = f_pluslambda.T @ f_pluslambda
        B_tilde = R.T @ np.linalg.solve(f_pluslambda_2.T, f_pluslambda.T).T
        B_hat = R_demean.T @ np.linalg.solve((f_demean.T @ f_demean).T, f_demean.T).T
        Sigma_hat = (R_demean - f_demean @ B_hat.T).T @ (R_demean - f_demean @ B_hat.T) / (T-K-1)
        Rmean_betalambda = np.mean(R, axis = 0)[:,None] - B_tilde @ Lambda0
        c_coef = T / (1 - np.linalg.solve((f_pluslambda_2/T).T, Lambda0).T @ Lambda0)
        FAR = c_coef * (np.linalg.solve(Sigma_hat.T, Rmean_betalambda).T @ Rmean_betalambda)
        GLSLM = c_coef * np.linalg.solve(Sigma_hat.T, Rmean_betalambda).T @ np.linalg.solve((np.linalg.solve(Sigma_hat.T, B_tilde).T @ B_tilde).T,B_tilde.T).T @ np.linalg.solve(Sigma_hat.T, B_tilde).T @ Rmean_betalambda
        JGLS = FAR - GLSLM
        
        
        ans['Test'] = (T-N)/(N-K)/(T-K-1) * JGLS
        ans['pval'] = 1 - sps.f.cdf(ans['Test'],N-K,T-K-N) # p-value
    
    #------------------------------------------------------------------------------
    # Method 4.14) Asymptotic JGLS test 
    #------------------------------------------------------------------------------
    
    if method == 4.14:
        if excess_ret == 0:
            R = R @ np.append(np.eye(N-1), (-1)*np.ones((N-1,1)), axis = 1).T # subtract nth asset return from other returns to come up with excess returns
            N = N-1
        
        R_demean = R - np.ones((T,1)) @ np.mean(R, axis = 0)[:, None].T
        f_demean = f - np.ones((T,1)) @ np.mean(f, axis = 0)[:, None].T
        f_pluslambda = f_demean + np.ones((T,1)) @ Lambda0.T
        f_pluslambda_2 = f_pluslambda.T @ f_pluslambda
        B_tilde = R.T @ np.linalg.solve(f_pluslambda_2.T, f_pluslambda.T).T
        B_hat = R_demean.T @ np.linalg.solve((f_demean.T @ f_demean).T, f_demean.T).T
        Sigma_hat = (R_demean - f_demean @ B_hat.T).T @ (R_demean - f_demean @ B_hat.T) / (T-K-1)
        Rmean_betalambda = np.mean(R, axis = 0)[:,None] - B_tilde @ Lambda0
        c_coef = T / (1 - np.linalg.solve((f_pluslambda_2/T).T, Lambda0).T @ Lambda0)
        FAR = c_coef * (np.linalg.solve(Sigma_hat.T, Rmean_betalambda).T @ Rmean_betalambda)
        GLSLM = c_coef * np.linalg.solve(Sigma_hat.T, Rmean_betalambda).T @ np.linalg.solve((np.linalg.solve(Sigma_hat.T, B_tilde).T @ B_tilde).T,B_tilde.T).T @ np.linalg.solve(Sigma_hat.T, B_tilde).T @ Rmean_betalambda
        JGLS = FAR - GLSLM
        
        ans['Test'] = JGLS
        ans['pval'] = 1 - sps.chi2.cdf(ans['Test'],N-K) # p-value 
    
    #------------------------------------------------------------------------------
    # Method 4.15) FMLM test (under normality assumption)
    #------------------------------------------------------------------------------
    
    if method == 4.15:
        if excess_ret == 0:
            R = R @ np.append(np.eye(N-1), (-1)*np.ones((N-1,1)), axis = 1).T # subtract nth asset return from other returns to come up with excess returns
            N = N-1
        
        R_demean = R - np.ones((T,1)) @ np.mean(R, axis = 0)[:, None].T
        f_demean = f - np.ones((T,1)) @ np.mean(f, axis = 0)[:, None].T
        f_pluslambda = f_demean + np.ones((T,1)) @ Lambda0.T
        f_pluslambda_2 = f_pluslambda.T @ f_pluslambda
        B_tilde = R.T @ np.linalg.solve(f_pluslambda_2.T, f_pluslambda.T).T
        B_hat = R_demean.T @ np.linalg.solve((f_demean.T @ f_demean).T, f_demean.T).T
        Sigma_hat = (R_demean - f_demean @ B_hat.T).T @ (R_demean - f_demean @ B_hat.T) / (T-K-1)
        Rmean_betalambda = np.mean(R, axis = 0)[:,None] - B_tilde @ Lambda0
        c_coef = T / (1 - np.linalg.solve((f_pluslambda_2/T).T, Lambda0).T @ Lambda0)
        FMLM = c_coef * Rmean_betalambda.T @ np.linalg.solve((B_tilde.T @ Sigma_hat @ B_tilde).T,B_tilde.T).T @ B_tilde.T @ Rmean_betalambda
        
        
        ans['Test'] = (T-2*K)/K/(T-K-1) * FMLM
        ans['pval'] = 1 - sps.f.cdf(ans['Test'],K,T-2*K) # p-value
        
    #------------------------------------------------------------------------------
    # Method 4.16) Asymptotic FMLM test 
    #------------------------------------------------------------------------------
    
    if method == 4.16:
        if excess_ret == 0:
            R = R @ np.append(np.eye(N-1), (-1)*np.ones((N-1,1)), axis = 1).T # subtract nth asset return from other returns to come up with excess returns
            N = N-1
        
        R_demean = R - np.ones((T,1)) @ np.mean(R, axis = 0)[:, None].T
        f_demean = f - np.ones((T,1)) @ np.mean(f, axis = 0)[:, None].T
        f_pluslambda = f_demean + np.ones((T,1)) @ Lambda0.T
        f_pluslambda_2 = f_pluslambda.T @ f_pluslambda
        B_tilde = R.T @ np.linalg.solve(f_pluslambda_2.T, f_pluslambda.T).T
        B_hat = R_demean.T @ np.linalg.solve((f_demean.T @ f_demean).T, f_demean.T).T
        Sigma_hat = (R_demean - f_demean @ B_hat.T).T @ (R_demean - f_demean @ B_hat.T) / (T-K-1)
        Rmean_betalambda = np.mean(R, axis = 0)[:,None] - B_tilde @ Lambda0
        c_coef = T / (1 - np.linalg.solve((f_pluslambda_2/T).T, Lambda0).T @ Lambda0)
        FMLM = c_coef * Rmean_betalambda.T @ np.linalg.solve((B_tilde.T @ Sigma_hat @ B_tilde).T,B_tilde.T).T @ B_tilde.T @ Rmean_betalambda
        
        ans['Test'] = FMLM
        ans['pval'] = 1 - sps.chi2.cdf(ans['Test'],K) # p-value
        
    #------------------------------------------------------------------------------
    # Method 4.17) JFM test (under normality assumption)
    #------------------------------------------------------------------------------
    
    if method == 4.17:
        if excess_ret == 0:
            R = R @ np.append(np.eye(N-1), (-1)*np.ones((N-1,1)), axis = 1).T # subtract nth asset return from other returns to come up with excess returns
            N = N-1
        
        R_demean = R - np.ones((T,1)) @ np.mean(R, axis = 0)[:, None].T
        f_demean = f - np.ones((T,1)) @ np.mean(f, axis = 0)[:, None].T
        f_pluslambda = f_demean + np.ones((T,1)) @ Lambda0.T
        f_pluslambda_2 = f_pluslambda.T @ f_pluslambda
        B_tilde = R.T @ np.linalg.solve(f_pluslambda_2.T, f_pluslambda.T).T
        B_hat = R_demean.T @ np.linalg.solve((f_demean.T @ f_demean).T, f_demean.T).T
        Sigma_hat = (R_demean - f_demean @ B_hat.T).T @ (R_demean - f_demean @ B_hat.T) / (T-K-1)
        Rmean_betalambda = np.mean(R, axis = 0)[:,None] - B_tilde @ Lambda0
        c_coef = T / (1 - np.linalg.solve((f_pluslambda_2/T).T, Lambda0).T @ Lambda0)
        FAR = c_coef * (np.linalg.solve(Sigma_hat.T, Rmean_betalambda).T @ Rmean_betalambda)
        FMLM = c_coef * Rmean_betalambda.T @ np.linalg.solve((B_tilde.T @ Sigma_hat @ B_tilde).T,B_tilde.T).T @ B_tilde.T @ Rmean_betalambda
        JFM = FAR - FMLM
        
        # Finite sample distribution of GLSLM
        m = 20000
        v_JFM = np.full((m,1), np.NaN)
        C = np.eye(N,K)
        rng = np.random.default_rng(seed = 1)
        
        for i in range(0,m):
            psi = rng.multivariate_normal(np.zeros(N), np.eye(N))[:,None].T
            WZ_i = rng.multivariate_normal(np.zeros(N), np.eye(N),T-K-1)
            W = WZ_i.T @ WZ_i / (T-K-1)
            PSI = np.reshape(psi.T,N,'F')[:,None]
            v_JFM[i] = np.linalg.solve(W.T, PSI).T @ PSI - PSI.T @ np.linalg.solve((C.T @ W @ C).T, C.T).T @ C.T @ PSI
        
        v_JFM = np.sort(v_JFM)
        index_pos = np.argwhere(v_JFM >= JFM)
        
        if np.size(index_pos, axis = 0) == 0:
            index_pos = m
        else:
            index_pos = index_pos[0,1]
            
        ans['Test'] = JFM
        ans['pval'] = 1-index_pos/m # p-value
            
    #------------------------------------------------------------------------------
    # Method 4.18) Asymptotic JFM test 
    #------------------------------------------------------------------------------
    
    if method == 4.18:
        if excess_ret == 0:
            R = R @ np.append(np.eye(N-1), (-1)*np.ones((N-1,1)), axis = 1).T # subtract nth asset return from other returns to come up with excess returns
            N = N-1
        
        R_demean = R - np.ones((T,1)) @ np.mean(R, axis = 0)[:, None].T
        f_demean = f - np.ones((T,1)) @ np.mean(f, axis = 0)[:, None].T
        f_pluslambda = f_demean + np.ones((T,1)) @ Lambda0.T
        f_pluslambda_2 = f_pluslambda.T @ f_pluslambda
        B_tilde = R.T @ np.linalg.solve(f_pluslambda_2.T, f_pluslambda.T).T
        B_hat = R_demean.T @ np.linalg.solve((f_demean.T @ f_demean).T, f_demean.T).T
        Sigma_hat = (R_demean - f_demean @ B_hat.T).T @ (R_demean - f_demean @ B_hat.T) / (T-K-1)
        Rmean_betalambda = np.mean(R, axis = 0)[:,None] - B_tilde @ Lambda0
        c_coef = T / (1 - np.linalg.solve((f_pluslambda_2/T).T, Lambda0).T @ Lambda0)
        FAR = c_coef * (np.linalg.solve(Sigma_hat.T, Rmean_betalambda).T @ Rmean_betalambda)
        FMLM = c_coef * Rmean_betalambda.T @ np.linalg.solve((B_tilde.T @ Sigma_hat @ B_tilde).T,B_tilde.T).T @ B_tilde.T @ Rmean_betalambda
        JFM = FAR - FMLM
        
        ans['Test'] = JFM
        ans['pval'] = 1 - sps.chi2.cdf(ans['Test'],N-K) # p-value
        
    #------------------------------------------------------------------------------
    # Method 4.19) Hotelling (H) test (under normality assumption)
    #------------------------------------------------------------------------------
    
    if method == 4.19:
        if excess_ret == 0:
            R = R @ np.append(np.eye(N-1), (-1)*np.ones((N-1,1)), axis = 1).T # subtract nth asset return from other returns to come up with excess returns
            N = N-1
        
        R_demean = R - np.ones((T,1)) @ np.mean(R, axis = 0)[:, None].T
        f_demean = f - np.ones((T,1)) @ np.mean(f, axis = 0)[:, None].T
        f_2 = f_demean.T @ f_demean
        B_hat = R_demean.T @ np.linalg.solve((f_demean.T @ f_demean).T, f_demean.T).T
        Sigma_hat = (R_demean - f_demean @ B_hat.T).T @ (R_demean - f_demean @ B_hat.T) / (T-K-1)
        Rmean_betalambda = np.mean(R, axis = 0)[:,None] - B_hat @ Lambda0
        c_coef = T / (1 + np.linalg.solve((f_2/T).T, Lambda0).T @ Lambda0) # Check  
        H = c_coef * (np.linalg.solve(Sigma_hat.T, Rmean_betalambda).T @ Rmean_betalambda)
        
        ans['Test'] = (T-K-N)/N/(T-K-1) * H
        ans['pval'] = 1 - sps.f.cdf(ans['Test'],N,T-K-N) # p-value  I AM NOT SURE IF THIS F TEST MAKES SENSE!!!
    
    #------------------------------------------------------------------------------
    # Method 4.20) Asymptotic Hotelling (H) test
    #------------------------------------------------------------------------------
    
    if method == 4.20:
        if excess_ret == 0:
            R = R @ np.append(np.eye(N-1), (-1)*np.ones((N-1,1)), axis = 1).T # subtract nth asset return from other returns to come up with excess returns
            N = N-1
        
        R_demean = R - np.ones((T,1)) @ np.mean(R, axis = 0)[:, None].T
        f_demean = f - np.ones((T,1)) @ np.mean(f, axis = 0)[:, None].T
        f_2 = f_demean.T @ f_demean
        B_hat = R_demean.T @ np.linalg.solve((f_demean.T @ f_demean).T, f_demean.T).T
        Sigma_hat = (R_demean - f_demean @ B_hat.T).T @ (R_demean - f_demean @ B_hat.T) / (T-K-1)
        Rmean_betalambda = np.mean(R, axis = 0)[:,None] - B_hat @ Lambda0
        c_coef = T / (1 + np.linalg.solve((f_2/T).T, Lambda0).T @ Lambda0) # Check  
        H = c_coef * (np.linalg.solve(Sigma_hat.T, Rmean_betalambda).T @ Rmean_betalambda)
        
        ans['Test'] = H
        ans['pval'] = 1 - sps.chi2.cdf(ans['Test'],N) # p-value
    
    #------------------------------------------------------------------------------
    # Method 4.21) Bayesian FMB confidence bounds and p-values for H0: R2<=0
    #              (Bryzgalova/Huang/Julliard, 2020)
    #------------------------------------------------------------------------------
    
    if method == 4.21:
        sim_length = 500;
        CI_bound = 0.95;
        
        f_demean = f - np.mean(f, axis=0);
        R2_dist = np.zeros((sim_length,1))
        x = np.append(np.ones((T,1)), f_demean, axis = 1)
        xx_inv = np.linalg.pinv(x.T @ x)
        B_ols = xx_inv @ (x.T @ R)
        e = R - x @ B_ols
        Sigma_ols = (e.T @ e)/T
        
        for i in range(0, sim_length):
            Sigma = sps.invwishart.rvs(T-K-1, T * Sigma_ols)
            Var_B = np.kron(Sigma, xx_inv)
            B_ols_vec = B_ols.flatten('F')
            rng = np.random.default_rng()
            B_vec = rng.multivariate_normal(B_ols_vec.T, Var_B)
            B_i = np.reshape(B_vec, (K+1,N), order ='F')
            a_i = B_i[0,:]
            beta_i = B_i[1:,:].T
            Lambda_i = np.linalg.pinv(beta_i.T @ beta_i) @ (beta_i.T @ a_i)
            R2_dist[i,:] = 1 - ((a_i - beta_i @ Lambda_i).T @ (a_i - beta_i @ Lambda_i)) / ((N-1) * np.var(a_i, ddof = 1))
        
        ans['R2_dist'] = R2_dist
        ans['CI'] = np.quantile(R2_dist,((1-CI_bound)/2, 1-(1-CI_bound)/2), axis=0)
        ans['pval'] = np.mean(R2_dist <= 0, axis = 0)
    
    #------------------------------------------------------------------------------
    # SECTION 5)    MODEL TESTS assuming an intercept in cross-sectional
    #               regression 
    #------------------------------------------------------------------------------
    
    #------------------------------------------------------------------------------
    # Method 5.05)  CHI^2 test on pricing errors (OLS, no Shanken correction)
    #------------------------------------------------------------------------------
    
    if method == 5.05:
        Mx = np.eye(N) - np.linalg.solve((X.T @ X).T, X.T).T @ X.T
        ans['Test'] = T * PE[:,None].T @ np.linalg.pinv(Mx @ S @ Mx) @ PE[:,None] # Chi square test on pricing errors
        ans['pval'] = 1 - sps.chi2.cdf(ans['Test'],N-K)
    
    #------------------------------------------------------------------------------
    # Method 5.06)  CHI^2 test on pricing errors including Shanken correction
    #------------------------------------------------------------------------------
    
    if method == 5.06:
        Shanken = 1  + (np.linalg.solve(np.cov(f.T, ddof = 0).T, Lambda).T @ Lambda)
        Mx = np.eye(N) - np.linalg.solve((X.T @ X).T, X.T).T @ X.T
        ans['Test'] = T * PE[:,None].T @ np.linalg.pinv(Shanken * Mx @ S @ Mx) @ PE[:,None] # Chi square test on pricing errors (test stat)
        ans['pval'] = 1 - sps.chi2.cdf(ans['Test'],N-K)
        
    #------------------------------------------------------------------------------
    # Method 5.07)  CHI^2 test on pricing errors using GMM standard errors (Newey-West)
    #------------------------------------------------------------------------------
    
    if method == 5.07:
        # GMM Moments
        for i in range(0, np.size(e, axis=1)):
            if i == 0:
                g = np.matlib.repmat(e[:, i], K+1, 1).T * F 
            else:   
                g = np.append(g, np.matlib.repmat(e[:, i], K+1, 1).T * F, axis = 1)
                
        g_cs  = R - np.matlib.repmat((beta @ Lambda + const).T, T,1) # time series of cross section sample moments
        g_GMM = np.append(g, g_cs, axis = 1) # T x n*(k+2) -> time series of sample moments
        
        # GMM Standard Errors, Newey West
        Scs_hac = nw.nw_func(g_GMM, lags)
        
        aT = np.append(np.append(np.eye((N*(K+1))), np.zeros((N*(K+1), N)), axis = 1), np.append(np.zeros((K+1, N*(K+1))), X.T, axis = 1), axis = 0)
        Mf = F.T @ (F/T)
        dT = np.append(np.append((-1)*np.kron(np.eye(N), Mf),np.zeros((N*(K+1), K+1)),axis = 1),np.append((-1)*np.kron(np.eye(N),np.insert(Lambda,0,0)),(-1)* X,axis = 1), axis = 0)
        
        Vgt_hac = (np.eye(N*(K+2)) - np.linalg.solve((aT @ dT).T, dT.T).T @ aT) @ Scs_hac @ (np.eye(N*(K+2)) - np.linalg.solve((aT @ dT).T, dT.T).T @ aT).T
        ans['Test'] = T * PE[:,None].T @ np.linalg.pinv(Vgt_hac[N*(K+1):,N*(K+1):]) @ PE[:,None]
        ans['pval'] = 1 - sps.chi2.cdf(ans['Test'],N-K)
    
    #------------------------------------------------------------------------------
    # Method 5.08)  CHI^2 test on pricing errors using GMM standard errors (VARHAC)
    #------------------------------------------------------------------------------
    
    if method == 5.08:
        # GMM Moments
        for i in range(0, np.size(e, axis=1)):
            if i == 0:
                g = np.matlib.repmat(e[:, i], K+1, 1).T * F 
            else:   
                g = np.append(g, np.matlib.repmat(e[:, i], K+1, 1).T * F, axis = 1)
                
        g_cs  = R - np.matlib.repmat((beta @ Lambda + Theta[0]).T, T,1) # time series of cross section sample moments
        g_GMM = np.append(g, g_cs, axis = 1) # T x n*(k+2) -> time series of sample moments
        
        # GMM Standard Errors, VARHAC
        Scs_varhac = hac_var.hac_var_func(g_GMM-np.matlib.repmat(np.mean(g_GMM, axis=0),T,1),1,0) # Call VARHAC function from Burnside
        
        aT = np.append(np.append(np.eye((N*(K+1))), np.zeros((N*(K+1), N)), axis = 1), np.append(np.zeros((K+1, N*(K+1))), X.T, axis = 1), axis = 0)
        Mf = F.T @ (F/T)
        dT = np.append(np.append((-1)*np.kron(np.eye(N), Mf),np.zeros((N*(K+1), K+1)),axis = 1),np.append((-1)*np.kron(np.eye(N),np.insert(Lambda,0,0)),(-1)* X,axis = 1), axis = 0)
        
        Vgt_hac = (np.eye(N*(K+2)) - np.linalg.solve((aT @ dT).T, dT.T).T @ aT) @ Scs_varhac @ (np.eye(N*(K+2)) - np.linalg.solve((aT @ dT).T, dT.T).T @ aT).T
        ans['Test'] = T * PE[:,None].T @ np.linalg.pinv(Vgt_hac[N*(K+1):,N*(K+1):]) @ PE[:,None]
        ans['pval'] = 1 - sps.chi2.cdf(ans['Test'],N-K)
        
    #------------------------------------------------------------------------------
    # Method 5.09)  Shanken's (1985) asymptotic CSRT test
    #------------------------------------------------------------------------------
    
    if method == 5.09:
        f_demean = f - np.ones((T,1)) @ np.mean(f, axis = 0)[:, None].T
        WP = spy.linalg.null_space(X.T)
        wt = np.linalg.solve(np.cov(f.T, ddof = 0).T, f_demean.T).T @ Lambda
        yt = 1 - wt
        g = (R @ WP) * (yt[:,None] * np.ones((1,N-K-1)))
        S_krs = nw.nw_func(g,lags)
        e1 = WP.T @ PEi[:,None]
        
        ans['Test'] = np.linalg.solve(S_krs.T,e1).T @ e1
        ans['pval'] = 1 - sps.chi2.cdf(T*ans['Test'],N-K-1)
        
    #------------------------------------------------------------------------------
    # Method 5.10)  Shanken's (1985) finite sample CSRT test
    #------------------------------------------------------------------------------
    
    if method == 5.10:
        f_demean = f - np.ones((T,1)) @ np.mean(f, axis = 0)[:, None].T
        WP = spy.linalg.null_space(X.T)
        wt = np.linalg.solve(np.cov(f.T, ddof = 0).T, f_demean.T).T @ Lambda
        yt = 1 - wt
        g = (R @ WP) * (yt[:,None] * np.ones((1,N-K-1)))
        S_krs = nw.nw_func(g,lags)
        e1 = WP.T @ PEi[:,None]
        
        ans['Test'] = np.linalg.solve(S_krs.T,e1).T @ e1
        ans['pval'] = 1 - sps.f.cdf(ans['Test']*(T-N+1)/(N-K-1),N-K-1,T-N+1)
        
    #------------------------------------------------------------------------------
    # Method 5.11)  Kan/Robotti/Shanken test of hypothesis R^2=1
    #------------------------------------------------------------------------------
    
    if method == 5.11:
        f_demean = f - np.ones((T,1)) @ np.mean(f, axis = 0)[:, None].T
        WP = spy.linalg.null_space(X.T)
        wt = np.linalg.solve(np.cov(f.T, ddof = 0).T, f_demean.T).T @ Lambda
        yt = 1 - wt
        g = (R @ WP) * (yt[:,None] * np.ones((1,N-K-1)))
        S_krs = nw.nw_func(g,lags)
        e0 = R_bar - np.mean(R_bar, axis = 0)
        Q0 = e0.T @ e0
        xi = (-1) * np.linalg.eig(S_krs)[0] / Q0 # S is actually P'W^\frac{1}{2}SW^\frac{1}{2}P
        ans['pval'] = linchi2.linchi2_func(T*(R2i-1),xi) # p-val for specification test
        
    #------------------------------------------------------------------------------
    # Method 5.12)  Kan/Robotti/Shanken test of hypothesis R^2=0 
    #               (imposing H0: Lambda=0_K when estimating covariance matrix)
    #------------------------------------------------------------------------------
    
    if method == 5.12:
        V = np.cov(f.T, ddof=0)
        R_demean = R - np.ones((T,1)) @ np.mean(R, axis = 0)[:, None].T
        f_demean = f - np.ones((T,1)) @ np.mean(f, axis = 0)[:, None].T
        H = np.linalg.inv(X.T @ X)
        Theta_t = R_demean @  np.linalg.solve((X.T @ X).T, X.T).T
        ut = R_demean @ PEi[:,None]
        zt = np.append(np.zeros((T,1)), np.linalg.solve(V.T,f_demean.T).T, axis = 1)
        ht0 = Theta_t + np.linalg.solve((X.T @ X).T, zt.T).T * (ut @ np.ones((1,K+1)))
        V0 = nw.nw_func(ht0, lags)
        H22i = np.linalg.inv(H[1:K+1,1:K+1])
        H22iq = spl.sqrtm(H22i)
        mat = H22iq @ V0[1:K+1,1:K+1] @ H22iq
        e0 = R_bar - np.mean(R_bar, axis = 0)
        Q0 = e0.T @ e0
        xi = np.sort(np.linalg.eig((0.5/Q0) * (mat + mat.T))[0])
        ans['pval'] = 1-linchi2.linchi2_func(T*R2i,xi)     
    
    #------------------------------------------------------------------------------
    # Method 5.13)  Kan/Robotti/Shanken test of hypothesis R^2=0 
    #               (without imposing H0: Lambda=0_K when estimating covariance matrix)
    #------------------------------------------------------------------------------
    
    if method == 5.13:
        V = np.cov(f.T, ddof=0)
        R_demean = R - np.ones((T,1)) @ np.mean(R, axis = 0)[:, None].T
        f_demean = f - np.ones((T,1)) @ np.mean(f, axis = 0)[:, None].T
        H = np.linalg.inv(X.T @ X)
        Theta_t = R_demean @  np.linalg.solve((X.T @ X).T, X.T).T
        ut = R_demean @ PEi[:,None]
        zt = np.append(np.zeros((T,1)), np.linalg.solve(V.T,f_demean.T).T, axis = 1)
        wt = np.linalg.solve(V.T, f_demean.T)[:,None].T @ Lambda
        phi_t = Theta_t - np.append(np.zeros((T,1)), f_demean, axis = 1)
        ht0 = Theta_t + np.linalg.solve((X.T @ X).T, zt.T).T * (ut @ np.ones((1,K+1)))
        ht2 = ht0 - phi_t * (wt @ np.ones((1,K+1))) # With misspecification adjustment
        V2 = nw.nw_func(ht2, lags)
        H22i = np.linalg.inv(H[1:K+1,1:K+1])
        H22iq = spl.sqrtm(H22i)
        mat = H22iq @ V2[1:K+1,1:K+1] @ H22iq
        e0 = R_bar - np.mean(R_bar, axis = 0)
        Q0 = e0.T @ e0
        xi = np.sort(np.linalg.eig((0.5/Q0) * (mat + mat.T))[0])
        ans['pval'] = 1-linchi2.linchi2_func(T*R2i,xi)
    
    #------------------------------------------------------------------------------
    # Method 5.14)  Standard error of sample R2 (Kan/Robotti/Shanken)
    #------------------------------------------------------------------------------
    
    if method == 5.14:
        V = np.cov(f.T, ddof=0)
        R_demean = R - np.ones((T,1)) @ np.mean(R, axis = 0)[:, None].T
        f_demean = f - np.ones((T,1)) @ np.mean(f, axis = 0)[:, None].T
        ut = R_demean @ (PE - const)
        wt = np.linalg.solve(V.T, f_demean.T)[:,None].T @ Lambda
        yt = 1 - wt
        e0 = R_bar - np.mean(R_bar, axis = 0)
        Q0 = e0.T @ e0
        vt = R_demean @ e0
        nt = 2 * (-ut[:,None] * yt + (1-R2i) * vt[:,None]) / Q0
        vn = nw.nw_func(nt, lags)
        ans['SE'] = np.sqrt(vn/T)
           
    #------------------------------------------------------------------------------
    # Method 5.15)  Wald test of H0: Lambda=0_K (Kan/Robotti/Shanken)
    #               (imposing H0 when estimating covariance matrix)
    #------------------------------------------------------------------------------
    
    if method == 5.15:
        V = np.cov(f.T, ddof=0)
        R_demean = R - np.ones((T,1)) @ np.mean(R, axis = 0)[:, None].T
        f_demean = f - np.ones((T,1)) @ np.mean(f, axis = 0)[:, None].T
        Theta_t = R_demean @  np.linalg.solve((X.T @ X).T, X.T).T
        ut = R_demean @ PEi[:,None]
        zt = np.append(np.zeros((T,1)), np.linalg.solve(V.T,f_demean.T).T, axis = 1)
        ht0 = Theta_t + np.linalg.solve((X.T @ X).T, zt.T).T * (ut @ np.ones((1,K+1)))
        V0 = nw.nw_func(ht0, lags)
        ans['Test'] = T * np.linalg.solve(V0[1:K+1,1:K+1].T, Lambda).T @ Lambda
        ans['pval'] = 1 - sps.chi2.cdf(ans['Test'],K)    
        
    #------------------------------------------------------------------------------
    # Method 5.16)  Wald test of H0: Lambda=0_K (Kan/Robotti/Shanken)
    #               (without imposing H0 when estimating covariance matrix)
    #------------------------------------------------------------------------------
    
    if method == 5.16:
        V = np.cov(f.T, ddof=0)
        R_demean = R - np.ones((T,1)) @ np.mean(R, axis = 0)[:, None].T
        f_demean = f - np.ones((T,1)) @ np.mean(f, axis = 0)[:, None].T
        Theta_t = R_demean @  np.linalg.solve((X.T @ X).T, X.T).T
        wt = np.linalg.solve(V.T, f_demean.T)[:,None].T @ Lambda
        phi_t = Theta_t - np.append(np.zeros((T,1)), f_demean, axis = 1)
        ht1 = Theta_t - phi_t * (wt @ np.ones((1,K+1)))  # Without misspecification adjustment
        V1 = nw.nw_func(ht1, lags)
        ans['Test'] = T * np.linalg.solve(V1[1:K+1,1:K+1].T, Lambda).T @ Lambda
        ans['pval'] = 1 - sps.chi2.cdf(ans['Test'],K)
        
    #------------------------------------------------------------------------------
    # Method 5.17)  Bayesian FMB confidence bounds and p-values for R2
    #               (Bryzgalova/Huang/Julliard, 2020)
    #------------------------------------------------------------------------------
    
    if method == 5.17:
        sim_length = 10000;
        CI_bound = 0.95;
        
        f_demean = f - np.mean(f, axis=0);
        R2_dist = np.zeros((sim_length,1))
        x = np.append(np.ones((T,1)), f_demean, axis = 1)
        xx_inv = np.linalg.pinv(x.T @ x)
        B_ols = xx_inv @ (x.T @ R)
        e = R - x @ B_ols
        Sigma_ols = (e.T @ e)/T
        
        for i in range(0, sim_length):
            Sigma = sps.invwishart.rvs(T-K-1, T * Sigma_ols)
            Var_B = np.kron(Sigma, xx_inv)
            B_ols_vec = B_ols.flatten('F')
            rng = np.random.default_rng()
            B_vec = rng.multivariate_normal(B_ols_vec.T, Var_B)
            B_i = np.reshape(B_vec, (K+1,N), order ='F')
            a_i = B_i[0,:]
            beta_i = B_i[1:,:].T
            H = np.append(np.ones((N,1)), beta_i, axis = 1)
            Theta = np.linalg.pinv(H.T @ H) @ (H.T @ a_i)
            Lambda = Theta[1:]
            R2_dist[i,:] = 1 - ((a_i - beta_i @ Lambda).T @ (a_i - beta_i @ Lambda)) / ((N-1) * np.var(a_i, ddof = 1))
        
        ans['R2_dist'] = R2_dist
        ans['CI'] = np.quantile(R2_dist,((1-CI_bound)/2, 1-(1-CI_bound)/2), axis=0)
        ans['pval'] = np.mean(R2_dist <= 0, axis = 0)
        
    #------------------------------------------------------------------------------
    # Method 5.18)  Bayesian FMB confidence bounds for R2i
    #               (Bryzgalova/Huang/Julliard, 2020)
    #------------------------------------------------------------------------------
    
    if method == 5.18:
        sim_length = 10000;
        CI_bound = 0.95;
        
        f_demean = f - np.mean(f, axis=0);
        R2_dist = np.zeros((sim_length,1))
        x = np.append(np.ones((T,1)), f_demean, axis = 1)
        xx_inv = np.linalg.pinv(x.T @ x)
        B_ols = xx_inv @ (x.T @ R)
        e = R - x @ B_ols
        Sigma_ols = (e.T @ e)/T
        
        for i in range(0, sim_length):
            Sigma = sps.invwishart.rvs(T-K-1, T * Sigma_ols)
            Var_B = np.kron(Sigma, xx_inv)
            B_ols_vec = B_ols.flatten('F')
            rng = np.random.default_rng()
            B_vec = rng.multivariate_normal(B_ols_vec.T, Var_B)
            B_i = np.reshape(B_vec, (K+1,N), order ='F')
            a_i = B_i[0,:]
            beta_i = B_i[1:,:].T
            H = np.append(np.ones((N,1)), beta_i, axis = 1)
            Theta = np.linalg.pinv(H.T @ H) @ (H.T @ a_i)
            R2_dist[i,:] = 1 - ((a_i - H @ Theta).T @ (a_i - H @ Theta)) / ((N-1) * np.var(a_i, ddof = 1))
        
        ans['R2_dist'] = R2_dist
        ans['CI'] = np.quantile(R2_dist,((1-CI_bound)/2, 1-(1-CI_bound)/2), axis=0)
    
    #------------------------------------------------------------------------------
    # SECTION 6)    SIGNIFICANCE TESTS FOR LINEAR SDF SPECIFICATIONS 
    #               (SDF LOADINGS) 
    #------------------------------------------------------------------------------
    
    #------------------------------------------------------------------------------
    # Method 6.01)  Test of the SDF:  m=1-b'(f-E(f)), W=1, without common 
    #               pricing error (Cochrane (2005), Chapter 13.2)
    #------------------------------------------------------------------------------
    
    if method == 6.01:
        
        # estimate the vector of factor loadings, WT=eye(N,N);
        dT = ((R.T @ f / T) - (R_bar[:,None] @ np.mean(f, axis = 0)[:,None].T))
        b = np.linalg.solve((dT.T @ dT), dT.T) @ R_bar[:,None] # factor loadings
        
        # time-series of moments
        m = 1 - (f - np.matlib.repmat(np.mean(f, axis = 0), T, 1)) @ b # SDF
        u1 = R * (m @ np.ones((1,N)))                                  # asset pricing equations
        u2 = f - np.matlib.repmat(np.mean(f, axis = 0), T, 1)          # means
        
        if f.shape[1] == 1:
            Sf = np.expand_dims(np.cov(f.T, ddof=0), axis=(0,1))
        else:
            Sf = np.cov(f.T, ddof=0)   # BUGFIX: "Sf =" was missing in the original,
                                       # so 6.01/6.02 raised UnboundLocalError for K>1.
                                       # Matches the authors' own pattern in 3.02/3.04.
        k = int(K*(K+1) / 2)
        a = 0
        u3 = np.zeros((T,k))
        
        for i in range(0,k):
            for j in range(i,K):
                u3[:,a][:,None] = (f[:,i][:,None] - np.matlib.repmat(np.mean(f[:,i], axis = 0), T, 1)) * (f[:,j][:,None] - np.matlib.repmat(np.mean(f[:,j], axis = 0),T,1)) - Sf[i,j]
                a = a + 1
        u = np.concatenate((u1,u2,u3), axis = 1)
        S = nw.nw_func(u, lags)
        gT = np.mean(u1, axis=0)[:,None]
        
        # GMM Standard Errors
        aT = np.concatenate((np.append(dT, np.zeros((N,K)), axis=1),np.append(np.zeros((K,K)),np.eye(K), axis = 1)),axis = 0)
        delT = np.concatenate((np.append(dT, -R_bar[:,None] @ b.T, axis=1),np.append(np.zeros((K,K)),np.eye(K), axis = 1)),axis = 0)
        Vb = np.linalg.solve(aT.T @ delT, aT.T) @ S[0:(N+K),0:(N+K)] @ np.linalg.solve((aT.T @ delT).T, aT.T).T / T
        bSd = np.sqrt(np.diag(Vb[0:K,0:K]))[:,None]
        
        # J-test
        Md = np.eye(N) - np.linalg.solve((dT.T @ dT).T, dT.T).T @ dT.T
        Vu = Md @ S[0:N,0:N] @ Md.T
        
        ans['J'] = T * gT.T @ np.linalg.pinv(Vu) @ gT
        ans['b'] = b
        ans['T'] = (b / bSd)[:,-1]
        ans['pval'] = 1 - sps.t.cdf(np.abs(ans['T']),T-K)
        ans['J_pv'] = 1 - sps.chi2.cdf(ans['J'],N-K)
            
    #------------------------------------------------------------------------------
    # Method 6.02)  Test of the SDF:  m=1-b'(f-E(f)), W=1, with common pricing error 
    #               Burnside(2011)
    #------------------------------------------------------------------------------
    
    if method == 6.02:
        
        # estimate the vector of factor loadings
        WT = np.eye(N)
        dT = np.append(np.ones((N,1)),((R.T @ f / T) - (R_bar[:,None] @ np.mean(f, axis = 0)[:,None].T)), axis = 1)
        b = np.linalg.solve((dT.T @ dT), dT.T) @ WT @ R_bar[:,None] # factor loadings
        
        # time-series of moments
        m = 1 - (f - np.matlib.repmat(np.mean(f, axis = 0), T, 1)) @ b[1:,:] # SDF
        u1 = R * (m @ np.ones((1,N))) - b[0,0]                               # asset pricing equations
        u2 = f - np.matlib.repmat(np.mean(f, axis = 0), T, 1)                # means
        
        if f.shape[1] == 1:
            Sf = np.expand_dims(np.cov(f.T, ddof=0), axis=(0,1))
        else:
            Sf = np.cov(f.T, ddof=0)   # BUGFIX: "Sf =" was missing in the original,
                                       # so 6.01/6.02 raised UnboundLocalError for K>1.
                                       # Matches the authors' own pattern in 3.02/3.04.
        k = int(K*(K+1) / 2)
        a = 0
        u3 = np.zeros((T,k))
        
        for i in range(0,k):
            for j in range(i,K):
                u3[:,a][:,None] = (f[:,i][:,None] - np.matlib.repmat(np.mean(f[:,i], axis = 0), T, 1)) * (f[:,j][:,None] - np.matlib.repmat(np.mean(f[:,j], axis = 0),T,1)) - Sf[i,j]
                a = a + 1
        u = np.concatenate((u1,u2,u3), axis = 1)
        S = nw.nw_func(u, lags)
        
        # GMM Standard Errors
        aT = np.concatenate((np.append(WT @ dT, np.zeros((N,K)), axis=1),np.append(np.zeros((K,K+1)),np.eye(K), axis = 1)),axis = 0)
        delT = np.concatenate((np.append(dT, -R_bar[:,None] @ b[1:,:].T, axis=1),np.append(np.zeros((K,K+1)),np.eye(K), axis = 1)),axis = 0)
        B1 = np.linalg.solve(aT.T @ delT, aT.T)
        Vb = B1 @ S[0:(N+K),0:(N+K)] @ B1.T / T
        bSd = np.sqrt(np.diag(Vb[0:K+1,0:K+1]))[:,None]
        
        # J-test
        Md = np.eye(N) - np.linalg.solve((dT.T @ dT).T, dT.T).T @ dT.T
        VT = Md @ S[0:N,0:N] @ Md.T
        
        ans['J'] = T * R_bar[:,None].T @ Md.T @ np.linalg.pinv(VT) @ Md @ R_bar[:,None]
        ans['b'] = b
        ans['T'] = b / bSd
        ans['pval'] = 1 - sps.t.cdf(np.abs(ans['T'][1:,0]),T-K)
        ans['J_pv'] = 1 - sps.chi2.cdf(ans['J'],N-K)
    
    #------------------------------------------------------------------------------
    # Method 6.03)  Test of the SDF:  m=1-b'(f-E(f)),  W=S^-1, without common 
    #               pricing error, Gospodinov, Kan, and Robotti (2014)
    #------------------------------------------------------------------------------
    
    if method == 6.03:
        Rd = R - np.ones((T,1)) @ R_bar[:,None].T
        V22 = np.cov(R.T, ddof =0)
        W = np.linalg.inv(V22) # weighting matrix S^-1
        # W = np.eye(N) # TEST
        Fd = f - np.ones((T,1)) @ np.mean(f, axis = 0)[:,None].T
        V21 = (Rd.T @ f) / T
        V12 = V21.T
        H = np.linalg.inv(V12 @ W @ V21)
        ans['b'] = H @ V12 @ W @ R_bar[:,None]
        y = 1 - Fd @ ans['b']
        
        # inference for correctly specified models
        hc = (Rd * (y @ np.ones((1,N)))) @ W @ V21 @ H + np.ones((T,1)) @ ans['b'].T
        Vgamc = np.array((hc.T @ hc) / T, dtype = complex)
        # ans['T'] = np.sqrt(T) * np.linalg.solve(np.sqrt(Vgamc).T, ans['b'].T).T # Original Code
        ans['T'] = np.sqrt(T) * np.linalg.solve(np.sqrt(Vgamc).T, ans['b']).T # Corrected Code without dimension error in MATLAB
        ans['pval'] = 1 - sps.t.cdf(np.abs(ans['T'][:,0]),T-K)
        
    #------------------------------------------------------------------------------
    # Method 6.04)    Robust test of the SDF:  m=1-b'(f-E(f)),  W=S^-1, 
    #                 without common pricing error, 
    #                 Gospodinov, Kan, and Robotti (2014)
    #------------------------------------------------------------------------------
    
    if method == 6.04:
        Rd = R - np.ones((T,1)) @ R_bar[:,None].T
        V22 = np.cov(R.T, ddof =0)
        W = np.linalg.inv(V22) # weighting matrix S^-1
        Fd = f - np.ones((T,1)) @ np.mean(f, axis = 0)[:,None].T
        V21 = (Rd.T @ f) / T
        V12 = V21.T
        H = np.linalg.inv(V12 @ W @ V21)
        ans['b'] = H @ V12 @ W @ R_bar[:,None]
        ge = R_bar[:,None] - V21 @ ans['b']
        y = 1 - Fd @ ans['b']
        
        # inference for correctly specified models
        lam = W @ ge
        u = Rd @ lam
        h = (Rd * (y @ np.ones((1,N)))) @ W @ V21 @ H + np.ones((T,1)) @ ans['b'].T + ((Fd - Rd @ W @ V21) @ H) * (u @ np.ones((1,K)))
        Vgam = np.array((h.T @ h) / T, dtype = complex)
        # ans['T'] = np.sqrt(T) * np.linalg.solve(np.sqrt(Vgam).T, ans['b'].T).T # Original Code
        ans['T'] = np.sqrt(T) * np.linalg.solve(np.sqrt(Vgam).T, ans['b']).T # Corrected Code without dimension error in MATLAB
        ans['pval'] = 1 - sps.t.cdf(np.abs(ans['T'][:,0]),T-K)
        
    return ans

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
