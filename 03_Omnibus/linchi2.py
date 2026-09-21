# This Python program computes the probability that a linear combination
# of central chi-squared distribution (with 1 d.f.) is less than a constant c.
# d0: weights of the linear combination
# e: accuracy (1e-6 by default)
# delta: step size
# U: upper truncation point
# original code in MATLAB
import numpy as np
import scipy.stats as sps
import scipy.optimize as spo

def linchi2_func(c, d0, e = 10 ** (-6)):

    # BUGFIX: callers pass c as a size-1 array (it comes out of a quadratic
    # form) and d0 as a column. Older SciPy silently coerced these; SciPy/NumPy
    # from 2024 onward raise "RuntimeError: Unable to parse arguments" once the
    # array reaches spo.brentq / spo.fsolve, which need scalar bounds.
    # Squeezing to a true scalar / 1-D here changes no arithmetic.
    c = float(np.squeeze(np.asarray(c)))
    d0 = np.asarray(d0).ravel()

    if all(d0 >= 0) and (c <= 0):
        return 0
    
    elif all(d0 <= 0) and  (c >= 0):
        return 1
    
    else:
        # Sorting and normalization
        d = np.sort(d0)
        d = d[d != 0] # Get rid of zero elements
        n = len(d)-1
        
        if d[0] == d[n]: # Use chi-squared distribution when d(1)=d(n)
        
            if d[0] > 0:
                return sps.chi2.cdf(c/d[0],n+1)
            else:
                return 1 - sps.chi2.cdf(c/d[0],n+1)
            
        elif c ==  0: # Use F-distribution for this special case
            n1 = np.sum(d == d[0])
            n2 = np.sum(d == d[n])
            
            if n1+n2 == n+1:
                return sps.f.cdf(-d[0]*n1/(d[n]*n2),n1,n2)
        
        elif (n+1) == 2 and np.prod(d) >= 0: # Use non-central chi-squared
        
            if d[0] > 0:
                d1 = np.sum(np.sqrt(d)) ** 2 * c / (4*d[0]*d[1])
                d2 = d1 - c / np.sqrt(d[0]*d[1])
                return sps.ncx2.cdf(d1,2,d2) - sps.ncx2.cdf(d2,2,d1)
            
            else:
                d = np.absolute(d)
                c = np.absolute(c)
                d1 = np.sum(np.sqrt(d)) ** 2 * c / (4*d[0]*d[1])
                d2 = d1 - c / np.sqrt(d[0]*d[1])
                return sps.ncx2.cdf(d1,2,d2) - sps.ncx2.cdf(d2,2,d1)
    
            
        else:
            def lin1(t):
                x = 1 - 2*t*d
                psi3 = 0.5 * np.sum(np.log(x))
                psi4 = np.sum(d/x)
                y = psi3 + t*psi4 + ei
                return y
            
            def lin2(u,P1):
                x = 1 + 4*u*u*d2
                y = np.sum(np.log(x)) - np.log(1+(2/(np.pi * P1)) ** 4)
                return y
            
            def lin3(alphai,Ui,v):
                x = 1 + 4*v*v*d2
                y = 0.25*np.sum(np.log(x)) + np.log(1-alphai) + et - np.log(np.log(Ui/v))
                return y
            
            dmax = max(np.abs(d))
            d = d/dmax
            c = c/dmax
            absd = np.absolute(d)
            ei = np.log(0.1*e) #log of integration error
            et = np.log(0.9*e) #log of truncation error
            
            #Choosing delta and K for integration
            psi1 = 0
            psi2 = 0
            # BUGFIX: was np.array(10**(-9), ndmin=2), a (1,1) array (MATLAB
            # translation artifact). It becomes the bound passed to spo.brentq
            # below, and current SciPy rejects a non-scalar bound with
            # "RuntimeError: Unable to parse arguments". A plain float is the
            # same number and keeps the bounds scalar.
            mar = 1e-9
            eps = np.finfo(np.float64).eps
            
            if d[0] < 0:
                t1 = spo.brentq(lin1,1/(2*d[0])+mar,-mar,xtol=eps)
                psi1 = np.sum(d/(1-2*t1*d))
            
            elif d[n] > 0:
                t2 = spo.brentq(lin1,mar,1/(2*d[n])-mar,xtol=eps)
                psi2 = np.sum(d/(1-2*t2*d))
            
            delta = 2 * np.pi * min(1/max(c-psi1,eps),1/max(psi2-c,eps))
            
            #Imhof truncation bound
            sd = np.sum(np.log(absd))
            a = np.exp(-(sd+2*et)/(n+1)) # To avoid overflow problem
            U1 = 0.5 * a * ((2/((n+1)*np.pi)) ** (2/(n+1)))
            
            #AKS truncation bound
            d2 = d*d
            anonymlin2_1 = lambda u: lin2(u, np.exp(et)) 
            U2 = np.float64(np.absolute(spo.fsolve(anonymlin2_1,1, xtol = eps)))
            
            #Lu and King truncation bound
            n1 = (n+1)-0.5
            a = np.exp(-(sd+ 0.25*np.log(np.sum(1/d2)) + 2*et)/n1) # To avoid overflow problem
            U3 = 0.5*a*((2 ** (0.75) / (n1*np.pi)) ** (2/n1))
            u = np.array((U1,U2,U3))
            UU = min(u)
            K = np.ceil(UU/delta - 0.5)
            
            if K > 5000:
                Umin = min(u)
                ii = np.argmin(u)+1
                V = np.zeros(9)
                alpha = np.arange(1,10)/10
                et2 = np.log(alpha)+et
                
                if ii == 1:
                    a = np.exp(-(sd+2*et2)/(n+1)) #To avoid overflow problem
                    U = 0.5*a*((2/((n+1)*np.pi)) ** (2/(n+1)))
                
                elif ii == 2:
                    U = np.zeros(9)
                    anonymlin2_2 = lambda u: lin2(u, np.exp(et2)) 
                    
                    for i in range(0,9):
                        U[i] = np.float64(np.absolute(spo.fsolve(anonymlin2_2,1, xtol = eps)))
                
                elif ii == 3:
                    a = np.exp(-(sd+ 0.25*np.log(np.sum(1/d2)) + 2*et2)/n1) #To avoid overflow problem
                    U3 = 0.5*a*((2 ** (0.75) / (n1*np.pi)) ** (2/n1))
                
                for i in range(0,9):
                    anonymlin3 = lambda v: lin3(alpha[i], U[i], v)
                    V[i] = spo.brentq(anonymlin3,10 **-12*U[i],(1- 10 **-12)*U[i], xtol=eps)
                
                UU = min(Umin, min(V))
                K = np.ceil(UU/delta -0.5)
                
            if K > 3000000:
                raise RuntimeWarning("K = " + str(K) + ", please consider reducing the precision!")
                return np.NaN
            
            else:
                index = np.arange(0.5,K+1.5)
                u = index * delta
                du = 2*d[:,None] @ u[:,None].T
                thetak = 0.5 * np.sum(np.arctan(du),axis=0) - c*u
                gk = np.prod(1+ du*du, axis=0) ** 0.25
                y = 0.5 - np.sum(np.sin(thetak) / (gk * index)) / np.pi
                return min(max(y,0),1)
                
                    
                        
                    
                        
            
            
            
                






    







            
            
            
        
                
                
                
                
                
                
                
                
            
                
            
                