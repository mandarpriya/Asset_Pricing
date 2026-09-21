import scipy.special as sp

def cdfchic_func(x, d):
    return 1-sp.gammainc(d/2, x/2)[0]
    
    


