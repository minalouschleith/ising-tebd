from .TensorNetwork import * 
from .tebd import *
import numpy as np

"""Calculating observables like magnetisation and correlation length"""

def expectation_value(psi,operator):
    res = []
    for s in range(psi.Nx): #sites
        psi_site = np.tensordot(np.diag(psi.Lambdas[s]), psi.Bs[s], axes=([1],[0])) #singe-site wavefunction
        tmp = np.tensordot(operator, psi_site, axes = ([1],[1]))
        res.append(np.tensordot(psi_site.conj(), tmp, axes = ([0,1,2],[1,0,2])))
    result = np.sum(res)/psi.Nx
    return np.real_if_close(result) 