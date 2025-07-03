### performing the contraction of the tensor network
import numpy as np
from .TensorNetwork import * 

def Theta(psi,i,j): 
    theta1 = np.tensordot(np.diag(psi.Lambdas[i]), psi.Bs[i], axes=([1],[0])) 
    theta2 = np.tensordot(theta1, psi.Bs[j], axes=([2],[0])) 
    #print("function Theta, Lambda[i]=", psi.Lambdas[i], "B[i]= ", psi.Bs[i], "B[i+1]= ", psi.Bs[j])
    return theta2 

def update_bond(psi,model,chi_m,eps,bond):
    d = 2 
    i,j = sites_for_bond(psi,bond) 
    psi.eps = eps
    psi.chi_m = chi_m
    theta = Theta(psi,i,j) # B(n)*B(n*1) -> vL i* j* vR
    model.site_tensor()
    T = model.tensor 
    theta = np.tensordot(T, theta, axes=([2,3], [1,2])) # U's axes [2,3] with theta's axes [1,2]
    theta = theta/np.linalg.norm(theta)
    assert np.all(np.isfinite(theta)), "theta contains inf or nan"
    theta = np.transpose(theta, [2,0,1,3]) # vL i j vR
    A,Lambda,B = SVD(theta, psi.chi_m, psi.eps) # U1 * S * U2 (U1, U2 = unitaries, S = diagonal, singular values)
    inv_Lambda_i = np.diag(1.0 / (psi.Lambdas[i]+1e-20))
    B_tilde1 = np.tensordot(inv_Lambda_i, A, axes=(1,0))
    psi.Bs[i] = np.tensordot(B_tilde1, np.diag(Lambda), axes=(2,0))  #B_tilde(i)
    psi.Lambdas[j] = Lambda
    psi.Bs[j] = B

"""Time evolution"""

def sites_for_bond(psi, bond):
    """return site indices for bond index"""
    i = bond
    j = (i + 1) if psi.bc == "finite" else (i + 1) % psi.Nx
    return i, j

def norm_squared(psi):
    return np.sum([np.linalg.norm(B)**2 for B in psi.Bs])

def run_TEBD(psi, model):
    N_steps = model.N_steps
    chi_m= model.chi_m
    eps = model.eps
    nbonds = psi.nbonds 
    Mag_tot=0
   
    for step in range(N_steps):
        for k in [0,1]: 
            for bond in range(k, nbonds, 2): # range(start, stop, steps): even and then odd unitaries
                print("step = ", step,"bond= ", bond)
                update_bond(psi, model, chi_m, eps, bond)
                print("norm squared= ", norm_squared(psi))
            mag_tmp = expectation_value(psi,model.sigz) 
            mag_tmp_sum = np.sum(mag_tmp)
            Mag_tot += mag_tmp_sum
    Mag_tot = Mag_tot/N_steps/2
    return Mag_tot # return magnetization 


"""Final results"""

def contract_partition_function(MPS_initial,MPS_final):
    """not working yet"""
    Nx = MPS_initial.Nx
    d = 2
    Z = np.array([[1.0]])

    for s in range(Nx):
        tmp = np.tensordot(Z, np.conj(MPS_final.Bs[s]), axes=([0],[0]))
        C = np.tensordot(tmp, MPS_initial.Bs[s], axes=([0,1],[0,1]))

    return C

"""Calculating observables like magnetisation and correlation length"""

def expectation_value(psi,operator):
    res = []
    for s in range(psi.Nx): #sites
        psi_site = np.tensordot(np.diag(psi.Lambdas[s]), psi.Bs[s], axes=([1],[0])) #singe-site wavefunction
        tmp = np.tensordot(operator, psi_site, axes = ([1],[1]))
        res.append(np.tensordot(psi_site.conj(), tmp, axes = ([0,1,2],[1,0,2])))
    result = np.sum(res)/psi.Nx
    return np.real_if_close(result) 

def get_mag_curve(Nx,betas, J, N_steps, chi_m, eps,bc="finite"):
    mags = []
    for beta in betas: 
        initial_state = Initial_state(Nx,bc) 
        model_test = Ising_model(Nx, J, N_steps, beta, chi_m, eps, bc)  #self, Nx, J, N_steps, beta, chi_m, eps, bc= "finite"
        mag = run_TEBD(initial_state, model_test)
        mags.append(mag)
    return mags 

def get_correlation_curve(Nx,betas, J, N_steps, chi_m, eps):
    bc = "infinite"
    corrs = []
    for beta in betas: 
        initial