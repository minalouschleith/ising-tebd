### performing the contraction of the tensor network
import numpy as np
from .TensorNetwork import * 
from scipy.sparse.linalg import eigs 

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
            for bond in range(k, nbonds, 2): 
                update_bond(psi, model, chi_m, eps, bond)


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

def magnetization_Z(psi):
    sigz = np.array([[1.0,0.0],[0.0,-1.0]])
    return expectation_value(psi,sigz)

def get_mag_curve(Nx,betas, J, N_steps, chi_m, eps,bc="finite"):

    mags = []
    for beta in betas: 
        psi = Initial_state(Nx,bc) 
        model_test = Ising_model(Nx, J, N_steps, beta, chi_m, eps, bc)  #self, Nx, J, N_steps, beta, chi_m, eps, bc= "finite"
        run_TEBD(psi, model_test)
        mags.append(magnetization_Z(psi)) #magnetization in the final state
    return mags 

def critical_temp_analytic(J):
    return 2*J/(np.log(1+np.sqrt(2)))

def mag_analytic(beta,J): 
    beta_crit = 1/critical_temp_analytic(J)
    if beta>(beta_crit+1e-10):
        return (1-(np.sinh(2*beta))**(-4))**(1/8)
    else:
        return 0
    
def correlation_length(psi):
    assert psi.bc == "infinite"
    B = psi.Bs[0] #first state
    chi = B.shape[0] #virtual bond dimension
    T = np.tensordot(B, np.conj(B), axes = ([1],[1]))
    T = np.transpose(T, [0,2,1,3])
    for s in range(1, psi.Nx): 
        B = psi.Bs[s]
        T = np.tensordot(T,B, axes = ([2], [0]))
        T = np.tensordot(T, np.conj(B), axes = ([2,3], [0,1]))
    T = np.reshape(T, (chi**2, chi**2))
    epsilon = eigs(T, k=2, which= 'LM', return_eigenvectors=False, ncv=20) #two largest eigenvalues
    corr = -psi.Nx / np.log(np.min(np.abs(epsilon))) #second largest eigenvalue 
    return corr 

def get_correlation_curve(Nx,betas,J, N_steps, chi_m, eps):
    bc = "infinite"
    corrs = []
    for beta in betas:
        psi = Initial_state(Nx,bc)
        model = Ising_model(Nx, J, N_steps, beta, chi_m, eps, bc)
        run_TEBD(psi,model)
        corrs.append(correlation_length(psi)) #correlation in the final state
    return corrs 

def Onsager_analytical_correlation_curve(beta,J):
    beta_crit = 1/critical_temp_analytic(J)
    if beta>(beta_crit+1e-10):
        return 1/(1/beta_crit - 1/beta)     # T< Tcrit, xi ~ 1/(Tc-T)
    else: 
        return -1/np.log(np.tanh(beta*J)) #Onsager's result 
