### performing the contraction of the tensor network
import numpy as np
from .TensorNetwork import * 

def Theta(psi,i,j): 
    theta1 = np.tensordot(np.diag(psi.Lambdas[i]), psi.Bs[i], axes=([1],[0])) # contract leg 1 of Lambda(i) with leg 0 of B(i)
    theta2 = np.tensordot(theta1, psi.Bs[j], axes=([2],[0])) # contract leg 1 of theta1 with leg 0 of B(i+1)
    print("function Theta, Lambda[i]=", psi.Lambdas[i], "B[i]= ", psi.Bs[i], "B[i+1]= ", psi.Bs[j])
    return theta2 

def update_bond(psi,model,chi_m,eps,bond):
    d = psi.d # physical dymension = 2 
    i,j = sites_for_bond(psi,bond) 
    psi.eps = eps
    psi.chi_m = chi_m
    theta = Theta(psi,i,j) # B(n)*B(n*1) -> vL i* j* vR
    model.site_tensor()
    T = model.tensor 
    #T = np.reshape(T, [d,d,d,d])
    #T= T/np.linalg.norm(T)
    theta = np.tensordot(T, theta, axes=([2,3], [1,2])) # U's axes [2,3] with theta's axes [1,2]
    theta = theta/np.linalg.norm(theta)
    """
    contraction indices: ????????????

    -0---     Theta      ---3-
            |       |
            1       2
            |       |
            2       3
            |       |
             Unitary
            |       |
            0       1

    """ 
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

def run_TEBD(psi, model, N_steps, chi_m, eps):
    nbonds = psi.nbonds 
    model.Unitary() 
    Uni_list = model.Uni_list 
    assert len(Uni_list) == nbonds
    for step in range(N_steps):
        for k in [0,1]:
            for bond in range(k, nbonds, 2): # range(start, stop, steps): even and then odd unitaries
                print("step = ", step,"bond= ", bond)
                update_bond(psi, model, chi_m, eps, bond)


"""Final results"""

def contract_partition_function(MPS_initial,MPS_final):
    Nx = MPS_initial.Nx
    d = MPS_initial.d
    Z = np.array([[1.0]])

    for s in range(Nx):
        tmp = np.tensordot(Z, np.conj(MPS_final.Bs[s]), axes=([0],[0]))
        C = np.tensordot(tmp, MPS_initial.Bs[s], axes=([0,1],[0,1]))

    return C