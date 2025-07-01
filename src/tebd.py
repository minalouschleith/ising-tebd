### performing the contraction of the tensor network
import numpy as np
from .TensorNetwork import * 

def Theta(psi,i,j): 
    print("Bs length= ", len(psi.Bs), " j= ", j, "B[j]= ", psi.Bs[j])
    theta1 = np.tensordot(np.diag(psi.Lambdas[i]), psi.Bs[i], [1,0]) # contract leg 1 of Lambda(i) with leg 0 of B(i)
    theta2 = np.tensordot(theta1, psi.Bs[j], [2,0]) # contract leg 1 of theta1 with leg 0 of B(i+1)
    return theta2 

def update_bond(psi,model,chi_m,eps,bond):
    d = psi.d # physical dymension = 2 
    i,j = sites_for_bond(psi,bond) 
    print("bond= ", bond, "i= ", i , "j= ", j)
    psi.eps = eps
    psi.chi_m = chi_m
    theta = Theta(psi,i,j) # B(n)*B(n*1) -> vL i* j* vR
    model.Unitary() 
    U = model.Uni_list 
    print("U length =. ", len(U)) 
    print("theta =  ",theta, "thetashape = ", theta.shape, "theta length. ", len(theta ) )
    Ui = np.reshape(U[bond],[d,d,d,d]) # U has 4 legs: i* j*, i j
    print("Unitary = ", Ui, "U shape = ", Ui.shape)
    theta = np.tensordot(Ui, theta, axes=([2,3], [1,2])) # U's axes [2,3] with theta's axes [1,2]

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

    theta = np.transpose(theta, [2,0,1,3]) # vL i j vR
    A,Lambda,B = SVD(theta, psi.chi_m, psi.eps) # U1 * S * U2 (U1, U2 = unitaries, S = diagonal, singular values)
    B_tilde1 = np.tensordot((np.diag(psi.Lambdas[i]))**(-1), A, axes=(1,0))
    psi.Bs[i] = np.tensordot(B_tilde1, Lambda, axes=(2,0))  #B_tilde(i)
    psi.Lambdas[j] = Lambda
    psi.Bs[j] = B

"""Time evolution"""

def sites_for_bond(psi, bond):
    """return site indices for bond index"""
    i = bond
    j = (i + 1) if psi.bc == "finite" else (i + 1) % psi.Nx
    return i, j

def run_TEBD(psi, model, Uni_list, N_steps, chi_m, eps):
    nbonds = psi.nbonds 
    assert len(Uni_list) == nbonds
    for step in range(N_steps):
        for k in [0,1]:
            for bond in range(k, nbonds, 2): # range(start, stop, steps): even and then odd unitaries
                update_bond(psi, model, chi_m, eps, bond)
