### performing the contraction of the tensor network
import numpy as np
from scipy.linalg import expm
from .TensorNetwork import SVD

#why need a list of unitaries if the Hamiltonian is the same on each pair of sites anyway? 
def Unitary(self):
    Hami_list = self.Hami_list
    Uni_list = []
    for H in Hami_list:
        U = expm(-self.dt*H) # Unitary (imaginary) time evolution operator exp(-dt*H)
        Uni_list.append(U)
    return Uni_list

def Theta(self,i,j): 
    theta1 = np.tensordot(np.diag(self.Lambdas[i]), self.Bs[i], [1,0]) # contract leg 1 of Lambda(i) with leg 0 of B(i)
    theta2 = np.tensordot(theta1, self.Bs[j], [1,0]) # contract leg 1 of theta1 with leg 0 of B(i+1)
    return theta2 

def update_bond(psi, i): 
    d = psi.d # physical dymension = 2 
    j = (i+1) % psi.Nx
    theta = psi.Theta(i,j) # B(n)*B(n*1) -> vL i* j* vR
    U = psi.Unitary[i] 
    U = np.reshape(U,[d,d,d,d]) # U has 4 legs: i* j*, i j
    theta = np.tensordot(U, theta, axes=([2,3], [1,2])) # U's axes [2,3] with theta's axes [1,2]

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

