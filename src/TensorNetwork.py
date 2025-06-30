### main functions used to contract the tensor network of the 2D classical Ising model
import numpy as np
from scipy.linalg import svd 

class Ising:

    def Ising_model(self, Nx, J, T, dt, chi_m, eps, bc= "finite"):
        """
        self = MPS
        Nx = number of spins in x
        J = coupling constant between spins
        bc = boundary conditions (finite or infinite)
        T = number of time steps 
        Psi_0 = initial state of spin chain
        """

        assert bc in ["finite", "infinite"]
        self.Nx, self.J, self.T, self.dt, self.chi_m, self.eps, self.bc, = Nx, J, T, dt, chi_m, eps, bc
        self.d = 2 #Hilbert space dimension per site (spin up & down)
        ### Pauli matrices 
        self.sigx = np.array([[0.0,1.0],[1.0,0.0]])
        self.sigy = np.array([[0.0,-1.0j],[1.0j,0.0]])
        self.sigz = np.array([[1.0,0.0],[0.0,-1.0]])
        self.id = np.array([[1.0,0.0],[0.0,1.0]])

    def Hamiltonian(self):
        nbonds = self.Nx-1 if self.bc=="finite" else self.Nx
        Hami_list = []
        for _ in range(nbonds):
            H = -self.J * np.kron(self.sigz,self.sigz) #2 site Hamiltonian (-J * sigma_{z,i} x sigma_{z,i+1})
        Hami_list.append(H)
        self.Hami_list = Hami_list


class MPS:

    #initializer 
    def __init__(self,Bs,Lambdas,Nx,bc="finite"): 
        assert bc in ["finite","infinite"]
        self.Bs, self.Lambdas, self.Nx, self.bc = Bs,Lambdas,Nx,bc
        self.nbonds= Nx-1 if bc=="finite" else self.Nx

    def SVD(theta, chi_m, eps):
        chi_vL, dL, dR, chi_vR = theta.shape # bond dimensions (chi_vL,chi_vR=virtual; dL,dR=physical=2)
        theta = np.reshape(theta, [chi_vL*dL, dR*chi_vR]) #merge legs
        A,Lambda,B = svd(theta, full_matrices=False) #singular value decomposition 
        chiv_crit = min(chi_m, np.sum(Lambda>eps)) #critical bond dimension
        assert chiv_crit>=1 
        ids = np.argsort(Lambda)[::-1][:chiv_crit] # indices of the highest chiv_crit singular values
        A,Lambda,B = A[:,ids],Lambda[ids],B[ids,:]
        Lambda = Lambda/np.linalg.norm(Lambda) 
        A = np.reshape(A, [chi_vL,dL,chiv_crit]) 
        B = np.reshape(B, [chiv_crit,dR,chi_vR])
        """
        chi_vL - A - chiv_crit - Lambda - chiv_crit - B - chi_vR
                 |                                    |
                 dL                                   dR 
        """                                 
        return A,Lambda,B 
    
    


###################################################################################

def Initial_state(Nx ,d=2 ,bc="finite"):
    """
    Ferromagnetic MPS with all spins up (1), virtual bond dimension trivially =1
     """
    B = np.zeros([1,d,1], dtype=float)
    B[0,0,0]=1.0 # spin up, B[0,1,0] is spin down
    Lambda = np.ones([1], dtype= float)
    Bs = [B.copy() for i in range(Nx)] 
    Lambdas = [Lambda.copy() for i in range(Nx)]
    return MPS(Bs,Lambdas,Nx,bc)
    
