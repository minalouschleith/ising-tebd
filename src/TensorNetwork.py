### main functions used to contract the tensor network of the 2D classical Ising model
import numpy as np
from scipy.linalg import svd 
from scipy.linalg import expm

####################################################################################

def spin_vals(i):
    if i == 0:
        return -1
    else:
        return 1
    
"""Hamiltonian and unitary evolution of the 1D Ising chain"""

class Ising_model:

    def __init__(self, Nx, J, N_steps, dt, beta, chi_m, eps, bc= "finite", d=2):
        """
        self = MPS
        Nx = number of spins in x
        J = coupling constant between spins
        bc = boundary conditions (finite or infinite)
        T = number of time steps 
        Psi_0 = initial state of spin chain
        """

        assert bc in ["finite", "infinite"]
        self.Nx, self.J, self.N_steps, self.dt, self.beta, self.chi_m, self.eps, self.bc, self.d = Nx, J, N_steps, dt, beta, chi_m, eps, bc, d
        self.nbonds = self.Nx-1 if self.bc=="finite" else self.Nx
        ### Pauli matrices 
        self.sigx = np.array([[0.0,1.0],[1.0,0.0]])
        self.sigy = np.array([[0.0,-1.0j],[1.0j,0.0]])
        self.sigz = np.array([[1.0,0.0],[0.0,-1.0]])
        self.id = np.array([[1.0,0.0],[0.0,1.0]])

    def Hamiltonian(self):
        Hami_list = []
        for n in range(self.nbonds):
            H = -self.J * np.kron(self.sigz,self.sigz) #2 site Hamiltonian (-J * sigma_{z,i} x sigma_{z,i+1})
            Hami_list.append(H)
        self.Hami_list = Hami_list
    
    def Unitary(self):
        self.Hamiltonian()
        Hami_list = self.Hami_list 
        #print("Hami =   ",Hami_list, "nbonds =   ", self.nbonds)
        Uni_list = []
        for n in range(self.nbonds):
            U = expm(-self.dt*1j*Hami_list[n]) # Unitary (imaginary) time evolution operator exp(-dt*H)
            Uni_list.append(U)
        self.Uni_list = Uni_list 
        

    def Boltzman_weight(self):
        d=self.d
        beta = self.beta 
        J = self.J
        W = np.zeros((d,d))
        for s1 in range(d):
            for s2 in range(d):
                W[s1,s2] = np.sqrt(np.exp(beta*J*spin_vals(s1)*spin_vals(s2)))
        

    def site_tensor(self):
        d=self.d
        beta = self.beta 
        J = self.J
        W = np.zeros((d,d))
        for i in range(d):
            for j in range(d):
                W[i,j] = np.sqrt(np.exp(-beta*J*spin_vals(i)*spin_vals(j)))


        T=np.zeros((d,d,d,d))
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    for l in range(d):
                        for s in range(d):
                            T[i,j,k,l] += W[i,s]*W[j,s]*W[k,s]*W[l,s]
        self.tensor = T 


###################################################################################


"""Matrix product states that are the building blocks of the tensor network"""

class MPS:

    #initializer 
    def __init__(self, Bs, d, Lambdas, Nx, chi_m, eps, bc="finite"): 
        assert bc in ["finite","infinite"]
        self.Bs, self.d, self.Lambdas, self.Nx, self.chi_m, self.eps, self.bc = Bs, d, Lambdas, Nx, chi_m, eps, bc
        self.nbonds= Nx-1 if bc=="finite" else self.Nx 
    


###################################################################################

def SVD(theta, chi_m, eps):
        chi_vL, dL, dR, chi_vR = theta.shape # bond dimensions (chi_vL,chi_vR=virtual; dL,dR=physical=2)
        theta = np.reshape(theta, [chi_vL*dL, dR*chi_vR]) #merge legs
        A,Lambda,B = svd(theta, full_matrices=False) #singular value decomposition 
        chiv_crit = min(chi_m, np.sum(Lambda>eps)) #critical bond dimension
        print("chiv crit= ", chiv_crit)
        assert chiv_crit>=1, "SVD truncation resulted in chiv_crit<1"
        ids = np.argsort(Lambda)[::-1][:chiv_crit] # indices of the highest chiv_crit singular values
        A,Lambda,B = A[:, ids],Lambda[ids],B[ids,:]
        print("lambda norm", np.linalg.norm(Lambda))
        Lambda = Lambda/np.linalg.norm(Lambda) #renormalize tensor 
        A = np.reshape(A, [chi_vL,dL,chiv_crit]) 
        B = np.reshape(B, [chiv_crit,dR,chi_vR])
        """
        chi_vL - A - chiv_crit - Lambda - chiv_crit - B - chi_vR
                 |                                    |
                 dL                                   dR 
        """                                 
        return A,Lambda,B 

def Initial_state(Nx ,d=2, bc="finite"):
    
    """Ferromagnetic MPS with all spins up (1), virtual bond dimension trivially =1 """

    B = np.zeros([1,d,1], dtype=float)
    B[0,0,0]=1.0 # spin up, B[0,1,0] is spin down
    Lambda = np.ones([1], dtype= float)
    Bs = [B.copy() for i in range(Nx)] 
    Lambdas = [Lambda.copy() for i in range(Nx)]
    return MPS(Bs,d,Lambdas,Nx,1,1,bc) 
    
def Initial_Neel(Nx,d=2,bc="finite"):
    Bs=[]
    Lambda = np.ones([1], dtype= float)
    Lambdas = [Lambda.copy() for i in range(Nx)]
    for i in range(Nx):
        B = np.zeros([1,d,1], dtype=float)
        if i%2==0: #even site
            B[0,0,0]=1.0 # spin up
        else:
            B[0,1,0]=1.0 
        Bs.append(B) 
    
    return MPS(Bs,d,Lambdas,Nx,1,1,bc) 