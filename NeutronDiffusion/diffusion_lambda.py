""" Multigroup Diffusion Code """

from NeutronDiffusion import create

import numpy as np
import numpy.ctypeslib as npct
# from scipy import sparse
# from scipy.sparse.linalg import spsolve
import ctypes
import os


class Diffusion:

    def __init__(self,G,R,I,D,scatter,chi,fission,removal,BC,geo): 
        self.G = G # number of energy groups
        self.R = R # length of problem (cm)
        self.I = I # number of spatial cells
        self.D = D # Diffusion Coefficient
        self.scatter = scatter # scatter XS
        self.chi = chi # birth rate
        self.fission = fission # nu * fission
        self.removal = removal # removal XS
        self.BC = BC # boundary conditions
        self.geo = geo # geometry
        command = "gcc -fPIC -shared -o ../src/matrix_build.dll ../src/matrix_build.c \
            -DG={} -DI={} -Dmaterials=1".format(self.G,self.I)
        os.system(command)
        # There are problems with full fission matrix and others 
        # with the two vectors
        self.full_fission = True if chi is None else False

    @classmethod
    def run(cls,problem,G,I,geo):
        # This is for running the diffusion problems
        # Returns phi and keff
        attributes = create.selection(problem,G,I)
        initialize = cls(*attributes,geo=geo)
        # Create Geometry
        initialize.geometry()
        # Create LHS Matrix
        A = initialize.construct_A_B(RHS=False)
        # Solve for keff
        return initialize.solver(A)

    def geometry(self):
        """ Creates the grid and assigns the surface area and volume """
        # For the grid
        self.delta = float(self.R)/ self.I
        self.centers = np.arange(self.I) * self.delta + 0.5 * self.delta
        self.edges = np.arange(self.I + 1) * self.delta
        # for surface area and volume
        if (self.geo == 'slab'): 
            self.SA = 0.0*self.edges + 1 # 1 everywhere except at the left edge
            self.SA[0] = 0.0 #to enforce Refl BC
            self.V = 0.0*self.centers + self.delta # dr
        elif (self.geo == 'cylinder'): 
            self.SA = 2.0*np.pi*self.edges # 2pi r
            self.V = np.pi*(self.edges[1:(self.I+1)]**2 - self.edges[0:self.I]**2) # pi(r^2-r^2)
        elif (self.geo == 'sphere'): 
            self.SA = 4.0*np.pi*self.edges**2 # 4 pi^2        
            self.V = 4.0/3.0*np.pi*(self.edges[1:(self.I+1)]**3 - self.edges[0:self.I]**3) # 4/3 pi(r^3-r^3)


    def change_space(self,ii,gg): 
        """ Change the cell spatial position 
        include left and right of each
        """
        left = gg * (self.I+1) + ii - 1
        middle = gg * (self.I+1) + ii
        right = gg * (self.I+1) + ii + 1
        return left,middle,right


    def construct_b(self,phi):
        b = np.zeros((self.G*(self.I+1)))
        for global_x in range(self.G*(self.I+1)):
            local_cell = global_x % (self.I+1)
            if local_cell == (self.I):  # for boundary
                continue
            r = self.centers[local_cell] #determine the physical distance
            group_in = int(global_x / (self.I+1))
            for group_out in range(self.G):
                global_y = group_out * (self.I+1) + local_cell
                b[global_x] += self.chi(r)[group_in] * self.fission(r)[group_out] * phi[global_y]
        return b


    def construct_b_fast(self,phi):
        # Load library
        library = ctypes.CDLL('../src/matrix_build.dll')
        # Set up phi
        phi = phi.astype('float64')
        phi_ptr = ctypes.c_void_p(phi.ctypes.data)
        # Set up b
        b = np.zeros((self.G*(self.I+1)),dtype='float64')
        b_ptr = ctypes.c_void_p(b.ctypes.data)
        # Set up chi
        chi = self.chi(0).astype('float64')
        chi_ptr = ctypes.c_void_p(chi.ctypes.data)
        # Set up nufission
        fission = self.fission(0).astype('float64')
        fission_ptr = ctypes.c_void_p(fission.ctypes.data)
        library.construct_b_lambda(phi_ptr,b_ptr,chi_ptr,fission_ptr)
        return b


    def construct_A_fast(self):
        # Setting 2D Array sizes
        class full_matrix(ctypes.Structure): 
            _fields_ = [("array", (ctypes.c_double * (self.G*(self.I+1))) * (self.G*(self.I+1)))]

        class cross_section(ctypes.Structure):
            _fields_ = [("array", (ctypes.c_double * self.G) * self.G)]

        class boundary(ctypes.Structure):
            _fields_ = [("array", (ctypes.c_double * 2) * self.G)]

        slib = ctypes.CDLL("../src/matrix_build.dll")
        slib.construct_A_lambda.argtypes = [ctypes.POINTER(full_matrix),ctypes.POINTER(cross_section),
                             ctypes.POINTER(boundary),ctypes.c_void_p,ctypes.c_void_p,
                             ctypes.c_void_p,ctypes.c_void_p,ctypes.c_double]
        slib.construct_A_lambda.restype = None
        A = np.zeros((self.G*(self.I+1),self.G*(self.I+1))).astype('float64')

        A_ptr = full_matrix()
        A_ptr.array = npct.as_ctypes(A)

        scat_ptr = cross_section()
        scat_ptr.array = npct.as_ctypes(self.scatter(0))

        bc_ptr = boundary()
        bc_ptr.array = npct.as_ctypes(self.BC)

        diffusion = self.D(0).astype('float64')
        D_ptr = ctypes.c_void_p(diffusion.ctypes.data)

        SA = self.SA.astype('float64')
        SA_ptr = ctypes.c_void_p(SA.ctypes.data)

        V = self.V.astype('float64')
        V_ptr = ctypes.c_void_p(V.ctypes.data)

        removal = np.array(self.removal(0)).astype('float64')
        rem_ptr = ctypes.c_void_p(removal.ctypes.data)

        slib.construct_A_lambda(ctypes.byref(A_ptr),ctypes.byref(scat_ptr),ctypes.byref(bc_ptr),
                        D_ptr,SA_ptr,V_ptr,rem_ptr,ctypes.c_double(self.delta))
        return np.array(A_ptr.array)


    def construct_A_B(self,RHS=False):
        """ Creates the left and right matrices for 1-D neutron diffusion eigenvalue problem
        of form Ax = (1/k)Bx 
        Returns:
            A: left hand side matrix - removal cross-section
            B: right hand side matrix - fission cross-section
        """
        A = np.zeros((self.G*(self.I+1),self.G*(self.I+1))) 
        B = np.zeros((self.G*(self.I+1),self.G*(self.I+1))) if RHS else None
        # Iterate over energy groups
        for gg in range(self.G):
            # Iterate over spatial cells
            for ii in range(self.I):
                r = self.centers[ii] #determine the physical distance
                minus,cell,plus = Diffusion.change_space(self,ii,gg) #move to a given group submatrix

                A[cell,cell] = (2.0/(self.delta * self.V[ii])*
                                           ((self.D(r)[gg]*self.D(r+self.delta)[gg])/(self.D(r)[gg]+self.D(r+self.delta)[gg])*self.SA[ii+1]) + self.removal(r)[gg])
                A[cell,plus] = -2.0*(self.D(r)[gg]*self.D(r+self.delta)[gg])/(self.D(r)[gg]+self.D(r+self.delta)[gg])/(self.delta*self.V[ii])*self.SA[ii+1] 
                if ii > 0:
                    A[cell,minus] = -2.0*(self.D(r)[gg]*self.D(r-self.delta)[gg])/(self.D(r)[gg]+self.D(r-self.delta)[gg])/(self.delta*self.V[ii]) * self.SA[ii] 
                    A[cell,cell] += 2.0/(self.delta * self.V[ii])*((self.D(r)[gg]*self.D(r-self.delta)[gg])/(self.D(r)[gg]+self.D(r-self.delta)[gg]) * self.SA[ii])
                #in scattering
                for gpr in range(self.G):
                    _,prime,_ = Diffusion.change_space(self,ii,gpr)
                    if (gpr != gg): #skip the same group scattering
                        A[cell,prime] = -self.scatter(r)[gg,gpr] #scattering diagonal
                    if RHS:
                        B[cell,prime] = self.chi(r)[gg]*self.fission(r)[gpr] #set up the fission diagonal 
            # sets the boundary conditions at the edge of each submatrix, i = I
            minus,cell,plus = Diffusion.change_space(self,self.I,gg) 
            A[cell,cell] = self.BC[gg,0]*0.5 + self.BC[gg,1]/self.delta 
            A[cell,minus] = self.BC[gg,0]*0.5 - self.BC[gg,1]/self.delta 
        if RHS:
            return A,B
        return A

    def solver(self,A,B=None,fast=False,tol=1E-10,MAX_ITS=100):
        """ Solve the generalized eigenvalue problem Ax = (1/k)Bx
        Inputs:
            A: left-side (groups*N)x(groups*N) matrix
            B: right-side (groups*N)x(groups*N) matrix
        Outputs:
            keff: 1 / the smallest eigenvalue 
            phi: the associated eigenvector, broken up into Nxgroups matrix
        """
        # Initialize Phi
        phi_old = np.random.random(self.G*(self.I+1))
        phi_old /= np.linalg.norm(phi_old)

        converged = 0; count = 1
        while not(converged):
            if B is not None:
                phi = np.linalg.solve(A, B @ phi_old)
                # phi = spsolve(sparse.csr_matrix(A), B @ phi_old)
            elif fast:
                phi = np.linalg.solve(A, Diffusion.construct_b_fast(self,phi_old))

            else:
                phi = np.linalg.solve(A, Diffusion.construct_b(self,phi_old))
                            
            keff = np.linalg.norm(phi)
            phi /= keff

            change = np.linalg.norm(phi - phi_old)
            converged = (change < tol) or (count >= MAX_ITS)
            # print('Iteration: {} Change {}\tKeff {}'.format(count,change,keff))

            count += 1
            phi_old = phi.copy()

        phi = np.reshape(phi,(self.I+1,self.G),order='F')
        return phi[:self.I],keff
        