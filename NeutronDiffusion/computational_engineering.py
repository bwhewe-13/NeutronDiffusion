"""
Code Taken directly from Computational Nuclear Engineering and Radiological Science
Using Python by Ryan G. McClarren. Focus on Chapters 19 and 20.
"""

import numpy as np


def swap_rows(A, a, b):
    """ Rows two rows in a matrix, switch row a with row b
    args:
        A: matrix to perform row swaps on
        a: row index of matrix
        b: row index of matrix
    Returns: 
        Nothing
    Side Effects: 
        Changes A to have rows a and b swapped
    """
    assert(a>=0) and (b>=0)
    N = A.shape[0] #number of rows
    assert (a<N) and (b<N) #less than because 0-based indexing
    temp = A[a,:].copy()
    A[a,:] = A[b,:].copy()
    A[b,:] = temp.copy()


def LU_factor(A,LOUD=False):
    """ face in place A in L * U = A. The lower triangular parts of A
    are the L matrix. The L has implied ones on the diagonal.
    Args:
        A: N by N array
    Returns: 
        a vector holding the order of the rows
        relative to the original order
    Side Effects: 
        A is factored in place.
    """

    [Nrow,Ncol]= A.shape
    assert Nrow==Ncol
    N = Nrow

    #create scale factors
    s = np.zeros(N)
    count = 0
    row_order = np.arange(N)
    for row in A:
        s[count] = np.max(np.fabs(row))
        count += 1
    if LOUD:
        print('s = {}'.format(s))
        print('Original Matrix is\n{}'.format(A))

    for column in range(0,N):
        #swap rows if needed
        largest_pos = np.argmax(np.fabs(A[column:N,column]/s[column]))+column
        if (largest_pos != column):

            swap_rows(A,column,largest_pos)
            #keep track of changes to RHS
            tmp = row_order[column]
            row_order[column] = row_order[largest_pos]
            row_order[largest_pos] = tmp
            #re-order s
            tmp = s[column]
            s[column] = s[largest_pos]
            s[largest_pos] = tmp
            if LOUD:
                print('A =\n',A)
        for row in range(column+1,N):
            mod_row = A[row]
            factor = mod_row[column]/A[column,column]
            mod_row = mod_row - factor*A[column,:]
            # put the factor in the correct place in the modified row
            mod_row[column] = factor
            # only take the part of the modified row we need
            mod_row = mod_row[column:N]
            A[row,column:N] = mod_row
    return row_order


def LU_solve(A,b,row_order):
    """ Take a LU factorized matrix and solve it for RHS b

    Args:
        A: N by N array that has been LU factored with 
        assumed 1's on the diagonal of the L matrix
        b: N by 1 array of righthand side
        row_order: list giving the re-ordered equations
        from the LU factorization with pivoting
    Returns:
        x: N by 1 array of solutions
    """
    [Nrow,Ncol] = A.shape
    assert Nrow == Ncol
    assert b.size == Ncol
    assert row_order.max() == Ncol-1
    N = Nrow
    # Reorder the equations
    tmp = b.copy()
    for row in range(N):
        b[row_order[row]] = tmp[row]
        
    x = np.zeros(N)
    # temporary vector for L^-1 b
    y = np.zeros(N)
    # forward solve
    for row in range(N):
        RHS = b[row]
        for column in range(0,row):
            RHS -= y[column]*A[row,column]
        y[row] = RHS
    # back solve
    for row in range(N-1,-1,-1):
        RHS = y[row]
        for column in range(row+1,N):
            RHS -= x[column]*A[row,column]
        x[row] = RHS/A[row,row]
    return x


def create_grid(R,I):
    """ Create the cell edges and centers for a 
    domain of size R and I cells
    Args:
        R: size of domain
        I: number of cells

    Returns:
        Delta_r: the width of each cell
        centers: the cell centers of the grid
        edges: the cell edges of the grid
    """
    Delta_r = float(R)/I 
    centers = np.arange(I)*Delta_r + 0.5*Delta_r 
    edges = np.arange(I+1)*Delta_r 
    return Delta_r, centers, edges


def inversePower(A,B,epsilon=1.0e-6,LOUD=False):
    """ Solve the generalized eigenvalue problem 
    Ax = l B x using inverse power iteration
    Inputs
    A: the LHS matrix (must be invertible)
    B: the RHS matrix
    epsilon: tolerance on eigenvalue
    Outputs:
    l: the smallest eigenvalue of the problem
    x: the associated eigenvector
    """

    N,M = A.shape
    assert(N == M)
    # Generate initial guess
    x = np.random.rand((N))
    x /= np.linalg.norm(x)

    l_old = 0
    converged = 0

    row_order = LU_factor(A,LOUD=False)

    iteration = 1
    while not(converged):
        b = LU_solve(A, B @ x, row_order)
        l = np.linalg.norm(b)
        sign = b[0] / x[0] / l
        x = b / l

        converged = (np.fabs(l - l_old) < epsilon)
        l_old = l
        if LOUD:
            print("Iteration: {}\tMagnitude of l {}".format(iteration,1.0/l))

        iteration += 1

    return sign/l, x


def DiffusionEigenvalue(R,I,D,Sig_a,nuSig_f,BC,geometry,epsilon=1.0e-8):
    """ Solves a neutron diffusion eigenvalue problem in 1-D geometry 
    using cell-averaged unknowns
    Args:
        R: size of domain
        I: number of cells 
        D: name of function that returns diffusion coefficients for a given r
        Sig_a: name of function that returns Sigma_a for a given r
        nuSig_f: name of function that returns nu Sigma_f for a given r
        BC: Boundary Condition at r = R in form [A, B]
        geometry: shape of problem
            0 for slab
            1 for cylindrical
            2 for spherical
    
    Returns:
        k: the multiplication factor of the system
        phi: the fundamental mode with norm 1
        centers: position at cell centers

    """
    
    # Set up the Grid
    Delta_r, centers, edges = create_grid(R,I) 
    A = np.zeros((I+1,I+1)) 
    B = np.zeros((I+1,I+1)) 

    #define surface areas and volumes
    assert((geometry == 0) or (geometry == 1) or (geometry == 2)) 
    if (geometry == 0): 
        # in slab it's 1 everywhere except at the left edge
        S = 0.0*edges+1 # surface area
        S[0] = 0.0 #to enforce Refl BC
        # in slab it's dr
        V = 0.0*centers + Delta_r #volume 
    elif (geometry == 1): #cylinder
        #i in cylinder it's 2pi r
        S = 2.0*np.pi*edges #surface area
        # in cylinder it's pi(r^2-r^2)
        V = np.pi*(edges[1:(I+1)]**2 - edges[0:I]**2) #volume
    elif (geometry == 2): #sphere
        # in sphere it's 4 pi^2
        S = 4.0*np.pi*edges**2 #surface area
        # in sphere its 4/3 pi(r^3-r^3)
        V = 4.0/3.0*np.pi*(edges[1:(I+1)]**3 - edges[0:I]**3) #volume
    
    # Set up BC at R
    A[I,I] = (BC[0]*0.5 + BC[1]/Delta_r)
    A[I,I-1] = (BC[0]*0.5 - BC[1]/Delta_r)

    for i in range(I):
        r = centers[i]
        A[i,i] = (0.5/(Delta_r * V[i]) * ((D(r) + D(r + Delta_r)) * S[i+1]) + Sig_a(r))
        B[i,i] = nuSig_f(r)
        if (i > 0):
            A[i, i - 1] = -0.5 * (D(r) + D(r - Delta_r))/(Delta_r * V[i]) * S[i]
            A[i,i] += 0.5/(Delta_r * V[i]) * ((D(r) + D(r - Delta_r)) * S[i])
        A[i,i+1] = -0.5 * (D(r) + D(r + Delta_r))/(Delta_r * V[i]) * S[i+1]


    # Find Eigenvalue
    l,phi = inversePower(A,B,epsilon)
    k = 1.0 / l
    
    # remove last row of phi
    phi = phi[:I]
    
    return k,phi,centers


def inversePowerBlock(M11,M21,M22,P11,P12,epsilon=1.0e-6,LOUD=False):
    """ Solve the generalize eigenvalue problem 
    (M11  0) (phi_1) = l (P11  P12) using the inverse power iteration
    (M21 M22)(phi_2) =   ( 0    0 )
    Inputs
        Mij: An LHS matrix (must be invertible)
        P1j: A fission matrix 
        epsilon: tolerance on eigenvalue
    Outputs:
        l: the smallest eigenvalue of the problem
        x1: the associated eigenvector for the first block
        x2: the associated eigenvector for the second block
    """
    N,M = M11.shape
    assert (N == M)
    # generate initial guess
    x1 = np.random.random((N))
    x2 = np.random.random((N))
    l_old = np.linalg.norm(np.concatenate((x1,x2)))
    x1 /= l_old
    x2 /= l_old
    converged = 0
    # compute LU factorization of M11
    row_order11 = LU_factor(M11,LOUD=False)
    # Compute LU factorization of M22
    row_order22 = LU_factor(M22,LOUD=False)
    iteration = 1
    while not (converged):
        # solve for b1
        b1 = LU_solve(M11,P11 @ x1 + P12 @ x2,row_order11)
        # solve for b2
        b2 = LU_solve(M22,-M21 @ b1,row_order22)
        # Eigenvalue estimate is norm of combined vectors
        l = np.linalg.norm(np.concatenate((b1,b2)))
        x1 = b1 / l
        x2 = b2 / l
        converged = ( np.fabs(l - l_old) < epsilon )
        l_old = l
        if LOUD:
            print('Iteration {}\tMagnitude of l = {}'.format(iteration,1.0/l))
        iteration += 1
    return 1.0 / l, x1, x2


def TwoGroupEigenvalue(R,I,D1,D2,Sig_r1,Sig_r2,nu_Sigf1,nu_Sigf2,Sig_s12,
                            BC1,BC2,geometry,epsilon=1.0e-8):
    """ Solve a neutron diffusion eigenvalue problem in a 1-D geometry
    using cell-averaged unknowns
    Args:
        R: size of domain
        I: number of cells
        Dg: name of function that returns diffusion coefficient for
            a given r
        Sig_rg: name of function that returns Sigma_rg for a given r
        nuSig_fg : name of function that returns nu Sigma_fg for a given r
        Sig_s12: name of function that returns Sigma_s12 for a given r
        BC1: Boundary Value of fast phi at r = R in form [A,B]
        BC2: Boundary Value of thermal phi at r = R in form [A,B]
        geometry: shape of problem
            0 for slab
            1 for cylindrical
            2 for spherical

    Returns:
        k: the multiplication factor of the system 
        phi_fast: the fast flux fundamental mode with norm 1
        phi_thermal: the thermal flux fundamental mode with norm 1
        centers: positions at cell centers

    """
    # create the grid
    Delta_r,centers,edges = create_grid(R,I)
    M11 = np.zeros((I+1,I+1))
    M21 = np.zeros((I+1,I+1))
    M22 = np.zeros((I+1,I+1))
    P11 = np.zeros((I+1,I+1))
    P12 = np.zeros((I+1,I+1))
    # define the surface areas and volumes
    assert ( (geometry == 0) or (geometry == 1) or (geometry == 2) )
    if (geometry == 0):
        # in slab it's 1 everywhere except at the left edge
        S = 0.0*edges+1
        S[0] = 0.0
        # in slab its dr
        V = 0.0*centers + Delta_r
    elif (geometry == 1):
        # in cylinder it is 2 pi r
        S = 2.0 * np.pi * edges
        # in cylinder its pi (r^2 - r^2)
        V = np.pi * (edges[1:(I+1)]**2 - edges[0:I]**2)
    elif (geometry == 2):
        # in sphere it is 4 pi r^2
        S = 4.0 * np.pi* edges**2 
        # in sphere its 4/3 pi (r^3 - r^3)
        V = 4.0/3.0 * np.pi * (edges[1:(I+1)]**3 - edges[0:I]**3 )

    # Set up BC at R
    M11[I,I] = (BC1[0]*0.5 + BC1[1]/Delta_r)
    M11[I,I-1] = (BC1[0]*0.5 - BC1[1]/Delta_r)
    M22[I,I] = (BC1[0]*0.5 + BC2[1]/Delta_r)
    M22[I,I-1] = (BC1[0]*0.5 - BC2[1]/Delta_r)

    # fill in rest of matrix
    for i in range(I):
        r = centers[i]
        M11[i,i] = (0.5/(Delta_r * V[i]) * ((D1(r)+D1(r+Delta_r))*S[i+1]) + 
                    Sig_r1(r))
        M22[i,i] = (0.5/(Delta_r * V[i]) * ((D2(r)+D2(r+Delta_r))*S[i+1]) + 
                    Sig_r2(r))
        M21[i,i] = -Sig_s12(r)
        P11[i,i] = nu_Sigf1(r)
        P12[i,i] = nu_Sigf2(r)
        if (i > 0):
            M11[i,i-1] = -0.5*(D1(r)+D1(r-Delta_r))/(Delta_r * V[i])*S[i]
            M11[i,i] += 0.5/(Delta_r * V[i])*((D1(r)+D1(r-Delta_r))*S[i])
            M22[i,i-1] = -0.5*(D2(r)+D2(r-Delta_r))/(Delta_r * V[i])*S[i]
            M22[i,i] += 0.5/(Delta_r * V[i])*((D2(r)+D2(r-Delta_r))*S[i])
        M11[i,i+1] = -0.5*(D1(r)+D1(r+Delta_r))/(Delta_r * V[i])*S[i+1]
        M22[i,i+1] = -0.5*(D2(r)+D2(r+Delta_r))/(Delta_r * V[i])*S[i+1]

    # find eigenvalue
    l,phi1,phi2 = inversePowerBlock(M11,M21,M22,P11,P12,epsilon)
    k = 1.0/l
    # remove last element of phi because it is outside the domain
    phi1 = phi1[:I]
    phi2 = phi2[:I]
    return k,phi1,phi2,centers

