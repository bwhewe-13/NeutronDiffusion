"""Multigroup Diffusion Code"""

from diffusion.create import selection

import numpy as np
import numpy.ctypeslib as npct
from scipy import sparse
from scipy.sparse.linalg import spsolve
import ctypes
import os
import pkg_resources


SRC_PATH = pkg_resources.resource_filename("diffusion", "../src/")
# SRC_PATH = "./diffusion/src/"


class Diffusion:

    def __init__(self, G, R, I, D, scatter, chi, fission, removal, BC, geo):
        self.G = G  # number of energy groups
        self.R = R  # length of problem (cm)
        self.I = I  # number of spatial cells
        self.D = D  # Diffusion Coefficient
        self.scatter = scatter  # scatter XS
        self.chi = chi  # birth rate
        self.fission = fission  # nu * fission
        self.removal = removal  # removal XS
        self.BC = BC  # boundary conditions
        self.geo = geo
        self.materials = len(D)
        # Compile C functions
        command = "gcc -fPIC -shared -o {}matrix_build.dll {}matrix_build.c \
                    -DG={} -DI={} -Dmaterials={}".format(
            SRC_PATH, SRC_PATH, self.G, self.I, self.materials
        )
        os.system(command)
        self.full_fission = True if chi is None else False

    @classmethod
    def run(cls, problem, G, I, geo, BC=0):
        # This is for running the diffusion problems
        attributes = selection(problem, G, I, BC)
        initialize = cls(*attributes, geo=geo)
        # Create Geometry
        initialize.geometry()
        # Create LHS Matrix
        A = initialize.construct_A_fast()
        # Solve for keff
        return initialize.solver(A, fast=True)

    def geometry(self):
        """Creates the grid and assigns the surface area and volume"""
        # For the grid
        self.delta = float(sum(self.R)) / self.I
        self.layers = [int(round(ii / self.delta)) for ii in self.R]

        # Verifying sizes match up
        assert sum(self.layers) == self.I

        centers = np.arange(self.I) * self.delta + 0.5 * self.delta
        edges = np.arange(self.I + 1) * self.delta
        # for surface area and volume
        if self.geo == "slab":
            self.SA = 0.0 * edges + 1  # 1 everywhere except at the left edge
            self.SA[0] = 0.0  # to enforce Refl BC
            self.V = 0.0 * centers + self.delta  # dr
        elif self.geo == "cylinder":
            self.SA = 2.0 * np.pi * edges  # 2pi r
            self.V = np.pi * (
                edges[1 : (self.I + 1)] ** 2 - edges[0 : self.I] ** 2
            )  # pi(r^2-r^2)
        elif self.geo == "sphere":
            self.SA = 4.0 * np.pi * edges**2  # 4 pi^2
            self.V = (
                4.0
                / 3.0
                * np.pi
                * (edges[1 : (self.I + 1)] ** 3 - edges[0 : self.I] ** 3)
            )  # 4/3 pi(r^3-r^3)

    def construct_b(self, phi):
        b = np.zeros((self.G * (self.I + 1)))
        material_index = 0
        material_cell = 0
        for global_x in range(self.G * (self.I + 1)):
            local_cell = global_x % (self.I + 1)
            if local_cell == (self.I):  # for boundary
                continue
            if local_cell == 0:
                material_index = 0
                material_cell = 0
            elif (
                material_cell % self.layers[material_index]
                == self.layers[material_index] - 1
            ):
                material_cell = 0
                material_index += 1
            else:
                material_cell += 1
            group_in = int(global_x / (self.I + 1))
            for group_out in range(self.G):
                global_y = group_out * (self.I + 1) + local_cell
                if self.full_fission:
                    b[global_x] += (
                        self.fission[material_index][group_in, group_out]
                        * phi[global_y]
                    )
                else:
                    b[global_x] += (
                        self.chi[material_index][group_in]
                        * self.fission[material_index][group_out]
                        * phi[global_y]
                    )
        return b

    def construct_b_fast(self, phi):
        # Setting 2D array sizes
        class multi_vec(ctypes.Structure):
            _fields_ = [("array", (ctypes.c_double * self.G) * self.materials)]

        class multi_mat(ctypes.Structure):
            _fields_ = [
                ("array", ((ctypes.c_double * self.G) * self.G) * self.materials)
            ]

        # Load library
        library = ctypes.CDLL(SRC_PATH + "matrix_build.dll")
        # Fission Matrix Function
        library.construct_b_list_fission.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(multi_mat),
            ctypes.c_void_p,
        ]
        library.construct_b_list_fission.restype = None
        # Separate fission and birth rate Function
        library.construct_b_list.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(multi_vec),
            ctypes.POINTER(multi_vec),
            ctypes.c_void_p,
        ]
        library.construct_b_list.restype = None
        # Set up phi
        phi = phi.astype("float64")
        phi_ptr = ctypes.c_void_p(phi.ctypes.data)
        # Set up b
        b = np.zeros((self.G * (self.I + 1)), dtype="float64")
        b_ptr = ctypes.c_void_p(b.ctypes.data)
        # Set up layers
        layers = np.array(self.layers).astype("int32")
        lay_ptr = ctypes.c_void_p(layers.ctypes.data)
        # Set up chi and fission
        if self.full_fission:
            fission_ptr = multi_mat()
            fission_ptr.array = npct.as_ctypes(self.fission)
            # Run the problem
            library.construct_b_list_fission(phi_ptr, b_ptr, fission_ptr, lay_ptr)
        else:
            chi_ptr = multi_vec()
            chi_ptr.array = npct.as_ctypes(self.chi)
            fission_ptr = multi_vec()
            fission_ptr.array = npct.as_ctypes(self.fission)
            # Run the problem
            library.construct_b_list(phi_ptr, b_ptr, chi_ptr, fission_ptr, lay_ptr)
        return b

    def construct_A_fast(self):
        # Setting 2D Array sizes
        class full_matrix(ctypes.Structure):
            _fields_ = [
                (
                    "array",
                    (ctypes.c_double * (self.G * (self.I + 1)))
                    * (self.G * (self.I + 1)),
                )
            ]

        class multi_mat(ctypes.Structure):
            _fields_ = [
                ("array", ((ctypes.c_double * self.G) * self.G) * self.materials)
            ]

        class multi_vec(ctypes.Structure):
            _fields_ = [("array", (ctypes.c_double * self.G) * self.materials)]

        class boundary(ctypes.Structure):
            _fields_ = [("array", (ctypes.c_double * 2) * self.G)]

        print(SRC_PATH + "matrix_build.dll")
        slib = ctypes.CDLL(SRC_PATH + "matrix_build.dll")

        slib.construct_A_list.argtypes = [
            ctypes.POINTER(full_matrix),
            ctypes.POINTER(multi_mat),
            ctypes.POINTER(boundary),
            ctypes.POINTER(multi_vec),
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(multi_vec),
            ctypes.c_void_p,
            ctypes.c_double,
        ]
        slib.construct_A_list.restype = None
        A = np.zeros((self.G * (self.I + 1), self.G * (self.I + 1))).astype("float64")

        A_ptr = full_matrix()
        A_ptr.array = npct.as_ctypes(A)

        scat_ptr = multi_mat()
        scat_ptr.array = npct.as_ctypes(self.scatter)

        bc_ptr = boundary()
        bc_ptr.array = npct.as_ctypes(self.BC)

        D_ptr = multi_vec()
        D_ptr.array = npct.as_ctypes(self.D)

        SA = self.SA.astype("float64")
        SA_ptr = ctypes.c_void_p(SA.ctypes.data)

        V = self.V.astype("float64")
        V_ptr = ctypes.c_void_p(V.ctypes.data)

        rem_ptr = multi_vec()
        rem_ptr.array = npct.as_ctypes(self.removal)

        layers = np.array(self.layers).astype("int32")
        lay_ptr = ctypes.c_void_p(layers.ctypes.data)

        slib.construct_A_list(
            ctypes.byref(A_ptr),
            ctypes.byref(scat_ptr),
            ctypes.byref(bc_ptr),
            ctypes.byref(D_ptr),
            SA_ptr,
            V_ptr,
            ctypes.byref(rem_ptr),
            lay_ptr,
            ctypes.c_double(self.delta),
        )
        return np.array(A_ptr.array)

    def solver(self, A, B=None, fast=False, tol=1e-10, MAX_ITS=100):
        """Solve the generalized eigenvalue problem Ax = (1/k)Bx
        Inputs:
            A: left-side (groups*N)x(groups*N) matrix
            B: right-side (groups*N)x(groups*N) matrix
        Outputs:
            keff: 1 / the smallest eigenvalue
            phi: the associated eigenvector, broken up into Nxgroups matrix
        """
        # Initialize Phi
        phi_old = np.random.random(self.G * (self.I + 1))
        phi_old /= np.linalg.norm(phi_old)

        converged = 0
        count = 1
        while not (converged):
            if B is not None:
                phi = np.linalg.solve(A, B @ phi_old)
                # phi = spsolve(sparse.csr_matrix(A), B @ phi_old)
            elif fast:
                phi = np.linalg.solve(A, Diffusion.construct_b_fast(self, phi_old))
            else:
                phi = np.linalg.solve(A, Diffusion.construct_b(self, phi_old))

            keff = np.linalg.norm(phi)
            phi /= keff

            change = np.linalg.norm(phi - phi_old)
            converged = (change < tol) or (count >= MAX_ITS)
            print("Iteration: {} Change {}\tKeff {}".format(count, change, keff))

            count += 1
            phi_old = phi.copy()

        phi = np.reshape(phi, (self.I + 1, self.G), order="F")
        return phi[: self.I], keff
