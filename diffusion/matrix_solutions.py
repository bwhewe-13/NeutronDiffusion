import numba
import numpy as np

import diffusion.utils as utils


def power_iteration(
    diffusion_coef,
    xs_scatter,
    xs_removal,
    chi,
    nusigf,
    boundary,
    medium_map,
    edges_x,
    geometry,
    EPSILON=1e-8,
    MAX_COUNT=100,
):
    cells_x = medium_map.shape[0]
    groups = xs_scatter.shape[1]

    flux_old = np.random.random(groups * (cells_x + 1))
    flux_old /= np.linalg.norm(flux_old)

    # Construct A matrix
    A = _construct_A(
        diffusion_coef, xs_removal, xs_scatter, boundary, medium_map, edges_x, geometry
    )

    converged = 0
    count = 1
    while not (converged):
        b = _construct_b(flux_old, medium_map, chi, nusigf)
        flux = np.linalg.solve(A, b)

        keff = np.linalg.norm(flux)
        flux /= keff

        change = np.linalg.norm(flux - flux_old)
        converged = (change < EPSILON) or (count >= MAX_COUNT)
        print(f"Count: {count:>2}\tKeff: {keff:.8f}", end="\r")

        count += 1
        flux_old = flux.copy()

    flux = np.reshape(flux, (cells_x + 1, groups), order="F")
    print(f"\nConvergence: {change:2.6e}")
    return flux[:cells_x], keff


def _construct_A(
    diffusion_coef, xs_removal, xs_scatter, boundary, medium_map, edges_x, geometry
):
    cells_x = medium_map.shape[0]
    groups = xs_removal.shape[1]

    # Get geometry
    delta_x = edges_x[1:] - edges_x[:-1]
    surface_area, volume = utils.surface_area_volume(geometry, edges_x)

    # For changing the cell position
    change_matrix = lambda gg, ii: gg * (cells_x + 1) + ii

    A = np.zeros((groups * (cells_x + 1), groups * (cells_x + 1)))
    for gg in range(groups):
        for ii in range(cells_x):
            global_x = change_matrix(gg, ii)
            # Get material properties
            mat = medium_map[ii]
            next_mat = medium_map[ii + 1] if ii < (cells_x - 1) else mat

            dg_avg = (diffusion_coef[mat, gg] * diffusion_coef[next_mat, gg]) / (
                diffusion_coef[mat, gg] + diffusion_coef[next_mat, gg]
            )
            A[global_x, global_x] = (
                2.0 / (delta_x[ii] * volume[ii]) * dg_avg * surface_area[ii + 1]
                + xs_removal[mat, gg]
            )
            A[global_x, global_x + 1] = (
                -2.0 / (delta_x[ii] * volume[ii]) * dg_avg * surface_area[ii + 1]
            )

            if ii > 0:
                prev_mat = medium_map[ii - 1]
                dg_avg = (diffusion_coef[mat, gg] * diffusion_coef[prev_mat, gg]) / (
                    diffusion_coef[mat, gg] + diffusion_coef[prev_mat, gg]
                )
                A[global_x, global_x - 1] = (
                    -2.0 / (delta_x[ii] * volume[ii]) * dg_avg * surface_area[ii]
                )
                A[global_x, global_x] += (
                    2.0 / (delta_x[ii] * volume[ii]) * dg_avg * surface_area[ii]
                )
            for gprime in range(groups):
                if gprime != gg:
                    prime_x = change_matrix(gprime, ii)
                    A[global_x, prime_x] = -xs_scatter[mat, gg, gprime]

        # Boundary conditions
        global_x = change_matrix(gg, cells_x)
        A[global_x, global_x] = 0.5 * boundary[gg, 0] + boundary[gg, 1] / delta_x[-1]
        A[global_x, global_x - 1] = (
            0.5 * boundary[gg, 0] - boundary[gg, 1] / delta_x[-1]
        )

    return A


@numba.jit("f8[:](f8[:], i4[:], f8[:,:], f8[:,:])", nopython=True, cache=True)
def _construct_b(flux, medium_map, chi, nusigf):
    # Get iterables
    mat = numba.int32
    global_x = numba.int32
    ig = numba.int32
    og = numba.int32

    # Get cells and groups
    cells_x = numba.int32(medium_map.shape[0])
    groups = numba.int32(chi.shape[1])

    # Construct b matrix
    b = np.zeros((groups * (cells_x + 1),))

    # Iterate over cells and energy groups
    for global_x in range(b.shape[0]):
        # Separate into spatial cell and energy group
        local_x = global_x % (cells_x + 1)
        ig = int(global_x / (cells_x + 1))

        # Boundary Cells
        if local_x == cells_x:
            continue

        # Get the right material
        mat = medium_map[local_x]

        # Iterate over outgoing energy groups
        for og in range(groups):
            phi_idx = og * (cells_x + 1) + local_x
            b[global_x] += chi[mat, ig] * nusigf[mat, og] * flux[phi_idx]

    return b


if __name__ == "__main__":
    cells_x = 20
    length_x = 76.5535

    diffusion_coef = np.array([[3.850204978408833]])
    xs_scatter = np.array([[[0.0]]])
    xs_absorption = np.array([[0.1532]])
    chi = np.array([[1.0]])
    nusigf = np.array([[0.1570]])
    boundary = np.array([[1.0, 0.0]])

    medium_map = np.zeros((cells_x,), dtype=np.int32)
    edges_x = np.linspace(0, length_x, cells_x + 1)
    geometry = "cylinder"

    flux, keff = power_iteration(
        diffusion_coef,
        xs_scatter,
        xs_absorption,
        chi,
        nusigf,
        boundary,
        medium_map,
        edges_x,
        geometry,
        EPSILON=1e-10,
        MAX_COUNT=100,
    )
    # print(keff)
    book_keff = 1.00001243892
    print(abs(keff - book_keff))
