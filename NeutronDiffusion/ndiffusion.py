
from NeutronDiffusion.create import selection

import numpy as np
import numba
# from scipy import sparse
from scipy import sparse 
import scipy.sparse.linalg

TOLERANCE = 1e-10
ITERATIONS = 200

def run_diffusion(problem, groups, cells, geo, BC=0):
    _, widths, _, dg, scatter, chi, fission, removal, BC \
        = selection(problem, groups, cells, BC=BC)
    return solve_diffusion(groups, cells, widths, dg, scatter, chi, \
                            fission, removal, BC, geo)

def solve_diffusion(groups, cells, widths, dg, scatter, chi, fission, \
                    removal, BC, geo):
    cell_width = float(sum(widths)) / cells
    layers = np.array([int(round(mat / cell_width)) for mat in widths],dtype=np.int32)
    assert sum(layers) == cells
    flux_old = np.random.random(groups * (cells+1))
    flux_old /= np.linalg.norm(flux_old)
    A = construct_A(groups, cells, cell_width, dg, removal, scatter, BC, layers, geo)
    print("Constructed A!")
    converged = 0
    count = 1
    while not(converged):
        if chi is None:
            flux = sparse.linalg.spsolve(A, construct_b_fission(groups, \
                                        cells, flux_old, fission, layers))
        else:
            flux = sparse.linalg.spsolve(A, construct_b(groups, cells, \
                                        flux_old, chi, fission, layers))            
        keff = np.linalg.norm(flux)
        flux /= keff
        change = np.linalg.norm(flux - flux_old)
        converged = (change < TOLERANCE) or (count >= ITERATIONS)
        print('Iteration: {} Change {}\tKeff {}'.format(count, change, keff))
        count += 1
        flux_old = flux.copy()
    flux = np.reshape(flux, (cells+1, groups), order='F')
    return flux[:cells], keff        
    
@numba.jit(nopython=True, cache=True)
def tri_diagonal(cells, group, cell_width, dg, removal, BC, V, SA, layers):
    diagonal = np.zeros((cells+1))
    Udiagonal = np.zeros((cells))
    Ldiagonal = np.zeros((cells))
    idx = 0
    mat_cell = 0
    cumulative = np.cumsum(layers)
    for cell in range(cells):
        if cell == 0:
            pass
        elif (mat_cell % layers[idx]) == (layers[idx] - 1):
            mat_cell = 0
            idx += 1
        else:
            mat_cell += 1
        add = 1 if (cell == cumulative[idx] - 1) and (cell != (cells - 1)) else 0
        sub = 1 if (cell == cumulative[idx - 1]) else 0
        diagonal[cell] = removal[idx][group] + 2.0 / (cell_width*V[cell]) * \
                        ((dg[idx][group] * dg[idx+add][group]) \
                        /(dg[idx][group] + dg[idx+add][group]) * SA[cell+1]) 
        Udiagonal[cell] = -2.0*(dg[idx][group] * dg[idx+add][group]) \
                        / (dg[idx][group] + dg[idx+add][group]) \
                        / (cell_width * V[cell]) * SA[cell+1]
        if cell != 0:
            diagonal[cell] += 2.0/(cell_width*V[cell]) * ((dg[idx][group]*dg[idx-sub][group]) \
                                / (dg[idx][group] + dg[idx-sub][group]) * SA[cell])
            Ldiagonal[cell-1] = -2.0*(dg[idx][group] * dg[idx-sub][group]) \
                                / (dg[idx][group] + dg[idx-sub][group]) \
                                / (cell_width * V[cell]) * SA[cell]
    diagonal[cells] = BC[group][0]*0.5 + BC[group][1] / cell_width
    Ldiagonal[cells-1] = BC[group][0]*0.5 - BC[group][1] / cell_width
    return Udiagonal, diagonal, Ldiagonal

# @numba.jit(nopython=True, cache=True)
def off_diagonal(groups, cells, scatter, layers):
    off_scatter_val = []
    off_scatter_loc = []
    cumulative = np.cumsum(layers)
    for group in range(1, groups):
        down_scatter = np.repeat(np.diag(-scatter[0], -group), cells+1)
        up_scatter = np.repeat(np.diag(-scatter[0], group), cells+1)        
        for num in range(len(layers) - 1):
            idx = np.sort(np.array([np.arange(ii, (groups-group)*(cells+1), \
                            cells+1) for ii in range(cumulative[num], \
                            cumulative[num+1])]).flatten())
            temp_down = np.repeat(np.diag(-scatter[num+1], -group), layers[num+1])
            down_scatter[idx] = temp_down.copy()
            temp_up = np.repeat(np.diag(-scatter[num+1], group), layers[num+1])
            up_scatter[idx] = temp_up.copy()
        down_scatter[cells::cells+1] *= 0
        up_scatter[cells::cells+1] *= 0
        off_scatter_val.append(down_scatter)
        off_scatter_loc.append(-group * (cells+1))
        off_scatter_val.append(up_scatter)
        off_scatter_loc.append(group * (cells+1))
    return off_scatter_val, off_scatter_loc

@numba.jit(nopython=True, cache=True)
def on_diagonal(groups, cells, cell_width, dg, removal, BC, V, SA, layers):
    upper = np.zeros((0), dtype=np.float64)
    center = np.zeros((0), dtype=np.float64)
    lower = np.zeros((0), dtype=np.float64)
    for group in range(groups):
        up, mid, low = tri_diagonal(cells, group, cell_width, dg, removal, \
                                    BC, V, SA, layers)
        center = np.append(center, mid)
        if group == (groups - 1):
            upper = np.append(upper, up)
            lower = np.append(lower, low)
        else:
            upper = np.append(upper, np.append(up, 0))
            lower = np.append(lower, np.append(low, 0))
    return upper, center, lower

def construct_A(groups, cells, cell_width, dg, removal, scatter, BC, layers, geo):
    SA, V = geometry(cells, cell_width, geo) 
    upper, center, lower = on_diagonal(groups, cells, cell_width, dg, \
                                        removal, BC, V, SA, layers)
    off_scatter_val, off_scatter_loc = off_diagonal(groups, cells, scatter, layers)
    diagonals = [upper.flatten(), center.flatten(), lower.flatten()] + off_scatter_val
    locations = [1, 0, -1] + off_scatter_loc
    A = sparse.diags(diagonals, locations, format="csr")
    return A

@numba.jit(nopython=True, cache=True)
def geometry(cells, cell_width, geo):
    edges = np.arange(cells+1) * cell_width
    if (geo == 'slab'): 
        SA = 0.0 * edges + 1 
        SA[0] = 0.0 
        V = 0.0 * edges[:cells] + cell_width
    elif (geo == 'cylinder'): 
        SA = 2.0 * np.pi * edges
        V = np.pi * (edges[1:(cells+1)]**2 - edges[0:cells]**2) 
    elif (geo == 'sphere'): 
        SA = 4.0 * np.pi * edges**2
        V = 4.0/3.0 * np.pi * (edges[1:(cells+1)]**3 - edges[0:cells]**3) 
    return SA, V        

@numba.jit(nopython=True, cache=True)
def construct_b_fission(groups, cells, flux, fission, layers):
    b = np.zeros((groups*(cells+1)))
    idx = 0
    mat_cell = 0
    for global_x in range(groups*(cells+1)):
        local_cell = global_x % (cells+1)
        if local_cell == (cells):
            continue
        if local_cell == 0:
            idx = 0
            mat_cell = 0
        elif mat_cell % layers[idx] == layers[idx] - 1:
            mat_cell = 0
            idx += 1
        else:
            mat_cell += 1
        group_in = int(global_x / (cells+1))
        for group_out in range(groups):
            global_y = group_out * (cells+1) + local_cell
            b[global_x] += fission[idx][group_in,group_out] * flux[global_y]
    return b 

@numba.jit(nopython=True, cache=True)
def construct_b(groups, cells, flux, chi, fission, layers):
    b = np.zeros((groups*(cells+1)))
    idx = 0
    mat_cell = 0
    for global_x in range(groups*(cells+1)):
        local_cell = global_x % (cells+1)
        if local_cell == (cells):
            continue
        if local_cell == 0:
            idx = 0
            mat_cell = 0
        elif mat_cell % layers[idx] == layers[idx] - 1:
            mat_cell = 0
            idx += 1
        else:
            mat_cell += 1
        group_in = int(global_x / (cells+1))
        for group_out in range(groups):
            global_y = group_out * (cells+1) + local_cell
            b[global_x] += chi[idx][group_in] * fission[idx][group_out] * flux[global_y]
    return b 