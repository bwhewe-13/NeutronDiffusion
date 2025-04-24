import pytest
import numpy as np

import diffusion.matrix_solutions as matrix
from diffusion import utils


@pytest.mark.slab1d
@pytest.mark.power_iteration
def test_slab_one_group_one_mat_01():
    cells_x = 20
    length_x = 50.0

    diffusion_coef = np.array([[3.850204978408833]])
    xs_scatter = np.array([[[0.0]]])
    xs_absorption = np.array([[0.1532]])
    chi = np.array([[1.0]])
    nusigf = np.array([[0.1570]])
    boundary = np.array([[1.0, 0.0]])

    medium_map = np.zeros((cells_x,), dtype=np.int32)
    edges_x = np.linspace(0, length_x, cells_x + 1)
    geometry = "slab"

    flux, keff = matrix.power_iteration(
        diffusion_coef,
        xs_scatter,
        xs_absorption,
        chi,
        nusigf,
        boundary,
        medium_map,
        edges_x,
        geometry,
    )

    reference_keff = 1.00001243892
    assert abs(keff - reference_keff) < 1e-5


@pytest.mark.slab1d
@pytest.mark.power_iteration
def test_slab_one_group_two_mat_01():
    cells_x = 100
    length_x = 10.0

    diffusion_coef = np.array([[5.0], [1.0]])
    xs_scatter = np.array([[[0.0]], [[0.0]]])
    xs_absorption = np.array([[0.5], [0.01]])
    chi = np.array([[1.0], [1.0]])
    nusigf = np.array([[0.7], [0.0]])
    boundary = np.array([[1.0, 0.0]])

    layers = [[0, "mat1", "0-5"], [1, "mat2", "5-10"]]
    edges_x = np.linspace(0, length_x, cells_x + 1)
    medium_map = utils.spatial1d(layers, edges_x)
    geometry = "slab"

    flux, keff = matrix.power_iteration(
        diffusion_coef,
        xs_scatter,
        xs_absorption,
        chi,
        nusigf,
        boundary,
        medium_map,
        edges_x,
        geometry,
    )
    reference_keff_exact = 1.2955
    reference_keff_approx = 1.29524
    assert abs(keff - reference_keff_approx) < 1e-3


@pytest.mark.cylinder1d
@pytest.mark.power_iteration
def test_cylinder_one_group_one_mat_01():
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

    flux, keff = matrix.power_iteration(
        diffusion_coef,
        xs_scatter,
        xs_absorption,
        chi,
        nusigf,
        boundary,
        medium_map,
        edges_x,
        geometry,
    )

    reference_keff = 1.00001243892
    assert abs(keff - reference_keff) < 1e-4


@pytest.mark.cylinder1d
@pytest.mark.power_iteration
def test_cylinder_one_group_two_mat_01():
    cells_x = 100
    length_x = 10.0

    diffusion_coef = np.array([[5.0], [1.0]])
    xs_scatter = np.array([[[0.0]], [[0.0]]])
    xs_absorption = np.array([[0.5], [0.01]])
    chi = np.array([[1.0], [1.0]])
    nusigf = np.array([[0.7], [0.0]])
    boundary = np.array([[1.0, 0.0]])

    layers = [[0, "mat1", "0-5"], [1, "mat2", "5-10"]]
    edges_x = np.linspace(0, length_x, cells_x + 1)
    medium_map = utils.spatial1d(layers, edges_x)
    geometry = "cylinder"

    flux, keff = matrix.power_iteration(
        diffusion_coef,
        xs_scatter,
        xs_absorption,
        chi,
        nusigf,
        boundary,
        medium_map,
        edges_x,
        geometry,
    )
    reference_keff_exact = 1.14147
    reference_keff_approx = 1.14068
    assert abs(keff - reference_keff_approx) < 1e-3


@pytest.mark.sphere1d
@pytest.mark.power_iteration
def test_sphere_one_group_one_mat_01():
    cells_x = 20
    length_x = 100.0

    diffusion_coef = np.array([[3.850204978408833]])
    xs_scatter = np.array([[[0.0]]])
    xs_absorption = np.array([[0.1532]])
    chi = np.array([[1.0]])
    nusigf = np.array([[0.1570]])
    boundary = np.array([[1.0, 0.0]])

    medium_map = np.zeros((cells_x,), dtype=np.int32)
    edges_x = np.linspace(0, length_x, cells_x + 1)
    geometry = "sphere"

    flux, keff = matrix.power_iteration(
        diffusion_coef,
        xs_scatter,
        xs_absorption,
        chi,
        nusigf,
        boundary,
        medium_map,
        edges_x,
        geometry,
    )

    reference_keff = 1.00001243892
    assert abs(keff - reference_keff) < 1e-4


@pytest.mark.sphere1d
@pytest.mark.power_iteration
def test_sphere_one_group_two_mat_01():
    cells_x = 150
    length_x = 10.0

    diffusion_coef = np.array([[5.0], [1.0]])
    xs_scatter = np.array([[[0.0]], [[0.0]]])
    xs_absorption = np.array([[0.5], [0.01]])
    chi = np.array([[1.0], [1.0]])
    nusigf = np.array([[0.7], [0.0]])
    boundary = np.array([[1.0, 0.0]])

    layers = [[0, "mat1", "0-5"], [1, "mat2", "5-10"]]
    edges_x = np.linspace(0, length_x, cells_x + 1)
    medium_map = utils.spatial1d(layers, edges_x)
    geometry = "sphere"

    flux, keff = matrix.power_iteration(
        diffusion_coef,
        xs_scatter,
        xs_absorption,
        chi,
        nusigf,
        boundary,
        medium_map,
        edges_x,
        geometry,
    )
    reference_keff_exact = 0.95888
    reference_keff_approx = 0.95735
    assert abs(keff - reference_keff_approx) < 2e-3


@pytest.mark.sphere1d
@pytest.mark.power_iteration
def test_sphere_two_group_one_mat_01():
    cells_x = 20
    length_x = 100.0

    diffusion_coef = np.array([[3.850204978408833, 3.850204978408833]])
    xs_scatter = np.array([[[0.0, 0.0], [0.0, 0.0]]])
    xs_absorption = np.array([[0.1532, 0.1532]])
    chi = np.array([[1.0, 0.0]])
    nusigf = np.array([[0.1570, 0.1570]])
    boundary = np.array([[1.0, 0.0], [1.0, 0.0]])

    medium_map = np.zeros((cells_x,), dtype=np.int32)
    edges_x = np.linspace(0, length_x, cells_x + 1)
    geometry = "sphere"

    flux, keff = matrix.power_iteration(
        diffusion_coef,
        xs_scatter,
        xs_absorption,
        chi,
        nusigf,
        boundary,
        medium_map,
        edges_x,
        geometry,
    )

    reference_keff = 1.0000295511
    assert abs(keff - reference_keff) < 1e-4


@pytest.mark.sphere1d
@pytest.mark.power_iteration
def test_sphere_two_group_one_mat_02():
    cells_x = 50
    length_x = 5.0

    diffusion_coef = np.array([[0.1, 0.1]])
    xs_scatter = np.array([[[0.0, 0.0], [0.0241, 0.0]]])
    xs_removal = np.array([[0.0362, 0.121]])
    chi = np.array([[1.0, 0.0]])
    nusigf = np.array([[0.0085, 0.185]])
    boundary = np.array([[0.0, 1.0], [0.0, 1.0]])

    medium_map = np.zeros((cells_x,), dtype=np.int32)
    edges_x = np.linspace(0, length_x, cells_x + 1)
    geometry = "sphere"

    flux, keff = matrix.power_iteration(
        diffusion_coef,
        xs_scatter,
        xs_removal,
        chi,
        nusigf,
        boundary,
        medium_map,
        edges_x,
        geometry,
    )

    # Test k-effective
    reference_keff = 1.25268252483
    assert np.fabs(keff - reference_keff) < 1e-4

    # Test flux
    reference_flux_ratio = 5.021
    assert np.fabs((flux[0, 0] / flux[0, 1]) - reference_flux_ratio) < 1e-3


@pytest.mark.sphere1d
@pytest.mark.power_iteration
def test_sphere_two_group_two_mat_01():
    cells_x = 100
    length_x = 100.0

    diffusion_coef = np.array([[1.0, 1.0], [1.0, 1.0]])
    xs_scatter = np.array([[[0.0, 0.0], [0.001, 0.0]], [[0.0, 0.0], [0.009, 0.0]]])
    xs_removal = np.array([[0.01, 0.01], [0.01, 0.00049]])
    chi = np.array([[1.0, 0.0], [1.0, 0.0]])
    nusigf = np.array([[0.00085, 0.057], [0.0, 0.0]])
    boundary = np.array([[0.25, 0.5], [0.25, 0.5]])

    layers = [[0, "mat1", "0-50"], [1, "mat2", "50-100"]]
    edges_x = np.linspace(0, length_x, cells_x + 1)
    medium_map = utils.spatial1d(layers, edges_x)
    geometry = "sphere"

    flux, keff = matrix.power_iteration(
        diffusion_coef,
        xs_scatter,
        xs_removal,
        chi,
        nusigf,
        boundary,
        medium_map,
        edges_x,
        geometry,
    )

    reference_keff = 1.06508498598
    assert abs(keff - reference_keff) < 1e-6


@pytest.mark.sphere1d
@pytest.mark.power_iteration
def test_sphere_two_group_one_mat_03():
    cells_x = 100
    length_x = 50.0

    diffusion_coef = np.array([[1.0, 1.0]])
    xs_scatter = np.array([[[0.0, 0.0], [0.001, 0.0]]])
    xs_removal = np.array([[0.01, 0.01]])
    chi = np.array([[1.0, 0.0]])
    nusigf = np.array([[0.00085, 0.057]])
    boundary = np.array([[0.25, 0.5], [0.25, 0.5]])

    edges_x = np.linspace(0, length_x, cells_x + 1)
    medium_map = np.zeros((cells_x,), dtype=np.int32)
    geometry = "sphere"

    flux, keff = matrix.power_iteration(
        diffusion_coef,
        xs_scatter,
        xs_removal,
        chi,
        nusigf,
        boundary,
        medium_map,
        edges_x,
        geometry,
    )

    reference_keff = 0.368702897492
    assert abs(keff - reference_keff) < 1e-6
