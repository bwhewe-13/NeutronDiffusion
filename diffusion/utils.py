import numpy as np


def surface_area_volume(geometry, edges_x):
    # Get cell edge locations
    cells_x = edges_x.shape[0] - 1

    # Get surface area and volume
    if geometry == "slab":
        surface_area = np.ones((cells_x + 1,))
        volume = edges_x[1:] - edges_x[:-1]

    # SA = 2 * pi * r, V = pi * (r^2 - r^2)
    elif geometry == "cylinder":
        surface_area = 2 * np.pi * edges_x
        volume = np.pi * (edges_x[1:] ** 2 - edges_x[:-1] ** 2)

    # SA = 4 * pi^2, V = 4/3 * pi * (r^3 - r^3)
    elif geometry == "sphere":
        surface_area = 4 * np.pi * edges_x**2
        volume = 4 / 3.0 * np.pi * (edges_x[1:] ** 3 - edges_x[:-1] ** 3)

    return surface_area, volume


def spatial1d(layers, edges_x, labels=False, check=True):
    """Creating one-dimensional medium map

    :param layers: list of lists where each layer is a new material. A
        layer is comprised of an index (int), material name (str), and
        the width (str) in the form [index, material, width]. The width
        is the starting and ending points of the material (in cm)
        separated by a dash. If there are multiple regions, a comma can
        separate them. I.E. layer = [0, "plutonium", "0 - 2, 3 - 4"].
    :param edges_x: Array of length I + 1 with the location of the cell edges
    :return: One-dimensional array of length I, identifying the locations
        of the materials
    """
    if labels:
        # Initialize label map
        medium_map = -1 * np.ones((len(edges_x) - 1))
    else:
        # Initialize medium_map
        medium_map = -1 * np.ones((len(edges_x) - 1), dtype=np.int32)
    # Iterate over all layers
    for layer in layers:
        for region in layer[2].split(","):
            start, stop = region.split("-")
            idx1 = np.argmin(np.fabs(float(start) - edges_x))
            idx2 = np.argmin(np.fabs(float(stop) - edges_x))
            medium_map[idx1:idx2] = layer[0]
    # Verify all cells are filled
    if check:
        assert np.all(medium_map != -1)

    return medium_map
