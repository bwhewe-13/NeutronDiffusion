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
