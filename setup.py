from setuptools import setup, find_packages

setup(
    name="diffusion",
    description="""A neutron diffusion solution for slab, spherical, and cylindrical 
    one-dimensional geometries for steady state, k-eigenvalue, and 
    time-dependent problems. Written in Python and accelerated with Numba.
    """,
    version="0.1.0",
    author="Ben Whewell",
    author_email="ben.whewell@pm.me",
    url="https://github.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["numpy", "numba", "pytest"],
)
