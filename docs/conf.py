"""Sphinx configuration for the ndiffusion documentation."""

from datetime import date

import ndiffusion

project = "ndiffusion"
author = "Ben Whewell"
copyright = f"2021-{date.today().year}, {author}"
release = ndiffusion.__version__
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "doxygen", "archive", "scripts"]

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 3

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "undoc-members": True}
# The pybind11 classes carry their constructor signature in the class
# docstring; repeating it under __init__ adds nothing.
autoclass_content = "class"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# The site's own pages (the Doxygen output under cpp/ in particular) only exist
# once a build has been deployed, so a PR that adds one would fail the check.
linkcheck_ignore = [r"https://bwhewe-13\.github\.io/NeutronDiffusion/.*"]

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

html_theme = "furo"
html_title = f"ndiffusion {release}"
html_static_path = ["_static"]
html_favicon = "_static/favicon.png"
html_theme_options = {
    "light_logo": "logo.svg",
    "dark_logo": "logo-dark.svg",
    "sidebar_hide_name": True,
    "source_repository": "https://github.com/bwhewe-13/NeutronDiffusion",
    "source_branch": "master",
    "source_directory": "docs/",
}
