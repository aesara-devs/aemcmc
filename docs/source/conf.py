import os

import aemcmc

# -- Project information

project = "aemcmc"
author = "Aesara Developers"
copyright = f"2021-2023, {author}"

version = aemcmc.__version__
if os.environ.get("READTHEDOCS", False):
    rtd_version = os.environ.get("READTHEDOCS_VERSION", "")
    if "." not in rtd_version and rtd_version.lower() != "stable":
        version = "dev"
else:
    branch_name = os.environ.get("BUILD_SOURCEBRANCHNAME", "")
    if branch_name == "main":
        version = "dev"
release = version


# -- General configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_math_dollar",
    "sphinx_design",
    "sphinx_copybutton",
]

templates_path = ["_templates"]

exclude_patterns = [
    "Thumbs.db",
    ".DS_Store",
    ".ipynb_checkpoints",
]

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "code"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

autodoc_typehints = "none"

# -- Options for extensions

# -- Options for HTML output

html_static_path = ["_static"]
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/aesara-devs/aemcmc",
    "use_repository_button": True,
    "use_download_button": False,
}
html_title = ""
html_logo = "_static/aemcmc_logo.png"

intersphinx_mapping = {
    "aesara": ("https://aesara.readthedocs.io/en/latest", None),
    "aeppl": ("https://aeppl.readthedocs.io/en/latest", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
}
