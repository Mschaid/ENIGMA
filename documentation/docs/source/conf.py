import os
import sys

sys.path.insert(0,os.path.abspath('src'))


project = 'ENIGMA'
copyright = '2023, Mike Schaid'
author = 'Mike Schaid'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
]

# Napoleon settings
napoleon_numpy_docstring = True

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'logo_only': False,
    'navigation_depth': 3
}
html_static_path = ['_static']
html_short_title = 'My Project'
html_show_sourcelink = True

