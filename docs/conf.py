# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('/mnt/src/Gem_X/functions'))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Gemini'
copyright = '2025, Adam Brierley, Charlotte Devillé'
author = 'Adam Brierley, Charlotte Devillé'
master_doc = 'index'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
]

autodoc_mock_imports = ['basix', 'basix.ufl', 'dolfinx']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

# -- Options for function output -----------------------------------------------
autoclass_content = 'both'  # Include class and __init__ docstrings
autodoc_member_order = 'bysource'  # Order members as they appear in the source

html_theme_options = {
    "show_related": True,
    "show_toc_level": 1,  # Only show top-level (document titles)
}