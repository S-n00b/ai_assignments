# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------

project = 'Lenovo AAITC Solutions'
copyright = '2025, Lenovo Advanced AI Technology Center'
author = 'Lenovo AAITC Team'
release = '1.0.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.graphviz',
    'sphinx.ext.imgmath',
    'sphinx.ext.imgconverter',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.linkcode',
    'sphinx_rtd_theme',
    'myst_parser',
    'sphinx_copybutton',
    'sphinx_tabs.tabs',
    'sphinxcontrib.mermaid',
    'sphinx_jinja',
    'sphinx_gallery.gen_gallery',
    'sphinx_panels',
    'sphinx_design',
    'sphinxcontrib.bibtex',
    'sphinxcontrib.plantuml',
    'sphinxcontrib.httpdomain',
    'sphinxcontrib.openapi',
    'sphinxcontrib.redoc',
    'sphinxcontrib.swaggerui',
    'sphinxcontrib.jsonschema',
    'sphinxcontrib.youtube',
    'sphinxcontrib.video',
    'sphinxcontrib.aafig',
    'sphinxcontrib.seqdiag',
    'sphinxcontrib.actdiag',
    'sphinxcontrib.nwdiag',
    'sphinxcontrib.blockdiag',
    'sphinxcontrib.rackdiag',
    'sphinxcontrib.packetdiag',
    'sphinxcontrib.c4',
    'sphinxcontrib.mermaid',
    'sphinxcontrib.plantuml',
    'sphinxcontrib.httpdomain',
    'sphinxcontrib.openapi',
    'sphinxcontrib.redoc',
    'sphinxcontrib.swaggerui',
    'sphinxcontrib.jsonschema',
    'sphinxcontrib.youtube',
    'sphinxcontrib.video',
    'sphinxcontrib.aafig',
    'sphinxcontrib.seqdiag',
    'sphinxcontrib.actdiag',
    'sphinxcontrib.nwdiag',
    'sphinxcontrib.blockdiag',
    'sphinxcontrib.rackdiag',
    'sphinxcontrib.packetdiag',
    'sphinxcontrib.c4',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
html_sidebars = {
    '**': [
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
    ]
}

# -- Extension configuration -------------------------------------------------

# -- Options for autodoc extension -------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# -- Options for autosummary extension ---------------------------------------
autosummary_generate = True

# -- Options for napoleon extension ------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for intersphinx extension ---------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'plotly': ('https://plotly.com/python-api-reference/', None),
    'gradio': ('https://gradio.app/docs/', None),
    'fastapi': ('https://fastapi.tiangolo.com/', None),
    'pydantic': ('https://pydantic-docs.helpmanual.io/', None),
    'typing': ('https://docs.python.org/3/library/typing.html', None),
}

# -- Options for todo extension ----------------------------------------------
todo_include_todos = True

# -- Options for coverage extension ------------------------------------------
coverage_show_missing_items = True

# -- Options for linkcode extension ------------------------------------------
def linkcode_resolve(domain, info):
    """Determine the URL corresponding to Python object."""
    if domain != 'py':
        return None
    if not info['module']:
        return None
    
    filename = info['module'].replace('.', '/')
    return f"https://github.com/s-n00b/ai_assignments/blob/main/src/{filename}.py"

# -- Options for MyST parser -------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# -- Options for sphinx-copybutton -------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_only_copy_prompt_lines = True

# -- Options for sphinx-tabs -------------------------------------------------
sphinx_tabs_disable_tab_closing = True

# -- Options for sphinx-gallery ----------------------------------------------
sphinx_gallery_conf = {
    'examples_dirs': '../../examples',
    'gallery_dirs': 'auto_examples',
    'filename_pattern': '/plot_',
    'ignore_pattern': r'__init__\.py',
    'plot_gallery': 'True',
    'download_all_examples': False,
    'download_section_examples': True,
    'line_numbers': False,
    'within_subsection_order': FileNameSortKey,
    'compress_images': ('images', 'thumbnails'),
    'docstring_style': 'google',
    'image_scrapers': ('matplotlib',),
    'matplotlib_animations': True,
    'remove_config_comments': True,
    'show_memory': True,
    'show_signature': True,
    'junit': '../../test-results/sphinx-gallery/junit.xml',
    'log_level': {'backreference_missing': 'warning'},
    'binder': {
        'org': 's-n00b',
        'repo': 'ai_assignments',
        'branch': 'main',
        'binderhub_url': 'https://mybinder.org',
        'dependencies': '../../requirements.txt',
        'notebooks_dir': 'notebooks',
        'use_jupyter_lab': True,
    },
    'notebook_images': 'https://mybinder.org/v2/gh/s-n00b/ai_assignments/main?filepath=notebooks/',
}

# -- Options for sphinx-panels -----------------------------------------------
panels_add_bootstrap_css = False

# -- Options for sphinx-design -----------------------------------------------
sphinx_design_theme = 'bootstrap'

# -- Options for sphinxcontrib-bibtex ----------------------------------------
bibtex_bibfiles = ['references.bib']

# -- Options for sphinxcontrib-plantuml --------------------------------------
plantuml = 'java -jar /usr/share/plantuml/plantuml.jar'
plantuml_output_format = 'svg'

# -- Options for sphinxcontrib-httpdomain ------------------------------------
http_index_shortname = 'API'
http_index_localname = 'API Reference'

# -- Options for sphinxcontrib-openapi ---------------------------------------
openapi_redoc_uri = 'https://cdn.jsdelivr.net/npm/redoc@2.0.0/bundles/redoc.standalone.js'

# -- Options for sphinxcontrib-redoc -----------------------------------------
redoc_uri = 'https://cdn.jsdelivr.net/npm/redoc@2.0.0/bundles/redoc.standalone.js'

# -- Options for sphinxcontrib-swaggerui -------------------------------------
swagger_ui_uri = 'https://cdn.jsdelivr.net/npm/swagger-ui-dist@4.15.5'

# -- Options for sphinxcontrib-jsonschema ------------------------------------
jsonschema_uri = 'https://json-schema.org/draft/2020-12/schema'

# -- Options for sphinxcontrib-youtube ---------------------------------------
youtube_video_width = 560
youtube_video_height = 315

# -- Options for sphinxcontrib-video -----------------------------------------
video_uri = 'https://cdn.jsdelivr.net/npm/video.js@8.5.2/dist/video.min.js'

# -- Options for sphinxcontrib-aafig -----------------------------------------
aafig_format = 'svg'

# -- Options for sphinxcontrib-seqdiag ---------------------------------------
seqdiag_html_image_format = 'SVG'
seqdiag_latex_image_format = 'PDF'
seqdiag_antialias = True

# -- Options for sphinxcontrib-actdiag ---------------------------------------
actdiag_html_image_format = 'SVG'
actdiag_latex_image_format = 'PDF'
actdiag_antialias = True

# -- Options for sphinxcontrib-nwdiag ----------------------------------------
nwdiag_html_image_format = 'SVG'
nwdiag_latex_image_format = 'PDF'
nwdiag_antialias = True

# -- Options for sphinxcontrib-blockdiag -------------------------------------
blockdiag_html_image_format = 'SVG'
blockdiag_latex_image_format = 'PDF'
blockdiag_antialias = True

# -- Options for sphinxcontrib-rackdiag --------------------------------------
rackdiag_html_image_format = 'SVG'
rackdiag_latex_image_format = 'PDF'
rackdiag_antialias = True

# -- Options for sphinxcontrib-packetdiag ------------------------------------
packetdiag_html_image_format = 'SVG'
packetdiag_latex_image_format = 'PDF'
packetdiag_antialias = True

# -- Options for sphinxcontrib-c4 --------------------------------------------
c4_uri = 'https://cdn.jsdelivr.net/npm/c4-plantuml@1.0.0'

# -- Options for sphinxcontrib-mermaid ---------------------------------------
mermaid_version = '9.4.3'

# -- Options for sphinxcontrib-plantuml --------------------------------------
plantuml = 'java -jar /usr/share/plantuml/plantuml.jar'
plantuml_output_format = 'svg'

# -- Options for sphinxcontrib-httpdomain ------------------------------------
http_index_shortname = 'API'
http_index_localname = 'API Reference'

# -- Options for sphinxcontrib-openapi ---------------------------------------
openapi_redoc_uri = 'https://cdn.jsdelivr.net/npm/redoc@2.0.0/bundles/redoc.standalone.js'

# -- Options for sphinxcontrib-redoc -----------------------------------------
redoc_uri = 'https://cdn.jsdelivr.net/npm/redoc@2.0.0/bundles/redoc.standalone.js'

# -- Options for sphinxcontrib-swaggerui -------------------------------------
swagger_ui_uri = 'https://cdn.jsdelivr.net/npm/swagger-ui-dist@4.15.5'

# -- Options for sphinxcontrib-jsonschema ------------------------------------
jsonschema_uri = 'https://json-schema.org/draft/2020-12/schema'

# -- Options for sphinxcontrib-youtube ---------------------------------------
youtube_video_width = 560
youtube_video_height = 315

# -- Options for sphinxcontrib-video -----------------------------------------
video_uri = 'https://cdn.jsdelivr.net/npm/video.js@8.5.2/dist/video.min.js'

# -- Options for sphinxcontrib-aafig -----------------------------------------
aafig_format = 'svg'

# -- Options for sphinxcontrib-seqdiag ---------------------------------------
seqdiag_html_image_format = 'SVG'
seqdiag_latex_image_format = 'PDF'
seqdiag_antialias = True

# -- Options for sphinxcontrib-actdiag ---------------------------------------
actdiag_html_image_format = 'SVG'
actdiag_latex_image_format = 'PDF'
actdiag_antialias = True

# -- Options for sphinxcontrib-nwdiag ----------------------------------------
nwdiag_html_image_format = 'SVG'
nwdiag_latex_image_format = 'PDF'
nwdiag_antialias = True

# -- Options for sphinxcontrib-blockdiag -------------------------------------
blockdiag_html_image_format = 'SVG'
blockdiag_latex_image_format = 'PDF'
blockdiag_antialias = True

# -- Options for sphinxcontrib-rackdiag --------------------------------------
rackdiag_html_image_format = 'SVG'
rackdiag_latex_image_format = 'PDF'
rackdiag_antialias = True

# -- Options for sphinxcontrib-packetdiag ------------------------------------
packetdiag_html_image_format = 'SVG'
packetdiag_latex_image_format = 'PDF'
packetdiag_antialias = True

# -- Options for sphinxcontrib-c4 --------------------------------------------
c4_uri = 'https://cdn.jsdelivr.net/npm/c4-plantuml@1.0.0'

# -- Options for sphinxcontrib-mermaid ---------------------------------------
mermaid_version = '9.4.3'
