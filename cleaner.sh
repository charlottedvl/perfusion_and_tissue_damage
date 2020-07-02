find . -name permeability -type d -exec rm -r {} ';'
find . -name boundary_data -type d -exec rm -r {} ';'
find . -name __pycache__ -type d -exec rm -r {} ';'
rm -r VP_results
