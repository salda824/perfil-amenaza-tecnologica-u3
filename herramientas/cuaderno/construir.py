import sys, nbformat
sys.path.insert(0, '.')
import nbhelp
for parte in ["parte_a", "parte_b", "parte_c", "parte_d"]:
    __import__(parte)
nb = nbformat.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
nb.cells = [nbformat.v4.new_markdown_cell(t) if k == "markdown" else nbformat.v4.new_code_cell(t) for k, t in nbhelp.CELLS]
nbformat.write(nb, sys.argv[1])
print(len(nb.cells), "celdas")
