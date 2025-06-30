# Convertir le notebook en script Python
import nbformat
from nbconvert import PythonExporter
import os

# Définition des chemins 
notebook_path = "/home/sacko/Documents/ProjetAchats/Scripts/fast_api.ipynb"
script_path = "/home/sacko/Documents/ProjetAchats/Scripts/fast_api.py"

# Fonction pour convertir le notebook en script Python
def convert_notebook_to_script(notebook_path, script_path):
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
        exporter = PythonExporter()
        script, _ = exporter.from_notebook_node(nb)

    with open(script_path, 'w') as f:
        f.write(script)

# Exécuter la conversion
convert_notebook_to_script(notebook_path, script_path)
print(f"Le notebook {notebook_path} a été converti en script Python.")



