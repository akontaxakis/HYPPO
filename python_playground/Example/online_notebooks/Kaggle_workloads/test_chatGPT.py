import nbformat
from nbconvert import PythonExporter
from IPython.display import display

def notebook_to_components(notebook_path):
    # Read the notebook file
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=nbformat.NO_CONVERT)

    # Initialize the PythonExporter
    exporter = PythonExporter()

    # Convert the notebook to Python code
    (python_code, _) = exporter.from_notebook_node(notebook)

    # Split the Python code into Dictionary
    components = python_code.split("#%%\n")

    # Display the Dictionary
    for i, component in enumerate(components):
        display(component)

# Provide the path to your notebook file
notebook_path = 'C:/Users/adoko/PycharmProjects/pythonProject1/Kaggle_workloads/intro-to-model-tuning-grid-and-random-search.ipynb'
if __name__ == '__main__':
    notebook_to_components(notebook_path)