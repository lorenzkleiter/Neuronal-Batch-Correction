from setuptools import setup, find_packages

setup(
    name="NBC",
    version="0.1.0",
    packages=['NBC', 'NBC.models', 'NBC.training'],
    url='https://github.com/lorenzkleiter/Neuronal-Batch-Correction',
    description='An autoencoder-based neural network for batch correction in single-cell data',
    install_requires=[
        "tensorflow==2.19.0",
        "scanpy==1.11.1",
        "scib==1.1.7"
    ],
    python_requires=">=3.10,<3.11"
)