from setuptools import setup, find_packages

setup(
    name="neuronal-batch-correction",
    version="1.0.0",
    packages=find_packages(where="src"),
    install_requires=[
        "tensorflow==2.19.0",
        "scanpy==1.11.1",
        "scib==1.1.7"
    ],
    description="An autoencoder-based neural network for batch correction in single-cell data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/lorenzkleiter/Neuronal-Batch-Correction",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires="==3.10.16",
)