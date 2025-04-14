from setuptools import setup, find_packages

setup(
    name="neuronal-batch-correction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.19.0",
        "scanpy>=1.11.1",
        "scib>=1.1.7",
        "numpy",
        "scipy",
        "anndata",
        "matplotlib",
        "pandas",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="An autoencoder-based neural network for batch correction in single-cell data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/lorenzkleiter/Neuronal-Batch-Correction",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
)