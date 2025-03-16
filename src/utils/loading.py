#---Module for loading data---
#imports
import os
import scanpy as sc
import anndata

def load(name):
    #set base path to load data: goes back one directory and then into the data
    base_path = os.path.join('..', 'data')

    # read dataset into an anndata object:  Category - Cells of the brain
    inPath = os.path.join(base_path, f"{name}.h5ad")
    adata = sc.read(inPath)
    return adata