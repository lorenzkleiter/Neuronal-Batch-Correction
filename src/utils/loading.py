#---Module for loading data---
#imports
import os
import scanpy as sc
import anndata
import tensorflow as tf

def load_dataset(name):
    #set base path to load data: goes back one directory and then into the data
    base_path = os.path.join('..', 'data')

    # read dataset into an anndata object:  Category - Cells of the brain
    inPath = os.path.join(base_path, f"{name}.h5ad")
    adata = sc.read(inPath)
    return adata

def load_model(name):
    #set base path to load data: goes back one directory and then into the data
    base_path = os.path.join('..', 'src', 'models', 'saved_models')

    # load autoencoder
    inPath = os.path.join(base_path, f"{name}.keras")
    model = tf.keras.models.load_model(inPath)
    return model
