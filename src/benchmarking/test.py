from utils import loading
from utils.scores import scoring_autoencoder
import pandas as pd
import scanpy as sc


#Import Data
dataset = 'Lung_atlas_public'
test = loading.load_dataset(dataset)
label_key = 'cell_type'
batch_key = 'batch'

#load autoencoder
autoencoder = loading.load_model("autoencoder_final_onestep")

#Score Integration
result = scoring_autoencoder(test, autoencoder, label_key, batch_key, True)

print(result)