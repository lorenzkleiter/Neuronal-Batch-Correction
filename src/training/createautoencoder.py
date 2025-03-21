
#imports
from utils import loading
from models.autoencoder import create_autoencoder
import os

#Import Data
test = loading.load_dataset('Lung_atlas_public')
label_key = 'cell_type'
batch_key = 'batch'

#Load previously trained autoencoder and discriminator
discriminator = loading.load_model('discriminator')
autoencoder = loading.load_model('autoencoder_mseloss')

"""
#Test: what if we don't pretrain the autoencoder? -> seems to not work: random init.
autoencoder = create_autoencoder(   test,                          #anndata object: Only necessary to get size
                                    256,                           #int: Number of Nodes that the Encoder comprsses the data to
                                    0.3,                           #int: Dropout rate
                                    'relu',                        #str: activation function of the encoder
                                    'linear',                      #str: activation function of the decoder
                                )
"""

#Save autoencoder into model directory
file_name = "autoencoder_untrained.keras"
save_path = f"models/saved_models/{file_name}"
autoencoder.save(save_path)
print(f"autoencoder saved to {save_path}")