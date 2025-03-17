#---Pretraining of models---
# 1. autoencoder is trained with mse as loss function to reproduce data
# 2. batch discriminator is trained to distinguish n batches 

#imports
from utils import loading
from models.autoencoder import train_autoencoder, create_autoencoder, autoencode
from models.discriminator import create_discriminator, train_discriminator
import os

#Import Data
"""
Update loading function: to load all data in a directory
"""
test = loading.load('large_atac_gene_activity')

#Create and train Autoencoder on imported Data
autoencoder = create_autoencoder(test, 256, 'relu', 'linear')
autoencoder = train_autoencoder(test, autoencoder, 10, 30)
print(autoencoder.input_shape)

#Autoencode Data
test_autoencoded = autoencode(test, autoencoder)
print(f"Shape of autoencoded_data: {test_autoencoded.shape}")

#Create and train the discriminator on autoencoded Data
discriminator = create_discriminator(test_autoencoded, 256, 128, 0.0075, 'relu')
discriminator = train_discriminator(test_autoencoded, discriminator, 10, 30)
print(f"Shape of discriminator_input: {discriminator.input_shape}")

#Save Discriminator and autoencoder into model directory
file_name = "autoencoder_mselossfunction.keras"
save_path = f"models/saved_models/{file_name}"
autoencoder.save(save_path)
print(f"autoencoder saved to {save_path}")#

file_name = "discriminator_pretrained.keras"
save_path = f"models/saved_models/{file_name}"
discriminator.save(save_path)
print(f"discriminator saved to {save_path}")