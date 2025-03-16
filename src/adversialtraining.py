#---Pretraining of models---
# autoencoder weights are updated to oppose the pretrained-batch discriminator

#imports
from utils import loading
from models.autoencoder import train_autoencoder, create_autoencoder, autoencode
from models.discriminator import create_discriminator, train_discriminator
import os 

#Import Data
test = loading.load('large_atac_gene_activity')

#Create and train Autoencoder on imported Data
autoencoder = create_autoencoder(test, 256, 'relu', 'linear')
autoencoder = train_autoencoder(test, autoencoder, 15, 30)

#Autoencode Data
test_autoencoded = autoencode(test, autoencoder)

#Create and train the discriminator on autoencoded Data
discriminator = create_discriminator(test, 256, 128, 0.0075, 'relu')
discriminator = train_discriminator(test, discriminator, 15, 30)

#Save Discriminator into model directory
file_name = "discriminator_256_128_0.0075_15_30.keras"
save_path = f"models/saved_models/{file_name}"
discriminator.save(save_path)