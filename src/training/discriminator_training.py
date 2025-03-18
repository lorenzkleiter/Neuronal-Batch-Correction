#---Pretraining of batch discriminator---
# batch discriminator is trained to distinguish n batches 

#imports
from utils import loading
from models.autoencoder import plot_ac_training, train_autoencoder, create_autoencoder, autoencode, plot_ac_training
from models.discriminator import create_discriminator, train_discriminator, plot_dc_training 
import os

#Import Data
test = loading.load_dataset('large_atac_gene_activity')

#Test: Dont use autoencoded Input
#Import Autoencoder
#autoencoder = loading.load_model('autoencoder_mseloss')

#Autoencode Data
#test_autoencoded = autoencode(test, autoencoder)

#Create and train the discriminator on autoencoded Data
discriminator = create_discriminator(test, 256, 128, 0.0075, 'relu')
history, discriminator = train_discriminator(test, discriminator, 10, 30, True)
#Plot history
figure = plot_dc_training(history)

#Save Discriminator into model directory
file_name = "discriminator_trained.keras"
save_path = f"models/saved_models/{file_name}"
discriminator.save(save_path)
print(f"discriminator saved to {save_path}")