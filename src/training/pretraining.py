#---Pretraining of an autoencoder---
# autoencoder is trained with mse as loss function to reproduce data - to get intilisation

#imports
from utils import loading
from models.autoencoder import plot_ac_training, train_autoencoder, create_autoencoder, autoencode, plot_ac_training
from models.discriminator import create_discriminator, train_discriminator, plot_dc_training 
import os

#Import Data
test = loading.load_dataset('large_atac_gene_activity')

#Create and train Autoencoder on imported Data
autoencoder = create_autoencoder(test, 256, 'relu', 'linear')
history, autoencoder = train_autoencoder(test, autoencoder, 1, 300)
#Plot history
figure = plot_ac_training(history)

#Save autoencoder into model directory
file_name = "autoencoder_mseloss.keras"
save_path = f"models/saved_models/{file_name}"
autoencoder.save(save_path)
print(f"autoencoder saved to {save_path}")