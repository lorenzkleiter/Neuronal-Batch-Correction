#---Adverserial Training of model---
# Autoencoder is trained with custom loss function to fool the discriminator

#imports
from utils import loading
from models.autoencoder import adversarial_training
import os

#Import Data
test = loading.load_dataset('large_atac_gene_activity')

#Load previously trained autoencoder and discriminator
discriminator = loading.load_model('discriminator_pretrained')
autoencoder = loading.load_model('autoencoder_mselossfunction')

#Train autoencoder adversially
autoencoder = adversarial_training(test, 20, 30, autoencoder, discriminator)

#Save autoencoder into model directory
file_name = "autoencoder_adverserialtrained.keras"
save_path = f"models/saved_models/{file_name}"
autoencoder.save(save_path)
print(f"autoencoder saved to {save_path}")#