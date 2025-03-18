#---Adverserial Training of model---
# Autoencoder is trained with custom loss function to fool the discriminator

#imports
from utils import loading
from models.autoencoder import adversarial_training, plot_ad_training,create_autoencoder
import os

#Import Data
test = loading.load_dataset('large_atac_gene_activity')

#Load previously trained autoencoder and discriminator
discriminator = loading.load_model('discriminator_trained')

#autoencoder = loading.load_model('autoencoder_mseloss')
#Test: what if we don't pretrain the autoencoder? -> seems to work maybe
autoencoder = create_autoencoder(test, 256, 'relu', 'linear')

#Train autoencoder adversially
history, autoencoder = adversarial_training(test, 10, 50, autoencoder, discriminator, "loss", 0.000001)
#Plot history
plot = plot_ad_training(history)

#Save autoencoder into model directory
file_name = "autoencoder_adverserialtrained.keras"
save_path = f"models/saved_models/{file_name}"
autoencoder.save(save_path)
print(f"autoencoder saved to {save_path}")