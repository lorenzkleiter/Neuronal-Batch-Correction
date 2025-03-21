#---Adverserial Training of model---
# Autoencoder is trained with custom loss function to fool the discriminator

#imports
from utils import loading, plot
from models.autoencoder import adversarial_training,create_autoencoder, autoencode
from models.discriminator import test_discriminator
import os

#Import Data
test = loading.load_dataset('Lung_atlas_public')
label_key = 'cell_type'
batch_key = 'batch'

#Load previously trained autoencoder and discriminator
discriminator = loading.load_model('discriminator')

#Load autoencoder
autoencoder = loading.load_model('autoencoder_mseloss')

"""
#Test: what if we don't pretrain the autoencoder? -> seems to not work: randomized
autoencoder = create_autoencoder(   test,                          #anndata object: Only necessary to get size
                                    256,                           #int: Number of Nodes that the Encoder comprsses the data to
                                    0.3,                           #int: Dropout rate
                                    'relu',                        #str: activation function of the encoder
                                    'linear',                      #str: activation function of the decoder
                                )
"""

"""
#Baseline Accuracy of Discrimiantor
print("Accuracy Discriminator data:")
test_discriminator(discriminator, test, batch_key)
print("autoencode Data: ")
test_ac = autoencode(test, autoencoder)
print("Accuracy Discriminator corrected data:")
test_discriminator(discriminator, test_ac, batch_key)
"""

#Train autoencoder adversially
history, autoencoder = adversarial_training(    test,               #anndata object: Dataset
                                                30,                 #Epochs
                                                64,                 #Batch size
                                                autoencoder,        #compiled autoencoder
                                                discriminator,      #compiled descriminator
                                                "uniform",          #loss function: log or uniform
                                                0.0001,             #learning rate
                                                0.00005,            #L2 regularisation
                                                batch_key           #name of batch collum   
                                           )
#Plot history
plot = plot.adversial(history)

#Save autoencoder into model directory
file_name = "autoencoder_adverserialtrained.keras"
save_path = f"models/saved_models/{file_name}"
autoencoder.save(save_path)
print(f"autoencoder saved to {save_path}")