#---Training of batch discriminator---
# batch discriminator is trained to distinguish n batches 

#imports
from utils import loading, plot
from models.autoencoder import train_autoencoder, create_autoencoder, autoencode
from models.discriminator import create_discriminator, train_discriminator 
import os

#Import Data
test = loading.load_dataset('Lung_atlas_public')
label_key = 'cell_type'
batch_key = 'batch'

#Function for training discriminator
def dctrainer(test, label_key, batch_key):
    #Import Autoencoder
    autoencoder = loading.load_model('autoencoder_mseloss')

    print("--Autoencode Data--")
    #Autoencode Data
    test_autoencoded = autoencode(test, autoencoder)

    print("--Initilize discriminator--")
    #Create and train the discriminator on (not) autoencoded Data
    discriminator = create_discriminator(   test_autoencoded,               #anndata object: Only necessary to get size
                                            64,                             #Number of Nodes of the first layer
                                            64,                             #Number of Nodes of the second layer
                                            0.075,                          #L2 regularisation amount
                                            0.001,                          #learning rate of adam optimizer                       
                                            'relu',                         #activation function used for first 2 layers
                                            batch_key                       #name of batch collumn
                                        )
    print("--Train discriminator--")
    history, discriminator = train_discriminator(   test_autoencoded,              #autoencoded Dataset
                                                    discriminator,                 #compiled discriminator
                                                    50,                            #Epochs
                                                    64,                            #Batch size
                                                    False,                         #shuffle
                                                    batch_key                      #name of batch collumn
                                                )

    #Save Discriminator into model directory
    file_name = "discriminator.keras"
    save_path = f"models/saved_models/{file_name}"
    discriminator.save(save_path)
    print(f"discriminator saved to {save_path}")

    return history

#run function
history = dctrainer(test, label_key, batch_key)

#Plot history
figure = plot.discriminator(history)