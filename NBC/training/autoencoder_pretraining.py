#---Pretraining of an autoencoder---
# autoencoder is trained with mse as loss function to reproduce data - to get intilisation

#imports
from NBC.models.autoencoder import train_autoencoder, create_autoencoder

def actrainer(test):
    #Create and train Autoencoder on imported Data
    print("--Initilize Autoencoder--")
    autoencoder = create_autoencoder(   test,                          #anndata object: Only necessary to get size
                                        256,                           #int: Number of Nodes that the Encoder comprsses the data to
                                        0.3,                           #int: Dropout rate
                                        'relu',                        #str: activation function of the encoder
                                        'linear',                      #str: activation function of the decoder
                                    )
    print("--Train Autoencoder for mse_loss--")
    history, autoencoder = train_autoencoder( 
                                                test,                           #anndata object: Training set
                                                autoencoder,                    #compiled autoencoder object from create_autoencoder
                                                10,                             #int: number of epochs trained
                                                64,                             #int: Size of Batch
                                                0.0001                          #int: learning rate
                                            )
    return autoencoder
