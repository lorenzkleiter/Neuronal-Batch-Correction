#---Training of cell classifier---
#cell classifier is trained to distinguish n cell types

#imports
from utils import loading, plot
from models.autoencoder import autoencode
from models.classifier import create_classifier, train_classifier 
import os

#Import Data
test = loading.load_dataset('Lung_atlas_public')
label_key = 'cell_type'
batch_key = 'batch'

def cltrainer(test, label_key, batch_key):
  #Import Autoencoder
  autoencoder = loading.load_model('autoencoder_mseloss')

  print("--Autoencode Data--")
  #Autoencode Data
  test_autoencoded = autoencode(test, autoencoder)

  print("--Initilize classifier--")
  #Create the classifier
  classifier = create_classifier( test,                        #anndata object: Only necessary to get size
                                  64,                          #int: Number of Nodes of the first layer
                                  64,                          #int: Number of Nodes of the second layer
                                  0.075,                       #int: L2 regularisation amount
                                  'relu',                      #str: activation function
                                  label_key                    #name of cell type collumn
                                )
  print("--Train classifier--")
  history, classifier = train_classifier(      test_autoencoded,              #autoencoded Dataset
                                                  classifier,                 #compiled discriminator
                                                  50,                            #Epochs
                                                  64,                            #Batch size
                                                  False,                         #shuffle
                                                  label_key                      #name of batch collumn
                                              )

  #Save Discriminator into model directory
  file_name = "classifier.keras"
  save_path = f"models/saved_models/{file_name}"
  classifier.save(save_path)
  print(f"classifier saved to {save_path}")

  return history

history = cltrainer(test, label_key, batch_key)

#Plot history
figure = plot.classifier(history)