#---Training of cell classifier---
#cell classifier is trained to distinguish n cell types

#imports
from NBC.models.autoencoder import autoencode
from NBC.models.classifier import create_classifier, train_classifier 
import os

def cltrainer(test, label_key, autoencoder):
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

  return classifier
