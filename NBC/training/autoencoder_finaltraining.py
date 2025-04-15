#---Final Training---
# Autoencoder, Cell Classifiers and  discriminator are trained together as all last step
# updating all model’s weights, apart from the discriminator

#imports
from NBC.models.classifier import create_classifier, autoencoder_classifier_jointtraining
from NBC.models.discriminator import test_model
from NBC.models.autoencoder import autoencode, create_autoencoder

def jointtrainer(test, batch_key, label_key, epochs, batch_size, autoencoder, classifier, discriminator):

  print("--Update autoencoder weights - joint training--")
  #Joint Training of all 2 components not updated discriminator weights
  history, autoencoder, classifier = autoencoder_classifier_jointtraining(
                                                                          test,           #anndata object: Dataset
                                                                          epochs,         #epochs
                                                                          batch_size,     #batch size
                                                                          autoencoder,    #autoencoder: gets updated
                                                                          classifier,     #classifier: biological loss
                                                                          discriminator,  #discriminator: adverserial loss
                                                                          0.0001,        #learning rate autoencoder
                                                                          0.001,          #learning rate classifier
                                                                          0.4,            #weighting of reconstructio loss vs. classifier loss
                                                                          True,           #True to enable simulatious adverserial training
                                                                          "log",          #loss function: log or uniform
                                                                          True,          #True to freeze classifier weight updating
                                                                          label_key,
                                                                          batch_key
                                                                          )

  return autoencoder