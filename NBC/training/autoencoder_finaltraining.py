#---Final Training---
# Autoencoder, Cell Classifiers and  discriminator are trained together as all last step
# updating all model’s weights, apart from the discriminator

#imports
from NBC.utils import loading, plot
from NBC.models.classifier import create_classifier, autoencoder_classifier_jointtraining
from NBC.models.discriminator import test_model
from NBC.models.autoencoder import autoencode, create_autoencoder

def jointtrainer(test, batch_key, label_key, epochs, batch_size):
  #Import discriminator
  discriminator = loading.load_model('discriminator')

  #Import autoencoder
  autoencoder = loading.load_model('autoencoder_mseloss')

  #Import Classifier
  classifier = loading.load_model('classifier')

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

  #Save autoencoder into model directory
  file_name = "autoencoder_final_onestep.keras"
  save_path = f"NBC/models/{file_name}"
  autoencoder.save(save_path)
  print(f"autoencoder saved to {save_path}")

  figure = plot.joint(history)

  return history

#history = jointtrainer(test, label_key, batch_key, epochs)
#figure = plot.joint(history)
