#---Final Training---
# Autoencoder, Cell Classifiers and  discriminator are trained together as all last step
# updating all model’s weights, apart from the discriminator

#imports
from utils import loading, plot
from models.classifier import create_classifier, autoencoder_classifier_jointtraining
from models.discriminator import test_model
from models.autoencoder import autoencode, create_autoencoder

#Import Data
test = loading.load_dataset('Lung_atlas_public')
label_key = 'cell_type'
batch_key = 'batch'
epochs = 30

def jointtrainer(test, label_key, batch_key, epochs):
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
                                                                          epochs,             #epochs
                                                                          128,            #batch size
                                                                          autoencoder,    #autoencoder: gets updated
                                                                          classifier,     #classifier: biological loss
                                                                          discriminator,  #discriminator: adverserial loss
                                                                          0.00001,        #learning rate autoencoder
                                                                          0.001,          #learning rate classifier
                                                                          0.5,            #weighting of reconstructio loss vs. classifier loss
                                                                          True,           #True to enable simulatious adverserial training
                                                                          "uniform",      #loss function: log or uniform
                                                                          True,          #True to freeze classifier weight updating
                                                                          label_key,
                                                                          batch_key
                                                                          )

  #Save autoencoder into model directory
  file_name = "autoencoder_final_onestep.keras"
  save_path = f"models/saved_models/{file_name}"
  autoencoder.save(save_path)
  print(f"autoencoder saved to {save_path}")

  #Save classifier into model directory
  file_name = "classifier.keras"
  save_path = f"models/saved_models/{file_name}"
  classifier.save(save_path)
  print(f"classifier saved to {save_path}")

  return history

history = jointtrainer(test, label_key, batch_key, epochs)
#Plot history
figure = plot.joint(history)
