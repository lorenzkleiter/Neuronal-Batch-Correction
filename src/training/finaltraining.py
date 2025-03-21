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

#Import discriminator
discriminator = loading.load_model('discriminator')

#Import autoencoder
autoencoder = loading.load_model('autoencoder_mseloss')

#Import Classifier
classifier = loading.load_model('classifier')
"""
#Create the classifier
classifier = create_classifier( test,                        #anndata object: Only necessary to get size
                                64,                          #int: Number of Nodes of the first layer
                                64,                          #int: Number of Nodes of the second layer
                                0.075,                       #int: L2 regularisation amount
                                'relu',                      #str: activation function
                                label_key                    # name of cell type collumn
                              )


# accuracy check
print("autoencode Data: ")
autocorrected_test=autoencode(test, autoencoder)
print("Accuracy Discriminator:")
test_model(discriminator, autocorrected_test, batch_key)
print("Accuracy Classifier:")
test_model(classifier, autocorrected_test, label_key)
"""


#Joint Training of all 2 components not updated discriminator weights
history, autoencoder, classifier = autoencoder_classifier_jointtraining(
                                                                        test,           #anndata object: Dataset
                                                                        30,             #epochs
                                                                        128,             #batch size
                                                                        autoencoder,    #autoencoder: gets updated
                                                                        classifier,     #compiled classifier: gets trained
                                                                        discriminator,  #discriminator: there to check discriminator accuracy
                                                                        0.00001,         #learning rate autoencoder
                                                                        0.001,          #learning rate classifier
                                                                        0.5,            #weighting of reconstructio loss vs. classifier loss
                                                                        True,           #True to enable simulatious adverserial training
                                                                        "uniform",      #loss function: log or uniform
                                                                        True,          #True to freeze classifier weight updating
                                                                        label_key,
                                                                        batch_key
                                                                        )

#Plot history
figure = plot.joint(history)

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