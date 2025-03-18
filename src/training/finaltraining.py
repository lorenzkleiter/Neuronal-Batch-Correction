#---Final Training---
# Autoencoder, Cell Classifiers and  discriminator are trained together as all last step
# updating all model’s weights, apart from the discriminator

#imports
from utils import loading
from models.classifier import create_classifier, autoencoder_classifier_jointtraining, plot_f_training

#Import Data
test = loading.load_dataset('large_atac_gene_activity')

#Import Autoencoder
autoencoder = loading.load_model('autoencoder_adverserialtrained')

#Create the classifier
classifier = create_classifier(test, 256, 128, 0.0075, 'relu')

#Joint Training
history, autoencoder, classifier = autoencoder_classifier_jointtraining(test, 10, 30, autoencoder, classifier, 0.0001, 0.5)

#Plot history
figure = plot_f_training(history)

#Save autoencoder into model directory
file_name = "autoencoder_final.keras"
save_path = f"models/saved_models/{file_name}"
autoencoder.save(save_path)
print(f"autoencoder saved to {save_path}")

#Save classifier into model directory
file_name = "classifier.keras"
save_path = f"models/saved_models/{file_name}"
classifier.save(save_path)
print(f"classifier saved to {save_path}")