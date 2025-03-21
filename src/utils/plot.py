#---Plotting of training history---
#import matplotlib
import matplotlib 
import matplotlib.pyplot as plt


#Used when you have trained an discriminator
def discriminator(history):
     #--Visualisize Training---
    # Plot training & validation accuracy
    figure = plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Discriminator Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Discriminator Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.show()

    return figure

#Used when you have trained an autoencoder with mse
def autoencoder(history):
    # Plot training & validation loss
    figure = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Autoencoder Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()
    return figure

#Used when you train an autoencoder adversial 
def adversial(history):

    # Plot training & validation loss
    figure = plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(history["accuracy"])
    #plt.plot(history["reconstruction_loss"])
    plt.title('discriminator accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('Epoch')

    plt.subplot(1, 3, 2)
    plt.plot(history["adversarial_loss"])
    plt.title('adversarial_loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.subplot(1, 3, 3)
    plt.plot(history["reconstruction_loss"])
    plt.title('reconstruction_loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.show()
    return figure

def joint(history):
    """
    Visualizes the first four metrics in the training history dictionary.
    
    :param history: Dictionary containing training metrics.
    """
    # Ensure history has at least four keys
    if len(history) < 4:
        print("Warning: History must contain at least four metrics to plot.")
        return  

    # Select the first four keys
    selected_keys = list(history.keys())[:4]

    # Create figure
    plt.figure(figsize=(12, 6))

    # Loop through the selected keys and create subplots
    for i, key in enumerate(selected_keys, 1):
        plt.subplot(2, 2, i)
        plt.plot(history[key], linestyle='--', marker='o')
        plt.title(key.replace("_", " ").capitalize())  # Format title nicely
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        
    # Adjust layout
    plt.tight_layout()
    plt.show()

def classifier(history):
    #--Visualisize Training---
    # Plot training & validation accuracy
    figure = plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Classifier Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Classifier Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.show()

    return figure