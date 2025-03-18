#---Classifier---
#---Inports---
#import numpy and keras modules
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
import tensorflow as tf

#import Scanpy modules
import scanpy as sc
import anndata as ad

#import matplotlib
import matplotlib 
import matplotlib.pyplot as plt
import pandas as pd

#import sklearn modules
from sklearn.preprocessing import OneHotEncoder

# Set a random seed for reproducibility
random_seed = 42

def create_classifier( 
                        adata,                           #anndata object: Only necessary to get size
                        N_HIDDEN_1,                      #int: Number of Nodes of the first layer
                        N_HIDDEN_2,                      #int: Number of Nodes of the second layer
                        L2_LAMBDA,                       #int: regularisation amount
                        ACTIVATION_FUNCTION,
                        random_seed=42,
                        ):
    #---Prepare Data---
    INPUT =  adata.X.toarray()
    #Encode the cell labels as One hot vector to use as output
    encoder = OneHotEncoder(sparse_output=False)  # `sparse=False` returns a dense array
    OUTPUT = encoder.fit_transform(adata.obs[['final_cell_label']])

    #Output and Input size of NN
    INPUT_size = INPUT[0].size
    OUTPUT_size = OUTPUT[0].size

    #---Build the model---
    #Initilize the model
    model = Sequential()
    #Define Hyperparameters
    OPTIMIZER = 'adam'                  #optimizer
    LOSS = 'categorical_crossentropy'   # loss function

    # First hidden layer with L2 regularization
    model.add(Dense(N_HIDDEN_1, input_shape=(INPUT_size,), activation=ACTIVATION_FUNCTION, kernel_regularizer=l2(L2_LAMBDA)))

    # Second hidden layer with L2 regularization
    model.add(Dense(N_HIDDEN_2, activation='relu', kernel_regularizer=l2(L2_LAMBDA)))

    # Output layer (softmax for multi-class classification)
    model.add(Dense(OUTPUT_size, activation='softmax'))

    # Summary of the whole model
    model.summary()

    # model compilation
    model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
    return model

@tf.function 
def joint_train(gene_expression, cell_labels, autoencoder, classifier, optimizer, lamda):
    """
    Joint training of autoencoder and classifier.
    
    Args:
        gene_expression: Input gene_expression data
        cell_labels: One-Hot encoded cell labels
        autoencoder: Pretrained autoencoder. Weights get updated and returned
        classifier: compiled classifier model: Weights get updated and returned
        optimizer: Optimizer for the autoencoder
        lambda: weight for reconstruction loss int: 0-1
    """
    with tf.GradientTape() as tape:
        #Reconstruction loss - minimize data alteration
        reconstructed_gene_expression = autoencoder(gene_expression)
        reconstruction_loss = tf.keras.losses.MeanSquaredError()(gene_expression, reconstructed_gene_expression)

        #classifier loss - minimize wrong class determination
        class_guess = classifier(gene_expression)
        classifier_loss = tf.keras.losses.CategoricalCrossentropy()(class_guess, cell_labels)

        #Total loss: weighted by lamda_1 and lambda_2
        lambda_1 = lamda
        lambda_2 = 1-lambda_1
        total_loss = lambda_1 * reconstruction_loss + lambda_2* classifier_loss

        # Get gradients and update both autoencoder and classifier weights
        gradients = tape.gradient(total_loss, autoencoder.trainable_variables + classifier.trainable_variables)
        optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables + classifier.trainable_variables))

    return total_loss, reconstruction_loss, classifier_loss

def autoencoder_classifier_jointtraining(adata, epochs, BATCHES, autoencoder, classifier, learning_rate, lamda):
    """
    Joint Training of autoencoder and classifier
    
    Args:
        adata: Input gene_expression data
        epochs: times the optimization is run
        BATCHES: size of batches
        autoencoder: Pretrained autoencoder model
        discriminator: Compiled classifier model
        learning_rate: int
        lamda: int: 0-1 Gewichtung des autoencoders

    returns:
    history: log file
    autoencoder: updated autoencoder model
    classifier: updated classifier mode
    """
    # Initialize an empty history arrays
    autoencoder_loss = []
    classifier_loss = []
    total_loss = []

    #Data preperation: adata to Tensorflow dataset
    #ADATA->NUMPY
    GENE_EXPRESSION = adata.X.toarray()
    #One-hot encoding the Cell Types
    encoder = OneHotEncoder(sparse_output=False)  # `sparse=False` returns a dense array
    CELL_TYPES = encoder.fit_transform(adata.obs[['final_cell_label']])
    #Combine NUMPY and One-hot encoded Batches in a Tensorflow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((GENE_EXPRESSION, CELL_TYPES))
    #Create training batches
    batch_size = BATCHES
    train_dataset = train_dataset.batch(batch_size)
    
    # Optimizer for the joint training
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        tot_loss_avg = tf.keras.metrics.Mean()      #loss of the total_loss
        rec_loss_avg = tf.keras.metrics.Mean()      #loss of the autoencoder
        cls_loss_avg = tf.keras.metrics.Mean()      #loss of the classifier
        
        for batch in train_dataset:
            #Batch number
            gene_expression, cell_labels = batch
            
            # Train autoencoder
            tot_loss, rec_loss, cls_loss = joint_train(
                gene_expression, cell_labels, autoencoder, classifier, optimizer, lamda
            )
            # Update adv loss
            tot_loss_avg.update_state(tot_loss)
            rec_loss_avg.update_state(rec_loss)
            cls_loss_avg.update_state(cls_loss)

        
        # print epoch results
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Total Loss: {tot_loss_avg.result():.4f}")
            
        #save history
        autoencoder_loss.append(rec_loss_avg.result())
        classifier_loss.append(cls_loss_avg.result())
        total_loss.append(tot_loss_avg.result())

    #Create history DataFrame    
    history = pd.DataFrame({
        'autoencoder_loss': autoencoder_loss,
        'classifier_loss': classifier_loss,
        'total_loss': total_loss
        })
    return history, autoencoder, classifier

 #--Visualisize Training---
def plot_f_training(history):

    # Plot training & validation loss
    figure = plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(history["autoencoder_loss"])
    #plt.plot(history["reconstruction_loss"])
    plt.title('autoencoder loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.subplot(1, 3, 2)
    plt.plot(history["classifier_loss"])
    plt.title('classifier loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.subplot(1, 3, 3)
    plt.plot(history["total_loss"])
    plt.title('total loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.show()
    return figure
