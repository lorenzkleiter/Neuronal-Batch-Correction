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
from sklearn.model_selection import train_test_split

#loss function
from models.autoencoder import loss_function_log, loss_function_uniform


def create_classifier(  adata,                           #anndata object: Only necessary to get size
                        N_HIDDEN_1,                      #int: Number of Nodes of the first layer
                        N_HIDDEN_2,                      #int: Number of Nodes of the second layer
                        L2_LAMBDA,                       #int: regularisation amount
                        ACTIVATION_FUNCTION,
                        label_key                       #name of cell label collum
                        ):
    #---Prepare Data---
    INPUT =  adata.X.toarray()
    #Encode the cell labels as One hot vector to use as output
    encoder = OneHotEncoder(sparse_output=False)  # `sparse=False` returns a dense array
    OUTPUT = encoder.fit_transform(adata.obs[[label_key]])

    #Output and Input size of NN
    INPUT_size = INPUT[0].size
    OUTPUT_size = OUTPUT[0].size

    #---Build the model---
    #Initilize the model
    model = Sequential()

    # First hidden layer with L2 regularization
    model.add(Dense(N_HIDDEN_1, input_shape=(INPUT_size,), activation=ACTIVATION_FUNCTION, kernel_regularizer=l2(L2_LAMBDA)))

    # Second hidden layer with L2 regularization
    model.add(Dense(N_HIDDEN_2, activation=ACTIVATION_FUNCTION, kernel_regularizer=l2(L2_LAMBDA)))

    # Output layer (softmax for multi-class classification)
    model.add(Dense(OUTPUT_size, activation='softmax'))

    # Summary of the whole model
    model.summary()

    # model compilation
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_classifier(   adata,                      #autoencoded Dataset
                        model,                      #compiled classifier
                        EPOCH,                      #Epochs
                        BATCH_SIZE,                 #Batch size
                        shuffle,                    #shuffle
                        label_key,                  #name of cell collumn
                        random_seed=42
                    ):

    #---Prepare Data---
    INPUT =  adata.X.toarray()

    #One-hot encode the cell types
    encoder = OneHotEncoder(sparse_output=False)  # `sparse=False` returns a dense array
    OUTPUT = encoder.fit_transform(adata.obs[[label_key]])

    #Perform a Train Test Split
    INPUT_train, INPUT_test, OUTPUT_train, OUTPUT_test = train_test_split(INPUT, OUTPUT, test_size=0.1, random_state=random_seed)

    #Perform the training
    #Fit the input to the output
    history = model.fit(INPUT_train, OUTPUT_train, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=2, validation_split=0.2, shuffle=shuffle)
     #score the network
    score = model.evaluate(INPUT_test, OUTPUT_test, verbose=0)
    print("\nTest score/loss:", score[0])
    print("Test accuracy:", score[1])
    
    return history, model
 
def joint_train(
        gene_expression, cell_labels, batch_labels, autoencoder, classifier, discriminator,
        optimizer_autoencoder, optimizer_classifier, lamda, adverserial, loss_function, freeze_classifier
        ):
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
    with tf.GradientTape(persistent=True) as tape:
        #Reconstruction loss - minimize data alteration
        reconstructed_gene_expression = autoencoder(gene_expression)
        reconstruction_loss = tf.keras.losses.MeanSquaredError()(gene_expression, reconstructed_gene_expression)

        #classifier loss - minimize wrong class determination
        class_guess = classifier(reconstructed_gene_expression)
        classifier_loss = tf.keras.losses.CategoricalCrossentropy()(class_guess, cell_labels)
        classifier_loss = tf.cast(classifier_loss, tf.float32)
        
        #adversial loss - minimize discriminator accuracy
        disc_output = discriminator(reconstructed_gene_expression)
        # Adversarial loss - the lower the loss the worse the discriminator is at picking the correct batch
        if loss_function == 'log': adverserial_loss = loss_function_log(batch_labels, disc_output)
        elif loss_function == 'uniform': adverserial_loss = loss_function_uniform(batch_labels, disc_output)

        #Make sure lamda is float32
        lamda = tf.cast(lamda, tf.float32)   

        if adverserial == True:
            #Total loss: weighted by lamda_1 and lambda_2 and lambda_3
            lambda_1 = lamda
            lambda_2 = 3*(1-lamda)/4
            lambda_3 = (1-lamda)/4
            total_loss = lambda_1 * reconstruction_loss + lambda_2* classifier_loss + lambda_3*adverserial_loss
        else:
            #Total loss: weighted by lamda_1 and lambda_2 
            lambda_1 = lamda
            lambda_2 = 1-lamda
            total_loss = lambda_1*reconstruction_loss + lambda_2*classifier_loss

    # Get gradients and update autoencoder weights
    gradients1 = tape.gradient(total_loss, autoencoder.trainable_variables)
    optimizer_autoencoder.apply_gradients(zip(gradients1, autoencoder.trainable_variables))

    if freeze_classifier == False:
        # Get gradients and update classifier weights
        gradients2 = tape.gradient(total_loss, classifier.trainable_variables)
        optimizer_classifier.apply_gradients(zip(gradients2, classifier.trainable_variables))
    del tape
    return total_loss, reconstruction_loss, classifier_loss, adverserial_loss

def autoencoder_classifier_jointtraining(
        adata, epochs, BATCHES, autoencoder, classifier, discriminator, learning_rate_autoencoder, 
        learning_rate_classifier, lamda, adverserial, loss_function, freeze_classifier, label_key, batch_key
    ):
    """
    Joint Training of autoencoder and classifier
    
    Args:
        adata: Input gene_expression data
        epochs: times the optimization is run
        BATCHES: size of batches
        autoencoder: Pretrained autoencoder model
        classifier: Compiled cell type classifier model
        discriminiator: batch discriminator. For sanity check. Is not trained 
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
    class_accuracies = []
    dis_accuracies = []

    # Freeze the discriminator weights
    discriminator.trainable = False
    if freeze_classifier == True:
        classifier.trainable = False

    #Data preperation: adata to Tensorflow dataset
    #ADATA->NUMPY
    GENE_EXPRESSION = adata.X.toarray()
    #One-hot encoding of keys
    # Create separate encoders
    cell_type_encoder = OneHotEncoder(sparse_output=False)
    batch_encoder = OneHotEncoder(sparse_output=False)

    # Encode each with its own encoder
    CELL_TYPES = cell_type_encoder.fit_transform(adata.obs[[label_key]])
    BATCH_LABELS = batch_encoder.fit_transform(adata.obs[[batch_key]])

    #Combine NUMPY and One-hot encoded Batches in a Tensorflow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((GENE_EXPRESSION, CELL_TYPES, BATCH_LABELS))
    #Create training batches
    batch_size = BATCHES
    train_dataset = train_dataset.batch(batch_size)
    
    # Optimizer for the joint training
    optimizer_autoencoder = tf.keras.optimizers.Adam(learning_rate=learning_rate_autoencoder)
    optimizer_classifier = tf.keras.optimizers.Adam(learning_rate=learning_rate_classifier)

    # Training loop
    for epoch in range(epochs):
        tot_loss_avg = tf.keras.metrics.Mean()      #loss of the total_loss
        rec_loss_avg = tf.keras.metrics.Mean()      #loss of the autoencoder
        cls_loss_avg = tf.keras.metrics.Mean()      #loss of the classifier
        adv_loss_avg = tf.keras.metrics.Mean()      #adverserial loss

        for batch in train_dataset:
            #Batch number
            gene_expression, cell_labels, batch_labels = batch
            
            # Train autoencoder
            tot_loss, rec_loss, cls_loss, adv_loss = joint_train(
                gene_expression, cell_labels, batch_labels, autoencoder, classifier, discriminator,
                optimizer_autoencoder, optimizer_classifier, lamda, adverserial, loss_function, freeze_classifier
            )
            # Update adv loss
            tot_loss_avg.update_state(tot_loss)
            rec_loss_avg.update_state(rec_loss)
            cls_loss_avg.update_state(cls_loss)
            adv_loss_avg.update_state(adv_loss)

        
        # print epoch results
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Total Loss: {tot_loss_avg.result():.4f}")

        #Deleted for now: computational heavy history with accuracies
        #save history
        autoencoder_loss.append(rec_loss_avg.result())
        classifier_loss.append(cls_loss_avg.result())
        print(f"Classifier accuracy: ")
        class_accuracy = classifier.evaluate(autoencoder(GENE_EXPRESSION), CELL_TYPES, verbose=2)[1]
        class_accuracies.append(class_accuracy)
        print(f"Discriminator accuracy: ")
        dis_accuracy = discriminator.evaluate(autoencoder(GENE_EXPRESSION), BATCH_LABELS, verbose=2)[1]
        dis_accuracies.append(dis_accuracy)

    #Create history DataFrame    
    history = pd.DataFrame({
        'autoencoder_loss': autoencoder_loss,
        'classifier_loss': classifier_loss,
        'classifier_accuracy': class_accuracies,
        'discriminator_accuracy': dis_accuracies
        })
    """
        #save history
        autoencoder_loss.append(rec_loss_avg.result())
        classifier_loss.append(cls_loss_avg.result())

    #Create history DataFrame    
    history = pd.DataFrame({
        'autoencoder_loss': autoencoder_loss,
        'classifier_loss': classifier_loss
        })
    """
    return history, autoencoder, classifier

