#---Discriminator---
#imports
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

#Function to create a discriminator:
#This returns the initilized discriminator

def create_discriminator( 
                        adata,                           #anndata object: Only necessary to get size
                        N_HIDDEN_1,                      #int: Number of Nodes of the first layer
                        N_HIDDEN_2,                      #int: Number of Nodes of the second layer
                        L2_LAMBDA,                       #int: regularisation amount
                        learning_rate,                   #learning rate of adam optimizer                       
                        ACTIVATION_FUNCTION,             #activation function used for first 2 layers
                        batch_key,
                        random_seed=42
                        ):
    #---Prepare Data---
    INPUT =  adata.X.toarray()

    #Encode the cell labels as One hot vector to use as additional information
    #encoder = OneHotEncoder(sparse_output=False)  # `sparse=False` returns a dense array
    #encoded_labels = encoder.fit_transform(adata.obs[['final_cell_label']])

    # Concatenate gen expreesion matrix with oneHotLabels
    #INPUT = np.concatenate((INPUT, encoded_labels), axis=1)
    """
    remove encoded_label for testing
    """

    #One-hot encode the Batches
    encoder = OneHotEncoder(sparse_output=False)  # `sparse=False` returns a dense array
    OUTPUT = encoder.fit_transform(adata.obs[[batch_key]])

    #Output and Input size of NN
    INPUT_size = INPUT[0].size
    OUTPUT_size = OUTPUT[0].size


    #---Initilize Model---
    #create optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    #create model
    model = Sequential()

    # First hidden layer with L2 regularization
    model.add(Dense(N_HIDDEN_1, input_shape=(INPUT_size,), activation=ACTIVATION_FUNCTION, kernel_regularizer=l2(L2_LAMBDA)))

    # Second hidden layer with L2 regularization
    model.add(Dense(N_HIDDEN_2, activation=ACTIVATION_FUNCTION, kernel_regularizer=l2(L2_LAMBDA)))

    # Output layer (softmax for multi-class classification)
    model.add(Dense(OUTPUT_size, activation='softmax'))

    # model compilation
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])   

    return model      #return an compiled discriminator object

#Function to train a discriminator:
#This returns the updated weights and history discriminator
def train_discriminator(
                        adata,                          #anndata object: Dataset
                        model,                          #compiled discriminator
                        EPOCH,                          #int: number of epochs trained
                        BATCH_SIZE,                     #int: Size of Batch
                        shuffle,                        #conditional
                        batch_key,
                        random_seed=42
                    ):
    #---Prepare Data---
    INPUT =  adata.X.toarray()
    """
    removed for testing: Impliment later?
    """
    #Encode the cell labels as One hot vector to use as additional information
    #encoder = OneHotEncoder(sparse_output=False)  # `sparse=False` returns a dense array
    #encoded_labels = encoder.fit_transform(adata.obs[['final_cell_label']])
    # Concatenate gen expreesion matrix with oneHotLabels
    #INPUT = np.concatenate((INPUT, encoded_labels), axis=1)
   
    #One-hot encode the Batches
    encoder = OneHotEncoder(sparse_output=False)  # `sparse=False` returns a dense array
    OUTPUT = encoder.fit_transform(adata.obs[[batch_key]])

    #Perform a Train Test Split
    INPUT_train, INPUT_test, OUTPUT_train, OUTPUT_test = train_test_split(INPUT, OUTPUT, test_size=0.1, random_state=random_seed)

    #score the network
    score = model.evaluate(INPUT_test, OUTPUT_test, verbose=2)
    print("\nTest score/loss:", score[0])
    print("Test accuracy:", score[1])

    #Perform the training
    #Fit the input to the output
    history = model.fit(INPUT_train, OUTPUT_train, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=2, validation_split=0.2, shuffle=shuffle)
   
    return history, model

def test_model(model, adata, key):
    #Inpt
    INPUT =  adata.X.toarray()
    #Output
    encoder = OneHotEncoder(sparse_output=False)  # `sparse=False` returns a dense array
    OUTPUT = encoder.fit_transform(adata.obs[[key]])
    score = model.evaluate(INPUT, OUTPUT, verbose=0)
    print("Test score/loss:", score[0])
    print("Test accuracy:", score[1])

    return score[1]