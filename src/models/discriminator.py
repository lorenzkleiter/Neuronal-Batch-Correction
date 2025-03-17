#---Discriminator---
#imports
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
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
                        ACTIVATION_FUNCTION,
                        random_seed=42,
                        ):
    #---Prepare Data---
    INPUT =  adata.X.toarray()

    #Encode the cell labels as One hot vector to use as additional information
    encoder = OneHotEncoder(sparse_output=False)  # `sparse=False` returns a dense array
    encoded_labels = encoder.fit_transform(adata.obs[['final_cell_label']])

    # Concatenate gen expreesion matrix with oneHotLabels
    #INPUT = np.concatenate((INPUT, encoded_labels), axis=1)
    """
    remove encoded_label for testing
    """

    #One-hot encode the Batches
    encoder = OneHotEncoder(sparse_output=False)  # `sparse=False` returns a dense array
    OUTPUT = encoder.fit_transform(adata.obs[['batchname_all']])

    #Perform a Train Test Split
    INPUT_train, INPUT_test, OUTPUT_train, OUTPUT_test = train_test_split(INPUT, OUTPUT, test_size=0.2, random_state=random_seed)

    #Output and Input size of NN
    INPUT_size = INPUT[0].size
    OUTPUT_size = OUTPUT[0].size
    print(INPUT_size)
    print(OUTPUT_size)

    #---Initilize Model---
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

    return model      #return an compiled discriminator object

#Function to train a discriminator:
#This returns the updated weights and history discriminator
def train_discriminator(
                        adata,                          #anndata object: Dataset
                        model,
                        EPOCH,                          #int: number of epochs trained
                        BATCH_SIZE,                      #int: Size of Batch
                        random_seed=42,
                    ):
    #---Prepare Data---
    INPUT =  adata.X.toarray()
    """
    removed for testing
    """
    #Encode the cell labels as One hot vector to use as additional information
    #encoder = OneHotEncoder(sparse_output=False)  # `sparse=False` returns a dense array
    #encoded_labels = encoder.fit_transform(adata.obs[['final_cell_label']])
    # Concatenate gen expreesion matrix with oneHotLabels
    #INPUT = np.concatenate((INPUT, encoded_labels), axis=1)
   
    #One-hot encode the Batches
    encoder = OneHotEncoder(sparse_output=False)  # `sparse=False` returns a dense array
    OUTPUT = encoder.fit_transform(adata.obs[['batchname_all']])

    #Perform a Train Test Split
    INPUT_train, INPUT_test, OUTPUT_train, OUTPUT_test = train_test_split(INPUT, OUTPUT, test_size=0.2, random_state=random_seed)

    #Perform a Train Test Split
    #Fit the input to the output
    history = model.fit(INPUT_train, OUTPUT_train, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=2, validation_split=0.2)

    #test the network
    score = model.evaluate(INPUT_test, OUTPUT_test, verbose=2)
    print("\nTest score/loss:", score[0])
    print("Test accuracy:", score[1])

    #--Visualisize Training---
    # Plot training & validation accuracy
    plt.figure(figsize=(12, 4))
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

    return model
