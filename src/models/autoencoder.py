#---autoencoder---
#import modules
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from scipy import sparse
import anndata

#Creating a autoencoder:
#This Function returns an compiled autoencoder
"""
Missing right now! L2 reg. Leaky layer
"""
def create_autoencoder( 
                        adata,                           #anndata object: Only necessary to get size
                        N_HIDDEN,                       #int: Number of Nodes that the Encoder comprsses the data to
                        ACTIVATION_FUNCTION_ENCODER,    #str: activation function of the encoder
                        ACTIVATION_FUNCTION_DECODER     #str: activation function of the decoder
                        ):
    #Get Input Size 
    INPUT =  adata.X.toarray()
    INPUT_size = INPUT[0].size

    #Initilize the encoder
    encoder = Sequential()
    # add one dense hidden layer to the encoder, with the input size 3580 and the output size 256
    encoder.add(Dense(N_HIDDEN, input_shape=(INPUT_size,), activation=ACTIVATION_FUNCTION_ENCODER)) 

    #Initilize the decoder
    decoder = Sequential()
    # add one output layer to the decoder, with the input size N_hidden and the output size to the input size
    decoder.add(Dense(INPUT_size, activation=ACTIVATION_FUNCTION_ENCODER))

    # Define the autoencoder model
    input = tf.keras.Input(shape=(INPUT_size,)) #Generic input
    encoded = encoder(input) #encoder input
    decoded = decoder(encoded)  #decoder input

    autoencoder = tf.keras.Model(input, decoded) #autoencoder: takes input and returns decoded data

    # Compile the autoencoder
    autoencoder.compile(optimizer='adam', loss='mse') #adam: type of optimizer used to minimize the loss function. mse: loss function. difference between input and output

    print(autoencoder.summary())

    return autoencoder      #return an compiled autoencoder object

#Train a autoencoder
#This Function returns the history of trainig

def train_autoencoder( 
                        adata,                           #anndata object: Training set
                        autoencoder,                    #compiled autoencoder object from creat_autoencoder
                        EPOCH,                          #int: number of epochs trained
                        BATCH_SIZE                      #int: Size of Batch
                        ):
    #Get the INPUT
    INPUT =  adata.X.toarray()

    # The autoencoder fits the data to the data -> in the end nothing should change
    history = autoencoder.fit(INPUT, INPUT,
                epochs=EPOCH,
                batch_size=BATCH_SIZE,
                shuffle=True, 
                validation_data=(INPUT, INPUT),
                verbose=2) # shuffle the data after each epoch to reduce overfitting
        
    #--Visualisize Training---
    # Plot training & validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Autoencoder Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show

    return autoencoder 

#Function to autoencode data:
#This returns the updated data as Anndata object
def autoencode(
                adata,          #anndata object: Input data
                autoencoder     #Compiled Autoencoder
                ):
    #adata -> numpy
    INPUT = adata.X.toarray()

    #autoencode data
    autoencoded_data = autoencoder.predict(INPUT)

    #numpy -> adata
    autoencoded_data = sparse.csr_matrix(autoencoded_data)
    #create anndata object
    OUTPUT = anndata.AnnData(autoencoded_data)
    #Take metadata from input
    OUTPUT.obs = adata.obs
    OUTPUT.var= adata.var
    return OUTPUT