#---autoencoder---
#import modules
from math import log, log1p
from matplotlib import figure
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from scipy import sparse
import anndata
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

#Function 1:
#---Creating an autoencoder---
#Input: Data, N_Hidden, activation functions 
#Output: This Function returns an compiled autoencoder
"""
Missing right now! L2 reg. Leaky layer, custom. learning rate
"""
def create_autoencoder( 
                        adata,                          #anndata object: Only necessary to get size
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

#Function 2:
#---Training an autoencoder---
#This Function trains the autoencoder on reconstruction loss
#Input: Data, Autoencoder, Epochs, BATCH_Size 
#Output: This Function returns an compiled autoencoder
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
    return history, autoencoder 

 #--Visualisize Training---
def plot_ac_training(history):
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

#Function 3: autoencode data:
#Input: Anndata Object
#Output: Anndata Object
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


#Function 4 and 5:
#--- Custom loss Functions---
#Used by Function 6
#Input: disc_output - discriminator output (guess), batch_labels: one hot encoded batch labels (truth)
#Output: adversarial_loss: the lower the loss the worse the discriminator is at picking the correct batch

def loss_function_log(disc_output, batch_labels):
    #get Batch_size
    batch_size = batch_labels.shape[0]
    #Custom loss function: First calculate log(1-Confidence in correct batch)
    batch_labels = tf.cast(batch_labels, tf.float32)                # make sure floats match
    disc_output  = tf.cast(disc_output, tf.float32)                # make sure floats match

    # Element-wise multiplication followed by summation
    pred_confidence = (tf.reduce_sum(batch_labels * disc_output))/batch_size
    #Adversarial loss - make discriminator classify reconstructions as target labels
    adversarial_loss = -tf.math.log(1-pred_confidence)
    return adversarial_loss

def loss_function_uniform(disc_output, batch_labels):
    # The target is to have a uniform distribution around all classes
    num_classes = batch_labels.shape[1]
    target_labels = tf.ones_like(batch_labels) / num_classes
    adv_loss = tf.keras.losses.CategoricalCrossentropy()(target_labels, batch_labels)
    return adv_loss

#Function 6:
# ---Adverserial Training: Define Trainng function--- 
#Used by Function7
#Define the autoencoder adversarial training function
#overriding the normal training function

@tf.function 
def train_autoencoder_adversarial(gene_expression, batch_labels, autoencoder, discriminator, optimizer, loss_function):
    """
    Train the autoencoder to fool the discriminator into classifying reconstructions as target classes.
    
    Args:
        gene_expression: Input gene_expression data
        batch_labels: One-hot encoded target batch labels to fool the discriminator
        autoencoder: Pretrained autoencoder. WEights get updated and returned
        discriminator: Pretrained discriminator model (frozen during this training)
        optimizer: Optimizer for the autoencoder
        loss_function: str: log or uniform. See Function 4/5
    """
    with tf.GradientTape() as tape:
        # CAll autoencoder to get reconstructed gene expression
        reconstructed_gene_expression = autoencoder(gene_expression)
        
        # Get discriminator output for reconstructed gene expression
        disc_output = discriminator(reconstructed_gene_expression)
        
        # Adversarial loss - the lower the loss the worse the discriminator is at picking the correct batch
        if loss_function == 'log': adverserial_loss = loss_function_log(batch_labels, disc_output)
        elif loss_function == 'uniform': adverserial_loss = loss_function_uniform(batch_labels, disc_output)

        # Reconstruction loss - check much the data is transformed
        reconstruction_loss = tf.keras.losses.MeanSquaredError()(gene_expression, reconstructed_gene_expression)
    
    # Get gradients and update autoencoder weights only
    gradients = tape.gradient(adverserial_loss, autoencoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))
    
    return adverserial_loss, reconstruction_loss

#Function 5: Adverserial Training loop
#---Adverserial Training: Create custom training loop---
#Input: Anndata Object, Epochs, Batch_size, Autoencoder, Discriminaotr
#Output: Autoencoder

def adversarial_training(adata, epochs, BATCHES, autoencoder, discriminator, loss_function, learning_rate):
    """
    Adversarial Training
    
    Args:
        adata: Input gene_expression data
        epochs: times the optimization is run
        BATCHES: size of batches
        autoencoder: Pretrained autoencoder model
        discriminator: Pretrained discriminator model (frozen during this training)
        loss_function: str: log or uniform. See Function 4/5
        learning_rate: int: suggested 0.000001
    Return:
    history: log file
    discriminator: updated autoencoder model
    """
    # Initialize an empty history arrays
    adversarial_losses = []
    reconstruction_losses = []
    accuracies = []

    #Data preperation: adata to Tensorflow dataset
    #ADATA->NUMPY
    GENE_EXPRESSION = adata.X.toarray()
    #One-hot encoding the Batches
    encoder = OneHotEncoder(sparse_output=False)  # `sparse=False` returns a dense array
    BATCH_LABELS = encoder.fit_transform(adata.obs[['batchname_all']])
    #Combine NUMPY and One-hot encoded Batches in a Tensorflow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((GENE_EXPRESSION, BATCH_LABELS))
    #Create training batches
    batch_size = BATCHES
    train_dataset = train_dataset.batch(batch_size)
    
    # Freeze the discriminator weights
    discriminator.trainable = False
    
    # Optimizer for the autoencoder
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Training loop
    for epoch in range(epochs):

        dis_loss_avg = tf.keras.metrics.Mean()      #loss of the discriminator
        rec_loss_avg = tf.keras.metrics.Mean()      #loss of the autoencoder

        for batch in train_dataset:
            #Batch number
            gene_expression, batch_labels = batch
            
            # Train autoencoder
            adv_loss, rec_loss = train_autoencoder_adversarial(
                gene_expression, batch_labels, autoencoder, discriminator, optimizer, loss_function
            )
            # Update adv loss
            dis_loss_avg.update_state(adv_loss)
            rec_loss_avg.update_state(rec_loss)

        
        # print epoch results
        print(f"Epoch {epoch+1}/{epochs}")
        print(f" Adversarial Loss: {dis_loss_avg.result():.4f}, " +
              f" Reconstruction Loss: {rec_loss_avg.result():.4f}")
            
        #save history
        adversarial_losses.append(dis_loss_avg.result())
        reconstruction_losses.append(rec_loss_avg.result())
        accuracy = discriminator.evaluate(autoencoder(GENE_EXPRESSION), BATCH_LABELS, verbose=2)[1]
        accuracies.append(accuracy)

    #Create history DataFrame    
    history = pd.DataFrame({
        'adversarial_loss': adversarial_losses,
        'reconstruction_loss': reconstruction_losses,
        'accuracy': accuracies
        })
    return history, autoencoder

 #--Visualisize Training---
def plot_ad_training(history):

    # Plot training & validation loss
    figure = plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(history["accuracy"])
    #plt.plot(history["reconstruction_loss"])
    plt.title('accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('Epoch')

    plt.subplot(1, 3, 2)
    plt.plot(history["adversarial_loss"])
    plt.title('adversarial_loss')
    plt.ylabel('Loss')

    plt.subplot(1, 3, 3)
    plt.plot(history["reconstruction_loss"])
    plt.title('reconstruction_loss')
    plt.ylabel('Loss')
    plt.show()
    return figure
