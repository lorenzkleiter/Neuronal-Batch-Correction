#---autoencoder---
#import modules
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from scipy import sparse
import anndata
from sklearn.preprocessing import OneHotEncoder

#Function 1:
#---Creating an autoencoder---
#Input: Data, N_Hidden, activation functions 
#Output: This Function returns an compiled autoencoder
"""
Missing right now! L2 reg. Leaky layer
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

#Function 3:
# ---Adverserial Training: Define Trainng function--- 
#Used by Function 4
#Define the autoencoder adversarial training function
#overriding the normal training function
@tf.function 
def train_autoencoder_adversarial(gene_expression, batch_labels, autoencoder, discriminator, optimizer):
    """
    Train the autoencoder to fool the discriminator into classifying reconstructions as target classes.
    
    Args:
        gene_expression: Input gene_expression data
        batch_labels: One-hot encoded target batch labels to fool the discriminator
        encoder: Pretrained encoder model
        decoder: Pretrained decoder model
        discriminator: Pretrained discriminator model (frozen during this training)
        optimizer: Optimizer for the autoencoder
    """
    with tf.GradientTape() as tape:
        # CAll autoencoder to get reconstructed gene expression
        reconstructed_gene_expression = autoencoder(gene_expression)
        
        # Get discriminator output for reconstructed gene expression
        disc_output = discriminator(reconstructed_gene_expression)
        
        # Adversarial loss - make discriminator classify reconstructions as target labels
        adversarial_loss = tf.keras.losses.CategoricalCrossentropy()(batch_labels, disc_output)
    
    # Get gradients and update autoencoder weights only
    gradients = tape.gradient(adversarial_loss, autoencoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))
    
    return adversarial_loss

#Function 4: Adverserial Training loop
#---Adverserial Training: Create custom training loop---
#Input: Anndata Object, Epochs, Batch_size, Autoencoder, Discriminaotr
#Output: Autoencoder
"""
Implement later: reconstruction loss print out
"""
def adversarial_training(adata, epochs, BATCHES, autoencoder, discriminator):
    #Data is prepared: adata to Tensorflow dataset
    #ADATA->NUMPY
    GENE_EXPRESSION = adata.X.toarray()

    #One-hot encoded Batches
    encoder = OneHotEncoder(sparse_output=False)  # `sparse=False` returns a dense array
    BATCH_LABELS = encoder.fit_transform(adata.obs[['batchname_all']])
    #Combine in a Tensorflow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((GENE_EXPRESSION, BATCH_LABELS))
    #Create batches
    batch_size = BATCHES
    train_dataset = train_dataset.batch(batch_size)

    # Freeze the discriminator weights
    discriminator.trainable = False
    
    # Optimizer for the autoencoder
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    # Training loop
    for epoch in range(epochs):
        adv_loss_avg = tf.keras.metrics.Mean()

        for batch in train_dataset:
            gene_expression, batch_labels = batch
            
            # Target same class as original
            target_labels = batch_labels
            
            # Train autoencoder
            adv_loss = train_autoencoder_adversarial(
                gene_expression, target_labels, autoencoder, discriminator, optimizer
            )
            
            # Update adv loss
            adv_loss_avg.update_state(adv_loss)

        
        # Print epoch results
        print(f"Epoch {epoch+1}/{epochs}")
        print(
              f"Adversarial Loss: {adv_loss_avg.result():.4f}, " 
            )
    return autoencoder

#Function 1: autoencode data:
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

