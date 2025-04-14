# Neuronal-Batch-Correction
NBC is an autoencoder-based neuronal network that is able to integrate scRNA-seq or scATAC-seq data from multiple sources, making higher-level analysis possible. The heart of NBC is an autoencoder (a neuronal network, which compresses and decompresses data), which is trained to fool a batch discriminator, thereby removing batch effects, while an output cell type classifier simultaneously trains the autoencoder to preserve biological information
![Encoubntered problems with loading](model.png)
