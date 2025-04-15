#master function
from NBC.training.autoencoder_pretraining import actrainer
from NBC.training.discriminatortraining import dctrainer
from NBC.training.classifiertraining import cltrainer
from NBC.training.autoencoder_finaltraining import jointtrainer
from NBC.models.autoencoder import autoencode


def integration(
        adata,                  	#anndata: your scRNA-dataset in anndata format. Batch and cell labels necessary
        collumn_name_batches,       #str: the name of your batch label collumn
        collum_name_celltypes,      #str: the name of your cell label collumn
        epochs,                      #int: number of epochs for joint training
        batch_size
        ):

    autoencoder = actrainer(adata)
    discriminator = dctrainer(adata, collumn_name_batches, autoencoder)
    classifier = cltrainer(adata, collum_name_celltypes, autoencoder)
    autoencoder = jointtrainer(adata, collumn_name_batches, collum_name_celltypes, epochs, batch_size, autoencoder, classifier, discriminator)
    print("--correct batch effect with trained autoencoder--")
    corrected_data = autoencode(adata, autoencoder)
    return corrected_data



