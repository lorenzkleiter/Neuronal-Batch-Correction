#master function
from training.pretraining import actrainer
from training.discriminatortraining import dctrainer
from training.classifiertraining import cltrainer
from training.finaltraining import jointtrainer
from models.autoencoder import autoencode
from utils import loading


def integration(
        adata,                  	#anndata: your scRNA-dataset in anndata format. Batch and cell labels necessary
        collumn_name_batches,       #str: the name of your batch label collumn
        collum_name_celltypes,      #str: the name of your cell label collumn
        epochs                      #int: number of epochs for joint training
        ):

    actrainer(adata, collumn_name_batches, collum_name_celltypes)
    dctrainer(adata, collumn_name_batches, collum_name_celltypes)
    cltrainer(adata, collumn_name_batches, collum_name_celltypes)
    jointtrainer(adata, collumn_name_batches, collum_name_celltypes, epochs)
    autoencoder = loading.load_model("autoencoder_final_onestep")
    print("--correct batch effect with trained autoencoder--")
    corrected_data = autoencode(adata, autoencoder)

    return corrected_data



