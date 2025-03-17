from pretraining import pretrainer
from adversialtraining import adtrainert
from celltypetraining import celltrainer


def NBC(
        adata,                  	#anndata: your scRNA-dataset in anndata format. Batch and cell labels necessary
        collumn_name_batches,       #str: the name of your batch label collumn
        collum_name_celltypes       #str: the name of your cell label collumn
        ):

    pretrainer(adata, collumn_name_batches, collum_name_celltypes)
    adtrainer(adata, collumn_name_batches, collum_name_celltypes)
    corrected_data = celltrainer(adata, collumn_name_batches, collum_name_celltypes)

    return corrected_data