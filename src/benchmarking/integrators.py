import scib
import nbc
import scanpy as sc
import nbc
from scgen import SCGEN

#Function including all integrators to test
def integrate(adata, batch_key, label_key, alg):
    if alg == 'None':
        return adata
    elif alg == 'combat':
        sc.pp.combat(adata, batch_key)
        return adata
    elif alg == 'harmony':
        ad = scib.integration.harmony(adata, batch_key, hvg=None)
        return ad
    elif alg == 'scanorama':
        ad = scib.integration.scanorama(adata, batch_key, hvg=None)
        return ad
    elif alg == 'scgen':
        SCGEN.setup_anndata(adata, batch_key=batch_key, labels_key=label_key) 
        model = SCGEN(adata)
        model.train(max_epochs=40, batch_size=128, early_stopping=False)
        ad = model.batch_removal(adata)
        return ad
    elif alg == 'nbc':
        ad = nbc.integration(adata, batch_key, batch_key, 40, 128)
        return ad
