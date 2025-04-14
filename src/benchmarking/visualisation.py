from utils import loading
from utils.scores import scoring_integration, scoring_autoencoder
from models.autoencoder import autoencode
import pandas as pd
import scanpy as sc
import warnings
import neuronal_batch_correction.nbc as nbc
import matplotlib

#Ignore warnings
warnings.filterwarnings("ignore")

#Import Data
dataset = 'Immune_ALL_human'
test = loading.load_dataset(dataset)
label_key = 'final_annotation'
batch_key = 'batch'

keys = [['Lung_atlas_public', 'cell_type', 'batch'], 
        ['large_atac_gene_activity','final_cell_label','batchname_all'],
        ['small_atac_gene_activity','final_cell_label','batchname_all'],
        ['Immune_ALL_human', 'final_annotation', 'batch']]

#Integrate Data
integrated = nbc.integration(test, batch_key, label_key, 40, 128)

#Score Integration
result = scoring_integration(test,                           #anndata object: Dataset
                            integrated,                      #Corrected Dataset: gets scored
                            label_key,
                            batch_key,
                            True                              #True to enable Top 2000 varible gene selection
                        )


#Save score
Score = pd.DataFrame({
        'Dataset': dataset,
        'avg_bio': result.loc[0, 'avg_bio'],
        'avg_batch': result.loc[0, 'avg_batch']
        }, index=[0])

print(Score)

# Save Score into directory
file_name = f"Score_Results_{dataset}.csv"
save_path = f"benchmarking/{file_name}"
Score.to_csv(save_path, index=False)
print(f"df_results saved to {save_path}")
print("starting umap projection")

#Create Umap projection
matplotlib.rc('ytick', labelsize=14)
matplotlib.rc('xtick', labelsize=14)
sc.set_figure_params(dpi_save=300)

#Unintegrated
sc.pp.pca(test, svd_solver="arpack")
sc.pp.neighbors(test, n_neighbors=25)
sc.tl.umap(test, random_state=42)
sc.pl.umap(test, palette=matplotlib.rcParams["axes.prop_cycle"],
           color=[batch_key], show=False, use_raw=False, save=f'umap_{dataset}_unitegrated_batch.png')
sc.pl.umap(test, palette=matplotlib.rcParams["axes.prop_cycle"],
           color=[label_key], show=False, use_raw=False, save=f'umap_{dataset}_unitegrated_cells.png')


#Integrated
sc.pp.pca(integrated, svd_solver="arpack")
sc.pp.neighbors(integrated, n_neighbors=25)
sc.tl.umap(integrated, random_state=42)
sc.pl.umap(integrated, palette=matplotlib.rcParams["axes.prop_cycle"],
           color=[batch_key], show=False, use_raw=False, save=f'umap_{dataset}_itegrated_batch.png')
sc.pl.umap(integrated, palette=matplotlib.rcParams["axes.prop_cycle"],
           color=[label_key], show=False, use_raw=False, save=f'umap_{dataset}_itegrated_cells.png')

print("finished")