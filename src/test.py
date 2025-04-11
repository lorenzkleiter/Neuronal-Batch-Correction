from utils import loading
from utils.scores import scoring_autoencoder
import pandas as pd
import scanpy as sc
import warnings
import matplotlib
from models.autoencoder import autoencode
#Ignore warnings
warnings.filterwarnings("ignore")

from training.autoencoder_pretraining import actrainer
from training.discriminatortraining import dctrainer
from training.classifiertraining import cltrainer
from training.autoencoder_finaltraining import jointtrainer

#Import Data
dataset = 'Lung_atlas_public'
test = loading.load_dataset(dataset)
label_key = 'cell_type'
batch_key = 'batch'

#train autoencoder mse
#actrainer(test, batch_key, label_key)
dctrainer(test, batch_key, label_key)
#cltrainer(test, batch_key, label_key)
jointtrainer(test, batch_key, label_key, 10, 128)
autoencoder = loading.load_model("autoencoder_final_onestep")

#Score Integration
result = scoring_autoencoder(test, autoencoder, label_key, batch_key, True)
print(result)

#Look at plot
#Integrated
#integrated = autoencode(test, autoencoder)
#sc.pp.pca(integrated, svd_solver="arpack")
#sc.pp.neighbors(integrated, n_neighbors=25)
#sc.tl.umap(integrated, random_state=42)
#sc.pl.umap(integrated, palette=matplotlib.rcParams["axes.prop_cycle"],
#           color=[batch_key], show=True, use_raw=False)
#sc.pl.umap(integrated, palette=matplotlib.rcParams["axes.prop_cycle"],
#           color=[label_key], show=True, use_raw=False)