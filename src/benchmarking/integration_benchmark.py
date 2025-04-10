import time
from utils import loading
from utils.scores import scoring_integration
from benchmarking.integrators import integrate
import pandas as pd
import warnings

#Ignore warnings
warnings.filterwarnings("ignore")

#Import Data
test = loading.load_dataset('Lung_atlas_public')
label_key = 'cell_type'
batch_key = 'batch'

#Define Integrators
integrators = ['scanorama', 'nbc']
#initilize data frame to score results
column_names = ['ari',	'hvg',	'asw',	'f1',	'nmi',	'sil',	'graph',	'pcr',	'sil_batch',	'avg_bio',	'avg_batch', 'time']
df_results = pd.DataFrame(columns=column_names)

#Iterate over all integrators
for alg in integrators:
    print("Start Integration with", alg)
    #meassure time
    start = time.time()
    #integrate data
    corrected = integrate(test, batch_key, label_key, alg)
    #meassure time
    end = time.time()
    length = end - start
    #and put into data frame
    df_length = pd.DataFrame({
        'time': length,
        }, index=[0])
    
    print("Integration done. It took", length, "seconds!")

    #score integrated data
    test = loading.load_dataset('Lung_atlas_public') #load data new to make sure adata is correct
    score = scoring_integration(test,               #anndata object: Dataset
                    corrected,                      #Corrected Dataset: gets scored
                    label_key,
                    batch_key,
                    True                           #True to enable Top 2000 varible gene selection
                        )
    print(score)
    # combine score with time horizontally
    score = pd.concat((score, df_length), axis=1)
    # Add to results DataFrame
    df_results = pd.concat((df_results, score), ignore_index=True, axis=0)
    print(df_results)

    # Save Scores into directory
    file_name = "Score_Results.csv"
    save_path = f"benchmarking/{file_name}"
    df_results.to_csv(save_path, index=False)
    print(f"df_results saved to {save_path}")
