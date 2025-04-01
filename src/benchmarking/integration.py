import time
import nbc
from utils import loading
from utils.scores import scoring_integration

#Import Data
test = loading.load_dataset('Lung_atlas_public')
label_key = 'cell_type'
batch_key = 'batch'
epochs = 30

print("Start Integration")
start = time.time()
corrected = nbc.integration(test, label_key, batch_key, epochs)
end = time.time()
length = end - start
print("Integration done. It took", length, "seconds!")

score = scoring_integration(test,               #anndata object: Dataset
                corrected,                      #Corrected Dataset: gets scored
                label_key,
                batch_key,
                True                           #True to enable Top 2000 varible gene selection
                    )
print(score)