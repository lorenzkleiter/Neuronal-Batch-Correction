#---Hyperparameter tuning--
#  final training parameters are optimised via grid search

#imports
from utils import loading
from models.classifier import autoencoder_classifier_jointtraining
import pandas as pd
from utils.scores import scoring_autoencoder
import warnings
import tensorflow as tf
import time

#Ignore warnings
warnings.filterwarnings("ignore")

#Import Data
test = loading.load_dataset('Lung_atlas_public')
label_key = 'cell_type'
batch_key = 'batch'

#Define search space of Hyperparameters
epochs = [30]
batches = [128, 256, 512, 1024]
learning_rate = [0.0001]
lambda_weight = [0.8]
loss_function = ["uniform"]
column_names = ['epochs', 'batch_size', 'learning_rate', 'lambda_weight', 'loss_function']
column_names_final = ['epochs', 'batch_size', 'learning_rate', 'lambda_weight', 'loss_function', 'ari',	'hvg',	'asw',	'f1',	'nmi',	'sil',	'graph',	'pcr',	'sil_batch',	'avg_bio',	'avg_batch', 'time']

#generate pandas dataframe of all Hyperparameter combinations
df = pd.DataFrame(columns=column_names)
i= 0
for epoch in epochs:
    for batch in batches:
        for rates in learning_rate:
            for lb in lambda_weight:
                for loss in loss_function:
                    temp = [epoch, batch, rates, lb, loss]
                    new_row = pd.DataFrame([temp], columns=column_names)
                    df = pd.concat([df, new_row], ignore_index=True)

#create DataFrames for final storage
df_results = pd.DataFrame(columns=column_names_final)
df_history = []

#Grid search loop
for i in range(df.shape[0]):
    # Calculate the start time
    start = time.time()

    #Import discriminator
    discriminator = loading.load_model('discriminator')

    #Import autoencoder
    autoencoder = loading.load_model('autoencoder_mseloss')

    #Import Classifier
    classifier = loading.load_model('classifier')

    print("Run", i+1)
    print("Hyperparameters: ")
    print(df.loc[i]["epochs"], df.loc[i]["batch_size"], df.loc[i]["learning_rate"], df.loc[i]["lambda_weight"], df.loc[i]["loss_function"])

    #Joint Training of all 3 components not updating discriminator weights
    history, autoencoder, classifier = autoencoder_classifier_jointtraining(
                                                                        test,                                #anndata object: Dataset
                                                                        df.loc[i]["epochs"],                 #epochs
                                                                        df.loc[i]["batch_size"],             #batch size
                                                                        autoencoder,                         #autoencoder: gets updated
                                                                        classifier,                          #classifier: biological loss
                                                                        discriminator,                       #discriminator: adverserial loss
                                                                        df.loc[i]["learning_rate"],          #learning rate autoencoder
                                                                        0.001,                               #learning rate classifier
                                                                        df.loc[i]["lambda_weight"],          #weighting of reconstructio loss vs. classifier/adverseriel loss
                                                                        True,                                #True to enable simulatious adverserial training
                                                                        df.loc[i]["loss_function"],          #loss function: log or uniform
                                                                        True,                                #True to freeze classifier weight updating
                                                                        label_key,
                                                                        batch_key
                                                                        )
    tf.keras.backend.clear_session()
    print("Start Scoring")
    score = scoring_autoencoder(test,               #anndata object: Dataset
                                autoencoder,        #autoencoder: gets scored
                                label_key,
                                batch_key,
                                True                #True to enable Top 2000 varible gene selection
                    )
    print(score)

    # Calculate the end time and time taken
    end = time.time()
    length = end - start
    df_length = pd.DataFrame({
        'time': length,
        }, index=[0])
    
    # Save history
    df_history.append({"run": i, "history": history})

    # Get the hyperparameters for this run as a DataFrame (already a DataFrame with 1 row)
    hyperparams = df.loc[[i]]  # Double brackets to keep it as a DataFrame
    hyperparams = hyperparams.reset_index(drop=True)  # Reset index to start at 0
    print(hyperparams)

    # Combine hyperparameters with scores horizontally
    new_row = pd.concat((hyperparams, score), axis=1)
    # Combine hyperparameters with times horizontally
    new_row = pd.concat((new_row, df_length), axis=1)
    print(new_row)
    # Add to results DataFrame
    df_results = pd.concat((df_results, new_row), ignore_index=True, axis=0)
    print(df_results)

    # Save Scores into directory
    file_name = "Score_Results.csv"
    save_path = f"hyperparameters/{file_name}"
    df_results.to_csv(save_path, index=False)
    print(f"df_results saved to {save_path}")

    # Show the results : this can be altered however you like
    print("It took", length, "seconds!")
