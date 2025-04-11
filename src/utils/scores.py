## Scoring Biological and Batch effects

#Imports
import scib
import os
import scanpy as sc
import tensorflow as tf
import matplotlib
import anndata
from scipy import sparse
import pandas as pd
from models.autoencoder import autoencode

def scoring_autoencoder(    adata,                  #unintegrated Iput
                            autoencoder,            #autoencoder
                            label_key,              #Batchkey
                            batch_key,              #Labelkey
                            n_top_genes            #True to enable Top 2000 varibel gene selection
            ):
    #Autoencode data
    integrated = autoencode(adata, autoencoder)
    #--Preprocessing--
    #Run PCA
    if n_top_genes == True:
        scib.pp.reduce_data(
            integrated, n_top_genes=2000, batch_key=batch_key, pca=True, neighbors=True
        )
    elif n_top_genes == False:
          scib.pp.reduce_data(
            integrated, batch_key=batch_key, pca=True, neighbors=True
        )    
          
    #Cluster PCA results
    scib.me.cluster_optimal_resolution(integrated, cluster_key="cluster", label_key=label_key)

    #--Bio Scores--

    #Adjusted Rand Index
    ari = scib.me.ari(integrated, cluster_key="cluster", label_key=label_key)

    #HVG overlap
    hvg = scib.me.hvg_overlap(adata, integrated, batch_key=batch_key)

    #isolated_labels_asw
    asw = scib.me.isolated_labels_asw(integrated, label_key=label_key, batch_key=batch_key, embed="X_pca")

    #isolated_labels_f1
    f1 = scib.me.isolated_labels_f1(integrated, label_key=label_key, batch_key=batch_key, embed="X_pca")

    #Normalized mutual information
    nmi = scib.me.nmi(integrated, cluster_key="cluster", label_key=label_key)

    #Average silhouette width (ASW)
    sil= scib.me.silhouette(integrated, label_key=label_key, embed="X_pca")

    #--Cluster Scores--

    #Graph Connectivity
    graph = scib.me.graph_connectivity(integrated, label_key=label_key)

    #Principal component regression score
    pcr = scib.me.pcr_comparison(adata, integrated, covariate=batch_key)

    # Modified average silhouette width (ASW) of batch
    sil_batch = scib.me.silhouette_batch(integrated, batch_key=batch_key, label_key=label_key, embed="X_pca")

    avg_bio = (ari+hvg+asw+f1+nmi+sil)/6
    avg_batch =(graph+pcr+sil_batch)/3

    Scores = pd.DataFrame({
        'ari': ari,
        'hvg': hvg,
        'asw': asw,
        'f1': f1,
        'nmi': nmi,
        'sil': sil,
        'graph': graph,
        'pcr': pcr,
        'sil_batch': sil_batch,
        'avg_bio': avg_bio,
        'avg_batch': avg_batch
        }, index=[0])
    return Scores

def scoring_integration(    adata,                  #unintegrated Iput
                            integrated,             #integrated Input
                            label_key,              #Batchkey
                            batch_key,              #Labelkey
                            n_top_genes             #True to enable Top 2000 varibel gene selection
            ):
    #Run PCA
    if n_top_genes == True:
        scib.pp.reduce_data(
            integrated, n_top_genes=2000, batch_key=batch_key, pca=True, neighbors=True
        )
    elif n_top_genes == False:
          scib.pp.reduce_data(
            integrated, batch_key=batch_key, pca=True, neighbors=True
        )    
          
    #Cluster PCA results
    scib.me.cluster_optimal_resolution(integrated, cluster_key="cluster", label_key=label_key)

    #--Bio Scores--

    #Adjusted Rand Index
    ari = scib.me.ari(integrated, cluster_key="cluster", label_key=label_key)

    #HVG overlap
    hvg = scib.me.hvg_overlap(adata, integrated, batch_key=batch_key)

    #isolated_labels_asw
    asw = scib.me.isolated_labels_asw(integrated, label_key=label_key, batch_key=batch_key, embed="X_pca")

    #isolated_labels_f1
    f1 = scib.me.isolated_labels_f1(integrated, label_key=label_key, batch_key=batch_key, embed="X_pca")

    #Normalized mutual information
    nmi = scib.me.nmi(integrated, cluster_key="cluster", label_key=label_key)

    #Average silhouette width (ASW)
    sil= scib.me.silhouette(integrated, label_key=label_key, embed="X_pca")

    #--Cluster Scores--

    #Graph Connectivity
    graph = scib.me.graph_connectivity(integrated, label_key=label_key)

    #Principal component regression score
    pcr = scib.me.pcr_comparison(adata, integrated, covariate=batch_key)

    # Modified average silhouette width (ASW) of batch
    sil_batch = scib.me.silhouette_batch(integrated, batch_key=batch_key, label_key=label_key, embed="X_pca")

    avg_bio = (ari+hvg+asw+f1+nmi+sil)/6
    avg_batch =(graph+pcr+sil_batch)/3

    Scores = pd.DataFrame({
        'ari': ari,
        'hvg': hvg,
        'asw': asw,
        'f1': f1,
        'nmi': nmi,
        'sil': sil,
        'graph': graph,
        'pcr': pcr,
        'sil_batch': sil_batch,
        'avg_bio': avg_bio,
        'avg_batch': avg_batch
        }, index=[0])
    return Scores