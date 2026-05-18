

# GNN for multi-omics data integration and classification
This repository contains the implementation of a Graph Neural Network (GNN) model for integrating multi-omics data and performing classification tasks. The model is designed to handle various types of omics data, such as genomics, transcriptomics, and proteomics, and can be applied to different biological datasets.


# Benchmark GNN models
The repository also includes implementations of benchmark GNN models for comparison purposes. These models serve as baselines to evaluate the performance of the proposed multi-omics GNN model.


# Tests
The repository contains a set of tests to validate the functionality and performance of the GNN models. 

## mogonet_test
This test suite focuses on the Multi-Omics Graph Convolutional Network (MOGONET) model. It includes tests for data loading, model training, and evaluation.

<!-- write in red in this md line-->
<span style="color:red"> Added a script to download the datasets to run tests: "get_dataset.py"</span>


Run the script to download datasets using:
```bash
python get_datasets.py
```

Run tests using:
```bash
python test_run.py
```

## gcn_test  
Run GCN model tests using:
```bash
python .\GCN_run.py -fd .\dataset\moGCN\fpkm_data.csv -ad .\dataset\moGCN\gistic_data.csv -ld .\dataset\moGCN\sample_classes.csv
```



## Reti che usano Proir knowledge
- AMOGEL 
- MPK-GNN 
- Multimodal-GNN-for-Cancer-Subtype-Clasification

## Reti che usano Patient similarity network
- MOGONET (GCN + VCDN)
- moGCN (AE + GCN)
- MOGAT (GAT)
- SUMPREME (AE + GCN)