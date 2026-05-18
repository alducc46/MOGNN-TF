#%%
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from utils import z_score_normalization
import os


def preprocess_data(omics_file="fpkm_data.zip", labels_file="sample_classes.csv"):
    omics = pd.read_csv(omics_file)
    print("Before sorting:")
    print(f"Dimensions of omics dataset: {omics.shape}")
    print(omics.head()[omics.columns[:5]])

    labels =  pd.read_csv(labels_file)
    print(f"Dimensions of labels dataset: {labels.shape}")
    print(labels["BRCA_Subtype"])
    

    # sort by sample_id
    omics = omics.sort_values(by="sample_id")
    labels = labels.sort_values(by="sample_id")
    print("After sorting:")
    print(omics.head()[omics.columns[:5]])
    print(labels.head()[labels.columns[:5]])

    omics_X = omics.drop(['sample_id'], axis='columns')
    omics_y = labels["BRCA_Subtype"]

    # display data examples
    print(omics_X.head())
    # print("-----")
    print(omics_y.head())
    return omics_X, omics_y

def run_boruta(omics_X, omics_y):
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, max_depth=5)
    trans = BorutaPy(clf, random_state=42, verbose=2)
    sel = trans.fit_transform(omics_X.values, omics_y.values)
    # Get the boolean mask of selected features
    selected_mask = trans.support_

    # Get the column names of selected features
    selected_features = omics_X.columns[selected_mask].tolist()

    print(f"Number of selected features: {trans.n_features_}")
    print(f"Selected feature names: {selected_features[:10]}...")

    print(f"Dimensions of selected features: {len(selected_features)}")

    return selected_features

def feature_selection_boruta(omics_file='filtered_omics/fpkm_processed.csv', 
                            labels_file='filtered_omics/clinical_processed.csv',
                            include_sample_id=True):
    """
    Complete Boruta feature selection pipeline that returns a filtered dataframe.
    
    Args:
        omics_file (str): Path to omics data CSV file
        labels_file (str): Path to labels CSV file  
        include_sample_id (bool): Whether to include sample_id in output dataframe
    
    Returns:
        pd.DataFrame: Dataframe with selected features (and optionally sample_id)
    """
    print("Starting Boruta feature selection pipeline...")
    
    # Load original data with sample_id
    omics_full = pd.read_csv(omics_file)
    labels_df = pd.read_csv(labels_file)
    
    # Ensure alignment by sorting
    omics_full = omics_full.sort_values(by="sample_id").reset_index(drop=True)
    labels_df = labels_df.sort_values(by="sample_id").reset_index(drop=True)

    print(f"Input data shape: {omics_full.shape}")
    print(f"Labels shape: {labels_df.shape}")

    # Preprocess data (without sample_id for Boruta)
    omics_X, omics_y = preprocess_data(omics_file, labels_file)
    
    # Run Boruta feature selection
    selected_features = run_boruta(omics_X, omics_y)
    
    # Create filtered dataframe with selected features
    if include_sample_id:
        # Include sample_id + selected features
        selected_columns = ['sample_id'] + selected_features
        omics_filtered = omics_full[selected_columns].copy()
        print(f"Final dataframe shape: {omics_filtered.shape}")
        print(f"{len(selected_features)} selected features")
    else:
        # Only selected features
        omics_filtered = omics_full[selected_features].copy()
        print(f"Final dataframe shape: {omics_filtered.shape}")
        print(f"Columns: {len(selected_features)} selected features only")
    
    print(f"Feature reduction: {omics_X.shape[1]} → {len(selected_features)} features")
    print(f"Reduction rate: {(1 - len(selected_features)/omics_X.shape[1])*100:.1f}%")
    
    return omics_filtered

def apply_boruta_and_normalization(input_omic, input_clinical, output_path):
    data_filtered = feature_selection_boruta(
        input_omic, 
        input_clinical,
        include_sample_id=True
    )
    data_filtered.to_csv(output_path, index=False)
    print(f"Saved Boruta-filtered FPKM data: {data_filtered.shape}")
    print(f"Saved to: {output_path}")

def get_boruta_selected_features(input_omic, input_clinical):
    omics_X, omics_y = preprocess_data(input_omic, input_clinical)
    selected_features = run_boruta(omics_X, omics_y)
    return selected_features