#%%
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from process_data import variance_processing, normalize_omics_data
from boruta_feature_selection import get_boruta_selected_features
import os

input_dir = "data/omics"
output_dir = "data/train_test_split"

def get_shuffled_indices(data_length, random_state=42):
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(data_length)
    return shuffled_indices

def split_and_save_data(file_name, test_size=0.2, random_state=42):
    # Load the dataset
    data = pd.read_csv(f"{input_dir}/{file_name}")
    
    # Shuffle indices
    print(f"Loading data from {file_name} with shape: {data.shape}")
    print(f"data length: {len(data)}")
    shuffled_indices = get_shuffled_indices(len(data), random_state)
    print(f"Shuffled indices for {file_name}: {shuffled_indices[:10]}...")  # Print first 10 for verification
    # export shuffled indices
    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(f"{output_dir}/shuffled_indices_{file_name.replace('.csv', '')}.txt", shuffled_indices, fmt='%d')
    # Split the data index
    train_indices, test_indices = train_test_split(
        shuffled_indices,
        test_size=test_size,
        random_state=random_state
    )
    print(f"Train indices for {file_name}: {train_indices[:10]}...")  # Print first 10 for verification
    print(f"Test indices for {file_name}: {test_indices[:10]}...")  # Create train and test datasets

    
    train_data = data.iloc[train_indices]
    test_data = data.iloc[test_indices]
    
    # Save the split datasets
    train_data.to_csv(f"{output_dir}/train_{file_name}", index=False)
    test_data.to_csv(f"{output_dir}/test_{file_name}", index=False)
    #export indices
    np.savetxt(f"{output_dir}/train_indices_{file_name.replace('.csv', '')}.txt", train_indices, fmt='%d')
    np.savetxt(f"{output_dir}/test_indices_{file_name.replace('.csv', '')}.txt", test_indices, fmt='%d')
    print(f"Saved train and test data for {file_name} to {output_dir}")

#def main():
files_to_split = [
    "fpkm.csv"
]
#%%
for file_name in files_to_split:
    split_and_save_data(file_name)
# load and process variance data
expression_train = os.path.join(output_dir, "train_fpkm.csv")
train_df = pd.read_csv(expression_train)
normalized_train = normalize_omics_data(train_df, "Expression")
# export normalized train
normalized_train.to_csv(os.path.join(output_dir, "normalized_train_fpkm.csv"), index=False)
#%%
# variance processing
variance_processing(os.path.join(output_dir, "normalized_train_fpkm.csv"), os.path.join(output_dir,"variance_filtered_genes.txt"))
#%%
#%%
#select top 1000 genes by variance
variance_file = os.path.join(output_dir,"variance_filtered_genes.txt")

top_genes = pd.read_csv(variance_file).nlargest(1000, 'variance')['gene'].tolist()
# load normalized train data
# export filtered train data
top_genes_file = os.path.join(output_dir, "top_1000_variance_genes.txt")
with open(top_genes_file, 'w') as f:
    for gene in top_genes:
        f.write(f"{gene}\n")
normalized_train = pd.read_csv(os.path.join(output_dir, "normalized_train_fpkm.csv"))
# filter clinical data by train samples
clinical_data = pd.read_csv(os.path.join(input_dir, "clinical.csv"))
train_samples = normalized_train['sample_id'].tolist()
filtered_clinical = clinical_data[clinical_data['sample_id'].isin(train_samples)]
filtered_clinical.to_csv(os.path.join(output_dir, "train_clinical.csv"), index=False)
# filter normalized train data by top genes
# apply boruta feature selection
selected_genes = get_boruta_selected_features(
    input_omic=os.path.join(output_dir, "normalized_train_fpkm.csv"),
    input_clinical=os.path.join(output_dir, "train_clinical.csv")
)
# export selected genes
boruta_genes_file = os.path.join(output_dir, "boruta_selected_genes.txt")
with open(boruta_genes_file, 'w') as f:
    for gene in selected_genes:
        f.write(f"{gene}\n")

# filter to top genes
#%%
mirna_data = pd.read_csv(os.path.join(input_dir, "mirna.csv"))
# filter sample ids to train samples
filtered_mirna = mirna_data[mirna_data['sample_id'].isin(train_samples)]
# export filtered mirna data
filtered_mirna.to_csv(os.path.join(output_dir, "filtered_mirna.csv"), index=False)
normalized_mirna = normalize_omics_data(filtered_mirna, "miRNA")
# export normalized mirna
normalized_mirna.to_csv(os.path.join(output_dir, "normalized_mirna.csv"), index=False)
#%%
variance_processing(os.path.join(output_dir, "normalized_mirna.csv"), os.path.join(output_dir,"mirna_variance_filtered_features.txt"))
# select top 100 mirna by variance
print("Selecting top 100 miRNAs by variance...")
mirna_variance_file = os.path.join(output_dir,"mirna_variance_filtered_features.txt")
print(f"Number of miRNAs in variance file: {len(pd.read_csv(mirna_variance_file))}")
top_mirnas = pd.read_csv(mirna_variance_file).nlargest(100, 'variance')['gene'].tolist()
top_mirnas_file = os.path.join(output_dir, "top_100_variance_mirnas.txt")
with open(top_mirnas_file, 'w') as f:
    for mirna in top_mirnas:
        f.write(f"{mirna}\n")


#if __name__ == "__main__":
#    main()
# %%
# filter
