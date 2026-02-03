import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
import tqdm as tqdm
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn.functional as F
import torch
import torch.optim as optim
import torch.nn as nn
import tqdm
import os
from torch.utils.data import random_split
import joblib
#set to use gpu
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
from openTSNE import TSNE
#hide gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use only the first GPU
os.environ["TMPDIR"] = "/doctorai/niccoloc/tmp"
os.environ['JOBLIB_TEMP_FOLDER'] = '/doctorai/niccoloc/tmp'



#create UMAP dataframes for the umap model  
def create_umap_dataframe_umap(
    irecptor_emb, ag_emb, 
    ireceptor_metadata_path=None,
    ag_metadata_path=None, 
    ireceptor_label='ireceptor',
    n_ireceptor=3789384,
    device='cpu',
    scaling=True
):
    irecptor_transf = irecptor_emb
    df_i = pd.read_csv(ireceptor_metadata_path)
    print(f"iReceptor metadata loaded with {len(df_i)} entries.")
#print shapes
    print(f"iReceptor embeddings shape: {irecptor_transf.shape}")
    print(f"Antigen embeddings shape: {ag_emb.shape}")
    print(f"iReceptor metadata shape: {df_i.shape}")
    print(n_ireceptor)

    df_i[['umap_1', 'umap_2']] = irecptor_transf[:n_ireceptor]
    df_i['specificity'] = ireceptor_label

    ag_metadata_df = pd.read_csv(ag_metadata_path)
    ag_metadata_df = ag_metadata_df.join(
        pd.DataFrame(ag_emb, columns=['umap_1', 'umap_2']),
        how='inner'
    )
    combined = pd.concat([df_i, ag_metadata_df], ignore_index=True)
    return combined







# File paths
ireceptor_metadata = '/doctorai/niccoloc/ireceptor_3M_nodup_max35L_pgen.csv'
irecep_esm2 = '/doctorai/userdata/airr_atlas/data/embeddings/ireceptor2/esm2/embeddings/ireceptor_3M_nodup_max35L_esm2_embeddings_layer_33.pt'
irecep_ab2 = '/doctorai/userdata/airr_atlas/data/embeddings/ireceptor2/ab2/embeddings/ireceptor_3M_nodup_max35L_ab2_embeddings_layer_16.pt'
ag_metadata = '/doctorai/userdata/airr_atlas/data/sequences/bcr/ALL_ANTIGENS/antigen_specific_df_2025.csv'
ag_esm2 = '/scratch/niccoloc/ag_dataset/esm2_t33_650M_UR50D/embeddings/ag_dataset_esm2_t33_650M_UR50D_embeddings_layer_33.npy'
ag_ab2 = '/scratch/niccoloc/ag_dataset/antiberta2-cssp/embeddings/ag_dataset_antiberta2-cssp_embeddings_layer_16.npy'


#NEW PATHS
ireceptor_metadata = '/doctorai/niccoloc/ireceptor_NEW_final_onlyseqid.csv'
irecep_esm2 = '/doctorai/niccoloc/ireceptor/esm2_t33_650M_UR50D/mean_pooled/ireceptor_2M_esm2_t33_650M_UR50D_mean_pooled_layer_33.npy'
irecep_ab2 = '/doctorai/niccoloc/ireceptor/antiberta2-cssp/mean_pooled/ireceptor_2M_antiberta2-cssp_mean_pooled_layer_16.npy'
ag_metadata = '/doctorai/userdata/airr_atlas/data/sequences/bcr/ALL_ANTIGENS/antigen_specific_df_2025.csv'
ag_esm2 = '/scratch/niccoloc/ag_dataset/esm2_t33_650M_UR50D/embeddings/ag_dataset_esm2_t33_650M_UR50D_embeddings_layer_33.npy'
ag_ab2 = '/scratch/niccoloc/ag_dataset/antiberta2-cssp/embeddings/ag_dataset_antiberta2-cssp_embeddings_layer_16.npy'



# import joblib
# joblib.dump(tsne_embeddings, f"/doctorai/niccoloc/vae_umap_dfs/tsne_model_Ireceptor_{emb_name}.joblib")

import tempfile
import os
from functools import partial

# Define your custom temp path
custom_tmp = "/doctorai/niccoloc/tmp"
os.makedirs(custom_tmp, exist_ok=True)

#optional
# patch TemporaryDirectory to allow a custom dir as temp dir , so that opentsne uses that instead of /tmp which has limited space
_real_TemporaryDirectory = tempfile.TemporaryDirectory

def _custom_tempdir(*args, **kwargs):
    kwargs["dir"] = custom_tmp
    return _real_TemporaryDirectory(*args, **kwargs)

tempfile.TemporaryDirectory = _custom_tempdir



#train on validation set #--------------------------------------------------------------------------------------------------------------------------------------

from torch.utils.data import random_split
import os
import pandas as pd
import joblib

# Define embedding sources
embedding_sources = {
    "esm2": {
        "ireceptor": irecep_esm2,
        "antigen": ag_esm2
    },
    "ab2": {
        "ireceptor": irecep_ab2,
        "antigen": ag_ab2
    }
}

tsne_models = {}




x1=np.load(irecep_esm2, mmap_mode='r')
idx=pd.read_csv('/doctorai/niccoloc/ireceptor/esm2_t33_650M_UR50D/ireceptor_NEW_final_idx.csv')
metadata=pd.read_csv(ireceptor_metadata)
print(f"Loaded {len(metadata)} ireceptor sequences with {x1.shape[1]} features each.")
print(f"Shape of ireceptor embeddings: {x1.shape}")
 



for emb_name, paths in embedding_sources.items():
    print(f"Processing {emb_name} embeddings...")
    model_in_use= emb_name

    # Load receptor and antigen embeddings
    if paths["ireceptor"].endswith(".pt"):
        irecp_tensor = torch.load(paths["ireceptor"]).float()
    else:
        irecp_tensor = torch.tensor(np.load(paths["ireceptor"])).float()


    if paths["antigen"].endswith(".pt"):
        ag_tensor = torch.load(paths["antigen"]).float()
    else:
        ag_tensor = torch.tensor(np.load(paths["antigen"])).float()


    # Run TSNE
    tsne = TSNE(
        perplexity=30,
        metric="cosine",
        n_jobs=100,
        random_state=123,
        verbose=True,
        early_exaggeration=25.0,
        initialization="pca"
    )

    # tsne_embeddings = tsne.fit(irecp_tensor.numpy()[:100000])  # Limit to first 1000 for speed
    tsne_embeddings = tsne.fit(irecp_tensor.numpy() )  
    tsne_embeddings_transform = tsne_embeddings.transform(irecp_tensor.numpy())
    ag_specific_tsne = tsne_embeddings.transform(ag_tensor.numpy())
    tsne_models[emb_name] = tsne_embeddings


    tsne_df = create_umap_dataframe_umap(
        irecptor_emb=tsne_embeddings_transform
,
        ag_emb=ag_specific_tsne,
        # n_ireceptor=100000,
        ireceptor_metadata_path=ireceptor_metadata,
        ag_metadata_path=ag_metadata,
        ireceptor_label='ireceptor'
    )

    out_path = f"/doctorai/niccoloc/vae_umap_dfs/tsne_Ireceptor_{emb_name}.csv"
    #save the fitted tsne model 
    try:
        joblib.dump(tsne_embeddings, f"/doctorai/niccoloc/vae_umap_dfs/tsne_model_Ireceptor_{emb_name}_NEW.joblib", compress=3)
    except Exception as e:
        print(f"Failed to save TSNE model for {emb_name}: {e}")

 

    # Save the dataframe to CSV
    tsne_df.to_csv(out_path, index=False)
    print(f"Saved TSNE dataframe for {emb_name} to {out_path}")


joblib.dump(tsne_models['esm2'], f"/doctorai/niccoloc/vae_umap_dfs/tsne_model_Ireceptor_{emb_name}.joblib")


#print the current temp dire
print(f"Temporary directory: {os.environ['TMPDIR']}")
# transform the Tz dataset

# paths for tz dataset and embeddings
input_metadata="/doctorai/userdata/airr_atlas/data/files_for_trastuzumab/tz_heavy_chains_airr_dedup_final.tsv"
esm2_tz='/doctorai/userdata/airr_atlas/data/embeddings/trastuzumab_npy/esm2_t33_650M_UR50D/embeddings/tz_cdr3_100k_esm2_t33_650M_UR50D_embeddings_layer_33.npy'
ab2_tz='/doctorai/userdata/airr_atlas/data/embeddings/trastuzumab_npy/antiberta2-cssp/embeddings/tz_cdr3_100k_antiberta2-cssp_embeddings_layer_16.npy'
idx_reference='/doctorai/userdata/airr_atlas/data/embeddings/trastuzumab_npy/antiberta2-cssp/tz_cdr3_100k_idx.csv'


metadata_porebski='/doctorai/userdata/airr_atlas/data/sequences/bcr/porebski/porebski_metadata_vsALL.csv'
esm2_pb='/doctorai/userdata/airr_atlas/data/embeddings/porebski_npy/esm2_t33_650M_UR50D/embeddings/porebski_cdr3_only_esm2_t33_650M_UR50D_embeddings_layer_33.npy'
ab2_pb='/doctorai/userdata/airr_atlas/data/embeddings/porebski_npy/antiberta2-cssp/embeddings/porebski_cdr3_only_antiberta2-cssp_embeddings_layer_16.npy'
idx_reference_pb='/doctorai/userdata/airr_atlas/data/embeddings/porebski_npy/esm2_t33_650M_UR50D/porebski_cdr3_only_idx.csv'




def load_data(input_metadata, input_embeddings,idx_reference ,df_junction_colname='cdr3_aa', df_affinity_colname='binding_label'):
    if not os.path.exists(input_metadata):
        raise FileNotFoundError(f"Metadata file not found: {input_metadata}")
    if not os.path.exists(input_embeddings):
        raise FileNotFoundError(f"Embeddings file not found: {input_embeddings}")
    
    tensors = np.load(input_embeddings, mmap_mode='r')
    # tensors = torch.load(input_embeddings).numpy()  
    seqs = pd.read_csv(input_metadata, sep=None , engine ='python')
    # Check for NaN values in seqs and convert to "unkn"
    print("Checking for NaN values in the sequences dataframe...")
    # Check if the dataframe contains any NaN values
    if seqs.isna().any().any():
        print(f"Found NaN values in the dataframe.")
        # Check specifically for NaN values in the binding label column
        if args.df_affinity_colname in seqs.columns and seqs[args.df_affinity_colname].isna().any():
            print(f"Found {seqs[args.df_affinity_colname].isna().sum()} NaN values in the binding label column '{args.df_affinity_colname}'. Converting them to 'unkn'...")
            seqs[args.df_affinity_colname] = seqs[args.df_affinity_colname].fillna("unkn")
        # Fill NaN values in other columns as well
        seqs = seqs.fillna("unkn")
    else:
        print("No NaN values found in the sequences dataframe.")
    if idx_reference == "":
        seqs['id'] = np.arange(0, len(seqs))
        tensors_df = pd.DataFrame({
            'id': np.arange(0, len(tensors)),
            'embedding': list(tensors)
        })
        df = pd.merge(seqs, tensors_df, on='id')
    else:
        #'/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis/antiberta2/full_chain/100k_sample_trastuzmab_full_chain_antiberta2_idx.csv'
        #'/doctorai/userdata/airr_atlas/data/files_for_trastuzumab/tz_heavy_chains_airr_dedup_final.tsv'
        idx_df= pd.read_csv(idx_reference, sep =None , engine ='python')
        print(idx_df.head())
        print(tensors[:5])
        print(seqs.head())
        tensors_df = pd.DataFrame({
            'tensor_id': idx_df['index'],
            'sequence_id' : idx_df['sequence_id'],
            # 'embedding': list(tensors)
        })
        df = pd.merge(seqs, tensors_df, on='sequence_id')
        # embeddings = tensors[[df['tensor_id'].values]]  # extracting the correct entries matching the sampled indices
        # print(embeddings)
    print("...Removing duplicated sequences ...")
    df = df[~df[df_junction_colname].duplicated(keep=False)]
    print(f"Number of sequences in the dataset after de-deuplication: {len(df)}, per class: {df[df_affinity_colname].value_counts()}")
    embeddings = tensors[df['tensor_id'].values]
    df = df.reset_index(drop=True)
    df['id'] = np.arange(0, len(df))

    return df , embeddings

df , embeddings  = load_data(input_metadata, esm2_tz, idx_reference)

# Transform the dataset using the TSNE model, both for esm2 and ab2
def transform_dataset_with_tsne(df, embeddings, tsne_model):
    print("Transforming dataset with TSNE...")
    transformed_embeddings = tsne_model.transform(embeddings)
    df['umap_1'] = transformed_embeddings[:, 0]
    df['umap_2'] = transformed_embeddings[:, 1]
    return df

#loop through the models and transform the dataset
for emb_name, tsne_model in tsne_models.items():
    print(f"Transforming dataset with {emb_name} TSNE model...")
    #load dataset and embeddings
    if emb_name == 'esm2':
        df, embeddings = load_data(input_metadata, esm2_tz, idx_reference)
    elif emb_name == 'ab2':
        df, embeddings = load_data(input_metadata, ab2_tz, idx_reference)

    transformed_df = transform_dataset_with_tsne(df, embeddings, tsne_model)
    
    # Save the transformed dataframe
    out_path = f"/doctorai/niccoloc/vae_umap_dfs/tsne_tz_{emb_name}_NEW.csv"
    transformed_df.to_csv(out_path, index=False)
    print(f"Saved transformed TSNE dataframe for {emb_name} to {out_path}")



for emb_name, tsne_model in tsne_models.items():
    print(f"Transforming Porebski dataset with {emb_name} TSNE model...")
    #load dataset and embeddings
    if emb_name == 'esm2':
        df_pb, embeddings_pb = load_data(metadata_porebski, esm2_pb, idx_reference_pb,
                                         df_junction_colname='cdr3', df_affinity_colname='binding_label')
    elif emb_name == 'ab2':
        df_pb, embeddings_pb = load_data(metadata_porebski, ab2_pb, idx_reference_pb,
                                         df_junction_colname='cdr3', df_affinity_colname='binding_label')

    transformed_df_pb = transform_dataset_with_tsne(df_pb, embeddings_pb, tsne_model)
    
    # Save the transformed dataframe
    out_path = f"/doctorai/niccoloc/vae_umap_dfs/tsne_porebski_{emb_name}_NEW.csv"
    transformed_df_pb.to_csv(out_path, index=False)
    print(f"Saved transformed TSNE dataframe for Porebski {emb_name} to {out_path}")


#load new ireceptor data
ireceptor_metadata_new = '/doctorai/niccoloc/ireceptor_3M_nodup_max35L_pgen.csv'
ireceptor_embeddings_esm2= '/scratch/niccoloc/ireceptor/esm2_t33_650M_UR50D/embeddings/ireceptor_3M_nodup_max35L_pgen_esm2_t33_650M_UR50D_embeddings_layer_33.npy'




#load syhtetic prepost data
prepost = pd.read_csv('/doctorai/niccoloc/prepost_sample_15aa_heavy_chain_100k_pgens.csv')
prepost_embeddings_esm2 = np.load('/scratch/niccoloc/prepost/esm2_t33_650M_UR50D/embeddings/prepost100k_esm2_t33_650M_UR50D_embeddings_layer_33.npy') 
prepost_embeddings_ab2 =  np.load('/scratch/niccoloc/prepost/antiberta2-cssp/embeddings/prepost100k_antiberta2-cssp_embeddings_layer_16.npy') 

#load new ireceptor data
ireceptor_metadata_new = '/doctorai/niccoloc/ireceptor_3M_nodup_max35L_pgen.csv'
ireceptor_embeddings_esm2= '/scratch/niccoloc/ireceptor/esm2_t33_650M_UR50D/embeddings/ireceptor_3M_nodup_max35L_pgen_esm2_t33_650M_UR50D_embeddings_layer_33.npy'


import pandas as pd



import joblib
tsne_models={}
tsne_models['esm2'] = joblib.load(f"/doctorai/niccoloc/vae_umap_dfs/tsne_model_Ireceptor_esm2_NEW.joblib")
tsne_models['ab2']  = joblib.load(f"/doctorai/niccoloc/vae_umap_dfs/tsne_model_Ireceptor_ab2_NEW.joblib")
# Transform the prepost dataset using the TSNE model, both for esm2 and ab2

def transform_with_tsne(metadata, input_embeddings, tsne_model):
    print("Transforming prepost dataset with TSNE...")
    transformed_embeddings = tsne_model.transform(input_embeddings)
    metadata['umap_1'] = transformed_embeddings[:, 0]
    metadata['umap_2'] = transformed_embeddings[:, 1]
    return metadata
#loop through the models and transform the prepost dataset
for emb_name, tsne_model in tsne_models.items():
    print(f"Transforming prepost dataset with {emb_name} TSNE model...")
    if emb_name == 'esm2':
        prepost_embeddings = prepost_embeddings_esm2
    elif emb_name == 'ab2':
        prepost_embeddings = prepost_embeddings_ab2
    
    transformed_prepost = transform_with_tsne(prepost, prepost_embeddings, tsne_model)
    
    # Save the transformed dataframe
    out_path = f"/doctorai/niccoloc/vae_umap_dfs/tsne_prepost_{emb_name}.csv"
    transformed_prepost.to_csv(out_path, index=False)
    print(f"Saved transformed TSNE dataframe for prepost {emb_name} to {out_path}")


ireceptor_df_new = pd.read_csv(ireceptor_metadata_new)
ireceptor_embeddings_esm2 = np.load(ireceptor_embeddings_esm2)  # Load the new ireceptor embeddings
#loop through the models and transform the ireceptor dataset
for emb_name, tsne_model in tsne_models.items():
    print(f"Transforming ireceptor dataset with {emb_name} TSNE model...")
    if emb_name == 'esm2':
        ireceptor_embeddings = ireceptor_embeddings_esm2
    elif emb_name == 'ab2':
        ireceptor_embeddings = ireceptor_embeddings_ab2
    
    transformed_ireceptor = transform_with_tsne(ireceptor_df_new, ireceptor_embeddings, tsne_model)
    
    # Save the transformed dataframe
    out_path = f"/doctorai/niccoloc/vae_umap_dfs/tsne_ireceptor_{emb_name}_PGEN.csv"
    transformed_ireceptor.to_csv(out_path, index=False)
    print(f"Saved transformed TSNE dataframe for ireceptor {emb_name} to {out_path}")

            






#load syhtetic prepost data
prepost = pd.read_csv('/doctorai/niccoloc/prepost_sonia_2M_pgens_OK2.csv')
prepost_embeddings_esm2 = np.load('/doctorai/niccoloc/simulated/esm2_t33_650M_UR50D/mean_pooled/prepost_2M_esm2_t33_650M_UR50D_mean_pooled_layer_33.npy') 
prepost_embeddings_ab2 =  np.load('/doctorai/niccoloc/simulated/antiberta2-cssp/mean_pooled/prepost_2M_antiberta2-cssp_mean_pooled_layer_16.npy') 


prepost = pd.read_csv('/doctorai/niccoloc/post_selection_1M_pgen_NEW.csv')
prepost_embeddings_esm2 = np.load('/doctorai/niccoloc/postselection_1M/esm2_t33_650M_UR50D/mean_pooled/postselection_1M_esm2_t33_650M_UR50D_mean_pooled_layer_33.npy') 
prepost_embeddings_ab2 =  np.load('/doctorai/niccoloc/postselection_1M/antiberta2-cssp/mean_pooled/postselection_1M_antiberta2-cssp_mean_pooled_layer_16.npy') 


import pandas as pd



import joblib
# tsne_models={}
# tsne_models['esm2'] = joblib.load(f"/doctorai/niccoloc/vae_umap_dfs/tsne_model_Ireceptor_esm2.joblib")
# tsne_models['ab2'] =joblib.load(f"/doctorai/niccoloc/vae_umap_dfs/tsne_model_Ireceptor_ab2.joblib")
# Transform the prepost dataset using the TSNE model, both for esm2 and ab2



def transform_with_tsne(metadata, input_embeddings, tsne_model):
    print("Transforming prepost dataset with TSNE...")
    transformed_embeddings = tsne_model.transform(input_embeddings)
    metadata['umap_1'] = transformed_embeddings[:, 0]
    metadata['umap_2'] = transformed_embeddings[:, 1]
    return metadata



#loop through the models and transform the prepost dataset
for emb_name, tsne_model in tsne_models.items():
    print(f"Transforming prepost dataset with {emb_name} TSNE model...")
    if emb_name == 'esm2':
        prepost_embeddings = prepost_embeddings_esm2
    elif emb_name == 'ab2':
        prepost_embeddings = prepost_embeddings_ab2
    
    transformed_prepost = transform_with_tsne(prepost, prepost_embeddings, tsne_model)
    
    # Save the transformed dataframe
    out_path = f"/doctorai/niccoloc/vae_umap_dfs/tsne_prepost_{emb_name}_NEW.csv"
    out_path = f"/doctorai/niccoloc/vae_umap_dfs/tsne_POST_1M_{emb_name}_NEW.csv"
    transformed_prepost.to_csv(out_path, index=False)
    print(f"Saved transformed TSNE dataframe for prepost {emb_name} to {out_path}")















#train regression model to predict the pgen model from the embeddings
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange



#load ireceptor embeddings and metadata
ireceptor_metadata = '/doctorai/niccoloc/ireceptor_3M_nodup_max35L_pgen.csv'
ireceptor_embeddings = '/doctorai/userdata/airr_atlas/data/embeddings/ireceptor2/esm2/embeddings/ireceptor_3M_nodup_max35L_esm2_embeddings_layer_33.pt'

ireceptor_df = pd.read_csv(ireceptor_metadata)
ireceptor_embeddings_tensor = torch.load(ireceptor_embeddings)[:3789384].float()  # Limit to first 3789384 

# Prepare data
X = ireceptor_embeddings_tensor.numpy()
y = ireceptor_df['pgen'].values.reshape(-1, 1)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Simple: use log1p so you don’t blow up on zero
e=1e-50
y_train_log = np.log(y_train + e)
y_test_log  = np.log(y_test +e)


# Then fit your scaler on the log‐values
scaler_y = StandardScaler().fit(y_train_log)
y_train_scaled = scaler_y.transform(y_train_log)
y_test_scaled  = scaler_y.transform(y_test_log)


# Convert to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# MLP regressor with Dropout, BatchNorm, and an added 512 layer
class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        return self.fc4(x)

model = MLPRegressor(X_train.shape[1])
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Training loop
epochs = 100
batch_size = 1024
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)



for epoch in trange(epochs, desc="Training Epochs"):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(X_train)}")
    #save the log train loss
    with open(f"/doctorai/niccoloc/vae_umap_dfs/train_loss_PGEN_regressor.txt", "w") as f:
        f.write(f"Epoch {epoch+1}, Loss: {total_loss / len(X_train)}\n")

# Evaluate
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_cpu = y_pred.cpu().numpy()
    y_test_cpu = y_test_tensor.cpu().numpy()
     
    # Now apply inverse_transform on the CPU NumPy arrays
    y_pred_inv = scaler_y.inverse_transform(y_pred_cpu)
    y_test_inv = scaler_y.inverse_transform(y_test_cpu)    
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)
    print(f"Test MSE: {mse:.4f}, R2: {r2:.4f}")


y_pred_inv
y_test_inv



#tesing transform with TSNE
import joblib

tsne_embeddings_transf = tsne_embeddings.transform(irecp_tensor.numpy()[:1000] )

tsne_embeddings== tsne_embeddings_transf
tsne_embeddings_transf


#save the model
joblib.dump(tsne_embeddings, f"/doctorai/niccoloc/vae_umap_dfs/tsne_model_Ireceptor_{emb_name}.joblib")


joblib.dump({
    'embedding': tsne_embeddings,
    'affinities': tsne.affinities,
    'params': {
        'perplexity': 30,
        'metric': 'cosine',  # whatever you used
        'random_state': 123,  # whatever you used
        'early_exaggeration' :25.0,  # whatever you used
        # add more params as needed
    }
}, "minimal_tsne.pkl")

