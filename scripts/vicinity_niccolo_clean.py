

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
#from umap import UMAP
import matplotlib.pyplot as plt
import pandas as pd
from scipy import linalg
import torch
# import umap.plot
import Levenshtein as lev
from sklearn.decomposition import PCA
from scipy.spatial import distance
from tqdm.auto import tqdm
from scipy.stats import pearsonr
from scipy.stats import mannwhitneyu
from scipy.spatial.distance import jensenshannon
import concurrent.futures
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
import time
import pickle


"""# **Load sequences and tensors**"""

seqs = pd.read_csv(f'/doctorai/marinafr/2023/airr_atlas/analysis/data/all_data/all_data.csv', sep=';')
seqs = seqs[seqs['dataset']=='trastuzumab']
# IMPORTANT !!!!changed from 'id': np.arange(1, len(tensors) + 1), leads to problems with the id retrievial in the 
#KNN code
seqs['id'] = np.arange(0, len(seqs) )   


#------ to concatenate multiple embeddings -------
# tensors_PLM = torch.load(f'/doctorai/niccoloc/trastuzumab_AB2_CSSP.pt')
# tensors_fold = torch.load(f'/doctorai/niccoloc/trastuzumab_IgFold_emb_512dim.pt')
# tensors_fold =torch.cat((tensors_fold[0,].unsqueeze(0),tensors_fold), dim=0) # IgFold missed the first row, this "corrects" it
# tensors = torch.cat((tensors_PLM,tensors_fold), dim=1).numpy()
# tensors.shape
#------------


tensors = torch.load(f'/doctorai/userdata/airr_atlas/data/embeddings/trastuzumab_AB2_CSSP.pt').numpy()


tensors_df = pd.DataFrame({
    'id': np.arange(0, len(tensors)), # IMPORTANT !!!!changed from 'id': np.arange(1, len(tensors) + 1),
    'embedding': list(tensors),
    # 'embedding_fold': list(tensors),
    # Convert the array of arrays into a list to be stored in the DataFrame
})

# merge and sample
df = pd.merge(seqs, tensors_df, on='id')
df = df[['junction_aa', 'id', 'embedding', 'dataset', 'affinity']] 
df['junction_length'] = df['junction_aa'].apply(len)
df = df[df['junction_length'] <= 40] 

df_hb = df[df['affinity'] == 'hb']
df_lb = df[df['affinity'] == 'lb']
df_mb = df[df['affinity'] == 'mb']

df_hb_1000 = df_hb.sample(n=20000, random_state=123)
df_lb_1000 = df_lb.sample(n=20000, random_state=123)
df_mb_1000 = df_mb.sample(n=20000, random_state=123)


df_hblbmb_5000 = pd.concat([df_hb_1000, df_lb_1000,df_mb_1000], ignore_index=True)
id_index_mb= df_hblbmb_5000['id']


# KNN code Niccolo
from sklearn.neighbors import NearestNeighbors


def compute_levenshtein(ref_seq, sequences):
    return [lev.distance(ref_seq, seq) for seq in sequences]

# Function to calculate the fraction of neighbors with the same affinity
def calculate_fraction(neigh, df, indices_slice, n_neighbors, given_index):
    given_affinity = df.iloc[given_index]['affinity']
    neighbors_indices = indices_slice[1:n_neighbors+1]  # Subsetting the indices for n_neighbors
    neighbors_affinity = df.iloc[neighbors_indices]['affinity']
    percentage = (neighbors_affinity == given_affinity).sum()
    percentage = (neighbors_affinity == given_affinity).sum() / len(neighbors_affinity)
    return percentage

# Calculate mean percentage of neighbors with the same affinity for a subset
def calculate_fractions_for_subset(df, neigh, indices, neighbor_subset, id_index):
    percentages = []
    for n in neighbor_subset:
        for idx, given_index in enumerate(id_index):
            indices_slice = indices[idx]  # Get the correct slice of indices for the current id_index
            percentage = calculate_fraction(neigh, df, indices_slice, n, given_index)
            percentages.append(percentage)
    percentages=  np.mean(percentages)
    return percentages

# Main function to calculate fractions for different neighbor numbers
def calculate_fractions_for_data(df, neighbor_numbers, id_index):
    neigh = NearestNeighbors()
    neigh.fit(list(df['embedding']))
    # Compute the nearest neighbors for the maximum number of neighbors needed
    distances, indices = neigh.kneighbors(df.iloc[id_index]['embedding'].tolist(), n_neighbors=max(neighbor_numbers))
    indices_affinity = df.loc[indices.flatten(), 'affinity'].values
    # Redimension affinity values array to corrispond to indices shape
    id_affinty_label=df.loc[id_index, 'affinity']
    indices_affinity_mat = indices_affinity.reshape(indices.shape)
    t_lev= time.time()
    lev_mat=[]
    for idx, seq_index in enumerate(id_index):
      NN_index=indices[idx]
      lev_mat.append(compute_levenshtein(df.iloc[seq_index]['junction_aa'],df.iloc[NN_index]['junction_aa']) )
    lev_mat= np.array(lev_mat)
    print(f' LEV running time {time.time()-t_lev}')
    res1=[]
    t_NN= time.time()
    for n in neighbor_numbers:
        # No parallel processing, direct function call
        result = calculate_fractions_for_subset(df, neigh, indices, [n], id_index)
        res1.append(result)
    print(f' NN running time {time.time()-t_lev}')
    return res1,indices,distances,indices_affinity_mat,lev_mat,id_affinty_label



max_neighbors = 1000 # This is the maximum number of neighbors you're interested in
neighbor_numbers = np.arange(2, max_neighbors+1, 4)  # This adjusts based on


# First part: numbers from 2 to 300 with steps of 4
part1 = np.arange(2, 304, 4)  # Includes 300
# Second part: numbers from 300 to 1000 with steps of 50
part2 = np.arange(350, max_neighbors+1, 50)  # Starts from 350 to avoid duplicating 300
neighbor_numbers = np.concatenate((part1, part2))

id_index_sample= df_hblbmb_5000['id'] # indexes of 20k each lb mb hb


fractions_results,NN_id,NN_dist, NN_label, NN_lev, ID_labels = calculate_fractions_for_data(df, neighbor_numbers, id_index_sample)

#runnin on the whole dataset
fractions_results,NN_id,NN_dist, NN_label, NN_lev, ID_labels = calculate_fractions_for_data(df, neighbor_numbers, df['id'])

data_to_save = {
    "fractions_results": fractions_results,
    "NN_id": NN_id,
    "NN_dist": NN_dist,
    "NN_label": NN_label,
    "NN_lev": NN_lev,
    "ID_labels": ID_labels
}

# Saving Obj to pickle
with open('KNN_trastuzumab_AB2_WHOLE.pkl', 'wb') as file:
    pickle.dump(data_to_save, file)

#to easily check the result at a give NN_threshold
result_dict = dict(zip(neighbor_numbers, fractions_results))

# TO DO!----------------------------------------------------------------
#I need help to add  the ID_labels to the multidim array: it's an single vector thus doesn't have the same 
#shape of the other arrays, 
#  UPDATE ----- the first entry of id, labels and distance is the vertex point!!!!

#----------OPEN for suggestions or a more felxible data structure ---------------
multidim_array=np.stack((NN_id,NN_label,NN_dist,NN_lev), axis=-1)

#first ID
multidim_array[0,:,:]

#first 2 NN
multidim_array[0,:2,:]

#first ID ,first 2 NN, just the NN id
multidim_array[0,:2,0]

# the last argument selects the type of information that you want to retrieve 
# 0 = NN_id, 1= NN affinity labels, 2= NN_distance, 3= NN_lev_distance 
#  UPDATE ----- the first entry of id, labels and distance is the vertex point!!!!
multidim_array[0,0,3]


# plots

# Basic Vicinity score plots
plt.clf()
plt.figure(figsize=(10, 6))  
plt.scatter(neighbor_numbers, fractions_results, color='blue', label='Data Points') 
plt.plot(neighbor_numbers, fractions_results, color='red', label='Interpolated Line')  
plt.title('Fraction Results by Number of Neighbors')  
plt.xlabel('Number of Neighbors')  
plt.ylabel('Fraction Results') 
plt.grid(True)
plt.legend()
plt.savefig('Vicinity_score_NN_WHOLE.png')

#combined plot  Fraction Results VS LD

#get mean_lev distance x each  NN_number thershold
NN_lev_mean=[]
row_mean=[]
for n in neighbor_numbers:
  for i in range(len(NN_lev[:,])):
    row_mean.append(np.mean(NN_lev[i,1:n+1]) )
  # print(row_mean)
  # print(len(row_mean))
  NN_lev_mean.append(np.mean(row_mean) )
  row_mean=[]

corr_pearson_NN_thr, _ = pearsonr(fractions_results, NN_lev_mean)
print(f"Pearson corr between EU vs LD of each kNN: {corr_pearson_NN_thr}")
# plot x K NN correlation
plt.clf()
plt.figure(figsize=(10, 6))
#first plot part
ax1 = plt.gca()  
ax1.scatter(neighbor_numbers, fractions_results, color='blue', label='AB2 embeddings eucl. distance ')
ax1.set_xlabel('Number of Neighbors')
ax1.set_ylabel('Fraction Results', color='blue')
ax1.set_xticks(neighbor_numbers)  
ax1.set_xticklabels(neighbor_numbers, rotation=90) 
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True)
#second plot part 
ax2 = ax1.twinx()  # shared y axis
ax2.scatter(neighbor_numbers, NN_lev_mean, color='red', label='LD distance')
ax2.set_ylabel('Mean Lev', color='red')
ax2.tick_params(axis='y', labelcolor='red')
# Title
plt.title(f'AB2 EU vs LD by kNN__ pearsonCorr = {corr_pearson_NN_thr}')
plt.savefig('combined_graph_EUvsLD_kNN.png')



### plottt rowCorrelation

avg_mat1 = np.mean(NN_lev, axis=1)
avg_mat2 = np.mean(NN_dist, axis=1)

#  Pearson betweeen rowMEANS
corr_pearson, _ = pearsonr(avg_mat1, avg_mat2)
# corr_pearson, _ = pearsonr(NN_lev, NN_dist)

print(f"Pearson corr. between rowMEANS: {corr_pearson}")

#rowMEANS scatter plot 
plt.clf()
plt.figure(figsize=(8, 6))
plt.scatter(media_matrice1, media_matrice2, label='Means',s=1, alpha=0.05)
plt.title(f'Correation between rowMean EU vs LD _ AB2 embeddings_ corr={corr_pearson}')
plt.xlabel('Mean by row LD distance')
plt.ylabel('Mean by row EU distance ')
slope, intercept, r_value, p_value, std_err = stats.linregress(media_matrice1, media_matrice2)
line = slope * media_matrice1 + intercept
plt.plot(media_matrice1, line, 'r')
plt.legend()
plt.savefig('scatter_plot_rowMEANS.png')



# pairwise EUvsLD correlation by row
correlations = []
for i in range(len(NN_lev)):
    corr, _ = pearsonr(NN_lev[i, :], NN_dist[i, :])
    correlations.append(corr)

#Violin plot 1
plt.clf()
plt.figure(figsize=(10, 6))
sns.violinplot(data=correlations)
plt.title('Distribution of Pearson Correlations Across each id')
plt.ylabel('Pearson Correlation Coefficient')
plt.savefig('violin_correlation_byRow.png')



# % of sequences with shared labels at a given distance- LEVENSTHEIN
num_samples = 60000 # num of sequences to sample
sampled_indices = np.random.choice(df.index, size=num_samples, replace=False)
sampled_indices=id_index_sample #could also use this one , 20k each hb lb mb
#prep array for results
lev_array= np.zeros((num_samples, 15))

#cycle on each sampled index
for row, index in enumerate(sampled_indices):
    
    initial_seq = df.loc[index, 'junction_aa']
    initial_affinity = df.loc[index, 'affinity']
    #compute lev VS WHOLE dataset
    lev_dists = compute_levenshtein(initial_seq, df.iloc[1:]['junction_aa'])
    
    # compute % of seqs with same labels, at a given LEV DIST
    for lev_dist in range(1, 16):
        indices_at_dist = [i for i, x in enumerate(lev_dists) if x == lev_dist]
        if indices_at_dist:
            affinities_at_dist = df.iloc[indices_at_dist]['affinity']
            percentage = np.mean(affinities_at_dist == initial_affinity) * 100
            lev_array[row, lev_dist-1] = percentage
        else:
            lev_array[row, lev_dist-1] = np.nan  # nan= no seqs at that length


#violin plot 2
plt.close()
plt.figure(figsize=(10, 6))
sns.violinplot(data=lev_array)
plt.title(f'% of sequences sharing the same label at each LD, AB2 , sample_size= {num_samples}')
ax = plt.gca()
ax.set_xticklabels(range(1,16))
plt.xlabel('LD distance')
plt.ylabel('% of seqs with same label')
plt.savefig('perc_id_givenLD.png')



# new part, try euclidian distance radius (on pre-computed KNNs)



import numpy as np

distance_thresholds = range(7,20)  # Example thresholds


# Function to calculate the percentage of sequences sharing the same label within distance thresholds
def calculate_percentages_with_precomputed_distances(distances, labels, affinity_labels,lev_mat, distance_thresholds):
    results = []
    LD1_res=[]
    LD2_res=[]
    res_df=pd.DataFrame(columns=[f'EU_{i}' for i in distance_thresholds ])
    # LD1_df=pd.DataFrame(columns=[f'LD1at_EU{i}' for i in distance_thresholds ])
    mean_num_points=[]
    for threshold in distance_thresholds:
        percentages = []
        LD1_list=[]
        LD2_list=[]
        num_of_points_within_rad=[]
        for i in range(distances.shape[0]):  # Iterate over each sequence , shape[0] is the num of points
        # for i in range(10):
        # for i in range(300):
          within_threshold_indices = np.where(distances[i] <= threshold)[0][1:] #1: to skip the vertex NN=0
          num_of_points_within_rad.append( len(within_threshold_indices) )
          # Calculate the percentage of sequences within the threshold distance that share the same label
          if len(within_threshold_indices) !=0:
              ref_label = affinity_labels.iloc[i]
              labels_within_threshold = labels[i, within_threshold_indices]
              # print(within_threshold_indices)
              percentage = np.sum(labels_within_threshold == ref_label) / len(within_threshold_indices)
              percentages.append(percentage)
              # print(f' distance= {threshold} percentage= {percentage}')
              LD1_list.append(np.sum(lev_mat[i,within_threshold_indices]==1) / len(within_threshold_indices))
              LD2_list.append(np.sum(lev_mat[i,within_threshold_indices]==2) / len(within_threshold_indices))
          else:
            percentages.append(np.nan)
            LD1_list.append(np.nan)
            LD2_list.append(np.nan)
            # # Store the average percentage for this threshold
        results.append(np.nanmean(percentages) )
        LD1_res.append(np.nanmean(LD1_list))
        LD2_res.append(np.nanmean(LD2_list))
        mean_num_points.append(np.mean(num_of_points_within_rad))
        res_df[f'EU_{threshold}']=percentages
        # LD1_df[f'LD1at_EU{threshold}']=LD1_list
    return  results, res_df , mean_num_points ,LD1_res, LD2_res



percentages_results_rad_LV, res_df1_LV , mean_num_points_LV ,LD1_perc, LD2_perc = calculate_percentages_with_precomputed_distances(NN_dist, NN_label, ID_labels,NN_lev, distance_thresholds)


for idx,i in enumerate(res_df1_LV.columns):
  null_points=sum(res_df1_LV[i].isna())
  # print(f'{i}:{percentages_results_rad_LV[idx]} ,n_points= {mean_num_points_LV[idx]}, %null={null_points/len(res_df1_LV[i])*100 }, perc_of_LD1= {LD1_perc}')
  print(f'{i}:{percentages_results_rad_LV[idx]} ,n_points= {mean_num_points_LV[idx]}, %null={null_points/len(res_df1_LV[i])*100 }, perc_of_LD1= {LD1_perc[idx]}, perc_of_LD2= {LD2_perc[idx]}')




percentages_results_rad, res_df1 , mean_num_points = calculate_percentages_with_precomputed_distances(NN_dist, NN_label, ID_labels, distance_thresholds)
percentages_results_rad


for idx,i in enumerate(res_df1.columns):
  null_points=sum(res_df1[i].isna())
  print(f'{i}:{percentages_results_rad[idx]} ,n_points= {mean_num_points[idx]}, %null={null_points/len(res_df1[i])*100 }')

len(np.where(NN_dist[0,1:] <= 1)[0])==0

# Assuming NN_dist is an array of distances to the nearest neighbors for each sequence,
# NN_label is an array of labels for each nearest neighbor,
# and ID_labels are the labels for the reference sequences (in the same order as NN_dist and NN_label)
# Here, distance_thresholds is a list of distance thresholds you're interested in
distance_thresholds = [ 5, 10,15,20]  # Example thresholds

# Calculate percentages
percentages_results_rad = calculate_percentages_with_precomputed_distances(NN_dist, NN_label, ID_labels, distance_thresholds)

# Print or return the results
print(percentages_results)




#first try for the moran index


def load_from_pickle(file_name):
    with open(file_name, 'rb') as file:
        data_loaded = pickle.load(file)
    
    fractions_results = data_loaded["fractions_results"]
    NN_id = data_loaded["NN_id"]
    NN_dist = data_loaded["NN_dist"]
    NN_label = data_loaded["NN_label"]
    NN_lev = data_loaded["NN_lev"]
    ID_labels = data_loaded["ID_labels"]
    # Uncomment the following line if "analysis_info" is included in the pickle
    # analysis_info = data_loaded.get("analysis_info")
    
    # Here you can return these values, print them, or process them further as needed
    return fractions_results,NN_id,NN_dist, NN_label,NN_lev,ID_labels


fractions_results,NN_id,NN_dist, NN_label, NN_lev, ID_labels = load_from_pickle("KNN_trastuzumab_AB2_WHOLE.pkl")


len(NN_dist)









n_neighbors = 1000

# Assuming `indices` is an (n_sequences, n_neighbors) array indicating the indices of the 150 NN for each sequence
# and `values` is an array of the values you're analyzing

# Convert distances to weights, example with Euclidean distances
# You can use a similar approach for Levenshtein distances
euclidean_weights_data = {}
# for i in range(len(NN_dist)):   # for each of the 60k points
for i in range((3)):
  for j in range(10):
        neighbor_index = NN_id[i, j] #indices
        # print(neighbor_index)
        weight = 1 / (NN_dist[i, j] + 1e-9)  # Adding a small constant to avoid division by zero
        # print(weight)
        euclidean_weights_data[(i, neighbor_index)] = weight
euclidean_weights_data

euclidean_weights_data = {}
for i in range(3):  #n seqs
    weight_list=[]
    for j in range(10):
        neighbor_index = NN_id[i, j]
        weight_list.append( 1 / (NN_dist[i, j] + 1e-9)  ) # Adding a small constant to avoid division by zero
    euclidean_weights_data[i] = weight_list

euclidean_weights_data

# Creating weights matrix based on nearest neighbors
euclidean_weights_data = {}
for i in range(10):
    neighbors_i = []
    for j in NN_id[i]:
        neighbors_i.append(j)
        # if i not in weights_data.get(j, []):
        #     weights_data.setdefault(j, []).append(i)  # Ensure symmetry
    euclidean_weights_data[i] = neighbors_i


euclidean_weights_data = {}
for i in range(3):  # Assuming '3' sequences or observations
    weights_dict = {}
    for j in range(10):  # Assuming '10' neighbors per observation
        neighbor_index = NN_id[i, j]  # Assuming this fetches the index of the j-th neighbor of i
        # Avoid division by zero and create a dictionary of {neighbor_index: weight}
        weights_dict[neighbor_index] = 1 / (NN_dist[i, j] + 1e-9)  
    euclidean_weights_data[i] = weights_dict




# Create spatial weights object
euclidean_weights = W(euclidean_weights_data)

# Calculate Moran's I for Euclidean distances
mi_euclidean = Moran(NN_label[:3,], euclidean_weights)
print(f"Moran's I (Euclidean): {mi_euclidean.I}, p-value: {mi_euclidean.p_sim}")


# domenica

n_points = NN_dist.shape[0]
weights_data = np.zeros((n_points, n_points))

for i in range(3):
    for j in range(3):
        if i != j:
            # Check if they are mutual nearest neighbors
            if i in NN_dist[j, :150] and j in NN_dist[i, :150]:
                mutual_distance = (NN_dist[i, np.where(NN_dist[j, :150] == i)] + NN_dist[j, np.where(NN_dist[i, :150] == j)]) / 2
                weights_data[i, j] = 1 / mutual_distance

w = W(weights_data)


NN_dist_full = vicinity_analysis_instance.NN_dist
NN_id_full = vicinity_analysis_instance.NN_id


# Assume 'distances' is your n x n matrix of Euclidean distances between sequences
n = NN_dist_full.shape[0]  # Number of sequences




# Convert distances to weights, using an inverse function
# Prevent division by zero and control influence of very close items
inverse_distance_weights = 1 / (NN_dist_full + 1e-9)

inverse_distance_weights = (NN_dist_full <= 14).astype(int)

# Optional: Set a threshold for maximum distance to consider (creates sparsity)
# threshold = np.percentile(distances, 25)  # Only consider closest 25% of distances
inverse_distance_weights[NN_dist_full > 14] = 0

# Create the neighbors and weights dictionaries
neighbors = {i: np.nonzero(NN_id_full[i])[0].tolist() for i in range(2)}

neighbors = {i: NN_id_full[i][inverse_distance_weights[i] > 0].tolist() for i in range(n)}

NN_id[0][True,False]
# neighbors = {i: list(NN_id[i]) for i in range(n)}
weights = {i: inverse_distance_weights[i][inverse_distance_weights[i] != 0].tolist() for i in range(2)}

# Create the W object using the neighbors and weights
w = W(neighbors, weights, silence_warnings=True)

w = W(neighbors)
# Print properties to ensure it's set up correctly
print("Number of observations:", w.n)
print("Percentage of nonzero weights:", "%.3f" % w.pct_nonzero)

label_encoding = {'hb': 1, 'lb': 0 , 'mb':0}
values = np.array([label_encoding[label] for label in vicinity_analysis_instance.NN_label[:,0]])


mi = Moran( values , w)
print(f"Moran's I: {mi.I}, p-value: {mi.p_sim}")





NN_dist_full = vicinity_analysis_instance.NN_dist
NN_id_full = vicinity_analysis_instance.NN_id


n = 




# Convert distances to weights, using an inverse function
# Prevent division by zero and control influence of very close items
inverse_distance_weights = 1 / (NN_dist_full + 1e-9)


#define who are the neighbors -- EU or LD thr
spatial_NN = (NN_dist_full <= 14).astype(int)

spatial_NN[NN_dist_full > 14] = 0

# Create the neighbors and weights dictionaries

neighbors = {i: NN_id_full[i][spatial_NN[i] > 0].tolist() for i in range(n)}
weights = {i: spatial_NN[i][spatial_NN[i] != 0].tolist() for i in range(2)} # first spatial NN should be dist_weights

# Create the W object using the neighbors and weights
w = W(neighbors, weights, silence_warnings=True)

w = W(neighbors)
# Print properties to ensure it's set up correctly
print("Number of observations:", w.n)
print("Percentage of nonzero weights:", "%.3f" % w.pct_nonzero)

label_encoding = {'hb': 1, 'lb': 0 , 'mb':0.5}
values = np.array([label_encoding[label] for label in vicinity_analysis_instance.NN_label[:,0]])


mi = Moran( values , w)
print(f"Moran's I: {mi.I}, p-value: {mi.p_sim}")


def calculate_moran_index(distance_mat, NN_id_mat, label_target, distance_threshold):
    # Define who are the neighbors -- EU or LD threshold
    print("calc spatial matrix")
    spatial_NN = (distance_mat <= distance_threshold).astype(int)
    # spatial_NN[distance_mat > distance_threshold] = 0
    
    # Number of observations
    n = distance_mat.shape[0]
    print("calc NN")
    # Create the neighbors and weights dictionaries
    neighbors = {i: NN_id_mat[i][spatial_NN[i] > 0].tolist() for i in range(n)}
    # weights = {i: spatial_NN[i][spatial_NN[i] > 0].tolist() for i in range(n)}  # Assume all nonzero are weights
    print("calc  weights")
    # Create the W object using the neighbors and weights
    
    # w = W(neighbors, weights, silence_warnings=True)
    w = W(neighbors, weights)
    
    # Encoding labels into numerical values
    label_encoder = LabelEncoder()
    labels = label_target
    values = label_encoder.fit_transform(labels)
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print("Label to number mapping:", label_mapping)
    print("Calculate Moran's I")
    mi = Moran(values, w)
    
    # Print properties to ensure it's set up correctly
    print("Number of observations:", w.n)
    print("Percentage of nonzero weights:", "%.3f" % w.pct_nonzero)
    print(f"Moran's I: {mi.I}, p-value: {mi.p_sim}")
    return mi.I, mi.p_sim


I, p_value = calculate_moran_index(NN_dist_full, NN_id_full, vicinity_analysis_instance.NN_label[:,0], distance_threshold=14)
# Example usage, assuming you have distance_mat, NN_id_full, and vicinity_analysis_instance prepared

data=[]
chosen_dist= "ED"
for i in range(8,20):
  I, p_value = calculate_moran_index(NN_dist_full, NN_id_full, vicinity_analysis_instance.NN_label[:,0], distance_threshold=i)
  data.append([I, p_value,i,chosen_dist])


moran_df_ED=pd.DataFrame(data,columns=['MoranI', 'pval', 'dist_thr'])

data=[]
chosen_dist= "LD"
for i in range(5,10): #should be 15 but we simplyfiy it , first run 1-5 then 5-10
  I, p_value = calculate_moran_index(vicinity_analysis_instance.NN_lev, NN_id_full, vicinity_analysis_instance.NN_label[:,0], distance_threshold=i)
  data.append([I, p_value,i,chosen_dist])

moran_df_LD=pd.DataFrame(data,columns=['MoranI', 'pval', 'dist_thr','chosen_dist'])

moran_df_LD1=pd.concat((moran_df_LD,pd.DataFrame(data,columns=['MoranI', 'pval', 'dist_thr','chosen_dist'])))






moran_df_LD['dist_thr2']= [13,15,17,20]


percentages_results, res_df, mean_num_points, LD1_res, LD2_res = vicinity_analysis_instance.calculate_percentages_with_precomputed_distances(range(8, 30) )

percent_null_eu = [ 96.8783,88.0183,70.4883,47.9550,28.5350,15.4433,8.1917,4.3367,2.1683,1.0167,0.4317,0.1500]

percent_NON_null_eu =[(100 -x )/100 for x in percent_null_eu]
percent_NON_null_ld =[(100 -x )/100 for x in [6,1.2,0.2,0.1]]


moran_df_ED['MoranI_norm']=  moran_df_ED['MoranI'] / percent_NON_null_eu
moran_df_LD['MoranI_norm']=  moran_df_LD['MoranI'] / percent_NON_null_ld



# Plotting

plt.figure(figsize=(10, 6))
plt.plot(moran_df_ED['dist_thr'], moran_df_ED['MoranI'], marker='o', color='blue', label='ED')
plt.plot(moran_df_LD['dist_thr'], moran_df_LD['MoranI'], marker='o', color='red', label='LD')

plt.plot(moran_df_ED['dist_thr'], moran_df_ED['MoranI_norm'], marker='o', color='blue', label='ED')
plt.plot(moran_df_LD['dist_thr'], moran_df_LD['MoranI_norm'], marker='o', color='red', label='LD')

plt.figure(figsize=(10, 6))
plt.plot(percent_null_eu, moran_df_ED['MoranI'], marker='o', color='blue', label='ED')
plt.plot([6,1.2,0.2,0.1,0.01,0,0,0,0], moran_df_LD1['MoranI'], marker='o', color='red', label='LD')


# Annotate each point for ED
for i, txt in enumerate(percent_null_eu):
    plt.text(percent_null_eu[i]+2, moran_df_ED['MoranI'][i], str(moran_df_ED['dist_thr'].iloc[i]), fontsize=8)

ld_null_perc=[6,1.2,0.2,0.1,0.01,0,0,0,0]
# Annotate each point for LD, adjust your data accordingly
for i, txt in enumerate(ld_null_perc):
    print(i)
    plt.text(ld_null_perc[i]-2, moran_df_LD1['MoranI'].iloc[i], str(moran_df_LD1['dist_thr'].iloc[i]), fontsize=8)


# Adding details to the plot
plt.xlabel('perc of null points')
plt.ylabel("Moran's I")
plt.title("Moran's I vs. Null points")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'MoranI_4.png')
plt.close()


#try unbalanced dataset  -Marina's code :)

sample_size = 2000

df_hb_50percent = df_hb.sample(n=min(len(df_hb), len(df_lb)), random_state=42)
df_lb_50percent = df_lb.sample(n=min(len(df_hb), len(df_lb)), random_state=42)

df_hblb_50percent = pd.concat([df_hb_50percent, df_lb_50percent], ignore_index=True)
# df_hblb_50percent['id'] = range(0, len(df_hblb_50percent) )


def create_subset(sample_size, hb_percentage, df_hblb_50percent):
    df_new = pd.concat([
        df_hblb_50percent.query("affinity == 'hb'").sample(int(sample_size * hb_percentage)),
        df_hblb_50percent.query("affinity == 'lb'").sample(int(sample_size * (1-hb_percentage)))
    ])
    df_new.reset_index(drop=True, inplace=True)  #important! 
    # TODO check discrepancies in the use of .loc and .iloc in Vicinity_class -- this causes problems with indexes
    df_new['id'] = range(0, len(df_new) )
  
    return df_new


unbal_df=create_subset(100000,0.1,df_hblb_50percent)
unbal_df['affinity']


unbal_df.iloc[46392] == unbal_df.loc[46392]  # must be true

max_neighbors = 1000 # This is the maximum number of neighbors you're interested in
part1 = np.arange(2, 304, 4)  # check in detail first 300 NN
part2 = np.arange(350, max_neighbors+1, 50)  # Second part: numbers from 300 to 1000 with steps of 50
neighbor_numbers = np.concatenate((part1, part2))

id_index_sample= df_hblbmb_5000['id'] # indexes of 20k each lb mb hb

#runnin on the sampled 60k vertex, whole dataset
# fractions_results,NN_id,NN_dist, NN_label, NN_lev, ID_labels = calculate_fractions_for_data(df, neighbor_numbers, id_index_sample)

#runnin on the WHOLE dataset
vicinity_analysis_unbal = Vicinity_analysis(unbal_df, neighbor_numbers, unbal_df['id'])
vicinity_analysis_unbal.run_analysis()  # This populates the necessary attributes
# to save -->      vicinity_analysis_instance.save_to_pickle("Vicinity_Tz_AB2_WHOLE.pkl")
# to load -->      ex1=Vicinity_analysis.load_from_pickle("Vicinity_Tz_AB2_WHOLE.pkl")
# SUGGESTED TO LOAD THE ALREADY COMPUTED FILE
ED_radius = range(7, 60)  # Define your Euclidian distance radius to check
percentages_results, res_df, mean_num_points, LD1_res, LD2_res = ex1.perc_Euclidian_radius(ED_radius)



# -----Euclidian distance radius method ------
ED_radius = range(7, 60)  # Define your Euclidian distance radius to check
percentages_results, res_df, mean_num_points, LD1_res, LD2_res = vicinity_analysis_unbal.perc_Euclidian_radius(ED_radius)


""" Moran index evaluation"""

from esda import Moran_Local
from libpysal.weights import W
y = df['numeric'].values
moran_local = Moran_Local(y, w, permutations=999)

NN_dist_unbal = vicinity_analysis_unbal.NN_dist  # NN_dist_unbal = ex1.NN_dist
NN_id_unbal = vicinity_analysis_unbal.NN_id
NN_label_unbal = vicinity_analysis_unbal.NN_label
NN_lev_unbal =vicinity_analysis_instance.NN_lev
 
x_spatial_NN = (NN_dist_unbal <= 14).astype(int)
w_loc = W(neighbors)
weights = {i: NN_dist_unbal[i][x_spatial_NN[i] > 0].tolist() for i in range(10)}



data=[]
chosen_dist= "ED"
for i in range(8,20):
  I, p_value, pct_nonzerow = calculate_moran_index(NN_dist_unbal, NN_id_unbal, NN_label_unbal[:,0], distance_threshold=i, weight_distance=True)
  data.append([I, p_value,i,chosen_dist])
  
moran_df_ED=pd.DataFrame(data,columns=['MoranI', 'pval', 'dist_thr','chosen_dist'])

data=[]
chosen_dist= "LD"
for i in range(1,10): #should be 15 but we simplyfiy it
  I, p_value ,pct_nonzerow = calculate_moran_index(NN_lev_unbal, NN_id_unbal, NN_label_unbal[:,0], distance_threshold=i, weight_distance=True)
  data.append([I, p_value,i,chosen_dist])
  
  
moran_df_LD=pd.DataFrame(data,columns=['MoranI', 'pval', 'dist_thr','chosen_dist'])
# pd.concat((moran_df_LD,pd.DataFrame(data,columns=['MoranI', 'pval', 'dist_thr','chosen_dist'])))




### gpttttt codeee


def run_vicinity_analysis(df, neighbor_numbers):
    vicinity_analysis = Vicinity_analysis(df, neighbor_numbers, df['id'])
    vicinity_analysis.run_analysis()
    return vicinity_analysis



# List of different hb percentages to analyze
hb_percentages = [0.1, 0.3, 0.5, 0.7, 0.9]

# Load the initial combined DataFrame
# df_hblb_50percent = load_your_dataframe_here()
def save_to_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_from_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)        
        
def save_to_csv(data, filename):
    data.to_csv(filename, index=False)



for hb_percentage in hb_percentages:
    unbal_df=create_subset(100000,hb_percentage,df_hblb_50percent)
    vicinity_result = run_vicinity_analysis(unbal_df, neighbor_numbers)
    percentages_results, res_df, mean_num_points, LD1_res, LD2_res = vicinity_result.perc_Euclidian_radius(range(7, 25))
    
    # Save the vicinity analysis instance
    vicinity_result.save_to_pickle(f'vicinity_analysis_unbal_{int(hb_percentage * 100)}.pkl')
    save_to_pickle(vicinity_result.summary_results,f'vicinity_radius_summary_unbal_{int(hb_percentage * 100)}.pkl')
    
    # Calculate and save Moran's indices for Euclidean distance (ED) and Lev distance (LD)
    moran_data_ED = []
    moran_data_LD = []
    for i in range(8, 20):  # Example range for Euclidean
        I, p_value, _ = calculate_moran_index(vicinity_result.NN_dist, vicinity_result.NN_id, vicinity_result.NN_label[:,0], i, weight_distance=True)
        moran_data_ED.append([I, p_value, i, "ED"])
    for i in range(1, 10):  # Example range for Lev
        I, p_value ,_ = calculate_moran_index(vicinity_result.NN_lev, vicinity_result.NN_id, vicinity_result.NN_label[:,0], i, weight_distance=True)
        moran_data_LD.append([I, p_value, i, "LD"])
    
    moran_df_ED = pd.DataFrame(moran_data_ED, columns=['MoranI', 'pval', 'dist_thr', 'chosen_dist'])
    moran_df_LD = pd.DataFrame(moran_data_LD, columns=['MoranI', 'pval', 'dist_thr', 'chosen_dist'])
    
    moran_df_ED.to_csv( f'moran_results_ED_unbal_{int(hb_percentage * 100)}.csv' ,index=False)
    moran_df_LD.to_csv( f'moran_results_LD_unbal_{int(hb_percentage * 100)}.csv' , index=False)

# Example of how to load one of the analyses
# loaded_analysis = load_from_pickle('vicinity_analysis_hb_10.pkl')


moran_res_df= pd.DataFrame()
for hb_percentage in hb_percentages:
  tmp_df = pd.concat( 
  (pd.read_csv( f'moran_results_ED_unbal_{int(hb_percentage * 100)}.csv'),
   pd.read_csv( f'moran_results_LD_unbal_{int(hb_percentage * 100)}.csv') ) )
  tmp_df['hb_perc']=hb_percentage
  moran_res_df= pd.concat((moran_res_df,tmp_df))

load_from_pickle(f'vicinity_radius_summary_unbal_{int(hb_percentage * 100)}.pkl')



plt.figure(figsize=(10, 6))
plt.plot(moran_res_df['dist_thr'], moran_df_ED['MoranI'], marker='o', color=[0,1], label=moran_res_df['moran_res_df['dist_thr']'])

# plt.plot(moran_res_df['dist_thr'], moran_df_ED['MoranI'], marker='o', color=[0,1], label=moran_res_df['moran_res_df['dist_thr']'])


# Adding details to the plot
plt.xlabel('Distance Thr')
plt.ylabel("Moran's I")
plt.title(f"Moran's I vs. distance --ubal hb perc {hb_perc}")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.savefig(f'MoranI__unbal.png')
plt.savefig(f'MoranI__unbal_09_unfinished.png')

plt.close()


# Create a figure and subplots
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 10))  # Adjust nrows and ncols based on your actual needs

# Flatten axes array if more than one row and column
axes = axes.flatten()

# Loop over the columns of the dataframe and create a line plot for each
for i, hb_perc in enumerate(hb_percentages):
    tmp_df=moran_res_df[moran_res_df['hb_perc']==hb_perc]
    tmp_ED = tmp_df[tmp_df['chosen_dist']=="ED"]
    tmp_LD = tmp_df[tmp_df['chosen_dist']=="LD"]
    axes[i].plot(tmp_ED['dist_thr'], tmp_ED['MoranI'], marker='o', color='blue', label=tmp_ED['dist_thr'])
    axes[i].plot(tmp_LD['dist_thr'], tmp_LD['MoranI'], marker='o', color='red', label=tmp_LD['dist_thr'])
    axes[i].set_title(f"Moran's I vs. distance --ubal hb perc {hb_perc}")
    axes[i].set_xlabel('Distance Thr')
    axes[i].set_ylabel('Moran s I')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.savefig(f'MoranI_unbal.png')
plt.close()





#preliminary plot  --- dati a metà


tmp_df1=moran_res_df[moran_res_df['hb_perc']==0.9]
tmp_ED1 = tmp_df[tmp_df['chosen_dist']=="ED"]
tmp_LD1 = tmp_df[tmp_df['chosen_dist']=="LD"]

tmp_ed_sum=load_from_pickle(f'vicinity_radius_summary_unbal_90.pkl')
hb_nan=d_sum.iloc[0,[1,3]].tolist()  #hb
hb_nan.extend([ 0,0,0,0,0,0,0])
hb_nan=np.array(hb_nan)*100

lb_nan=d_sum.iloc[1,[1,3]].tolist()  #lb
lb_nan.extend([ 0,0,0,0,0,0,0])
lb_nan=np.array(lb_nan)*100

  
plt.figure(figsize=(10, 6))
plt.plot(tmp_ed_sum['Percentage_Null'][1:13], moran_df_ED['MoranI'], marker='o', color='blue', label='ED_hb09')
plt.plot(hb_nan, moran_df_LD['MoranI'], marker='o', color='red', label='LD_hb09')
plt.plot(lb_nan, moran_df_LD['MoranI'], marker='o', color='green', label='LD_hb09')

# Adding details to the plot
plt.xlabel('Distance Thr')
plt.ylabel("Moran's I")
plt.title(f"Moran's I vs. distance --ubal hb perc {hb_perc}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'MoranI__unbal_{}.png')
plt.close()





def prepare_data_for_plotting(df, LD_dist , sampled_indices= None):
    num_samples = len(sampled_indices)
    if sampled_indices is None:
      sampled_indices =df['id']
      num_samples = len(sampled_indices)
    # sampled_indices = np.random.choice(df.index, size=num_samples, replace=False)
    
    # Inizializzazione degli array per i risultati
    results = np.zeros((num_samples, LD_dist))
    num_of_points = np.zeros((num_samples, LD_dist))
    affinities = []
    
    # Iterazione su ciascun indice campionato
    for row, index in tqdm(enumerate(sampled_indices), total=num_samples, desc="Processing samples"):
        initial_affinity = df.loc[index, 'affinity']
        initial_seq = df.loc[index, 'junction_aa']
        affinities.append(initial_affinity)
        
        # Calcolo delle distanze di Levenshtein
        lev_dists = compute_levenshtein(initial_seq, df.iloc[1:]['junction_aa'])
        for lev_dist in range(1, LD_dist + 1):
            indices_at_dist = [i for i, x in enumerate(lev_dists) if x == lev_dist]
            if indices_at_dist:
                affinities_at_dist = df.iloc[indices_at_dist]['affinity']
                percentage = sum(affinities_at_dist == initial_affinity) / len(affinities_at_dist)
                results[row, lev_dist - 1] = percentage
                num_of_points[row, lev_dist - 1] = len(affinities_at_dist)
            else:
                results[row, lev_dist - 1] = np.nan  # Uso NaN per le distanze senza sequenze
                num_of_points[row, lev_dist - 1] = 0
                
    # Combine results and num_of_points into a single DataFrame
    columns = [f'LD_{i}' for i in range(1, LD_dist + 1)]
    df_results = pd.DataFrame(results, columns=[f'Perc_{col}' for col in columns]) # PERCENTAGE OF Points with SAME LABLE (vicinity score)
    df_num_points = pd.DataFrame(num_of_points, columns=[f'Num_{col}' for col in columns])
    
    # Merge into one DataFrame
    df_combined = pd.concat([df_results, df_num_points], axis=1)
    df_combined['sample_id'] = sampled_indices
    df_combined['affinity'] = affinities
    
    # Combine results and num_of_points into a single DataFrame
    columns = [f'LD_{i}' for i in range(1, LD_dist + 1)]
    df_combined = pd.DataFrame({
        **{f'Perc_{col}': results[:, idx] for idx, col in enumerate(columns)},
        **{f'Num_{col}': num_of_points[:, idx] for idx, col in enumerate(columns)},
        'sample_id': sampled_indices,
        'affinity': affinities
    })
    
    # Calculate summary statistics
    df_summary = pd.DataFrame()
    for col in columns:
        df_combined[f'NaN_Count_{col}'] = df_combined[f'Perc_{col}'].isna()
        summary_stats = df_combined.groupby('affinity')[[f'Num_{col}', f'NaN_Count_{col}']].agg({
            f'Num_{col}': 'mean',
            f'NaN_Count_{col}': 'mean'
        }).rename(columns={f'Num_{col}': f'Avg_Num_{col}', f'NaN_Count_{col}': f'Avg_NaN_Percentage_{col}'})
        df_summary = pd.concat([df_summary, summary_stats], axis=1)
        
    return df_combined, df_summary 
    
    

 
def generate_plots(df_combined):
    # Melt the DataFrame for plotting
    df_melted = df_combined.melt(id_vars=['sample_id', 'affinity'], var_name='LD_type', value_name='Value')
    
    # Separate LD and Type (Perc or Num)
    df_melted['LD'] = df_melted['LD_type'].str.extract('(\d+)')
    df_melted['Type'] = df_melted['LD_type'].str.extract('([A-Za-z]+)')
    
    # Plotting
    unique_affinities = df_melted['affinity'].unique()
    for affinity in unique_affinities:
        df_affinity = df_melted[df_melted['affinity'] == affinity]
        # Adjusting g for LD and Type without col_wrap
        g = sns.FacetGrid(df_affinity, col='LD', row='Type', sharex=False, sharey=False)
        g.map_dataframe(sns.scatterplot, 'LD_type', 'Value', alpha=0.1, s=2)
        g.set_titles('{col_name} - {row_name}')
        g.set_axis_labels("LD Type", "Value")
        plt.subplots_adjust(top=0.9)
        plt.tight_layout()
        plt.savefig(f'FACET_perc_id_givenLD_{affinity}_n{60000}_200_with_LD_averages_NaNs_percentage22.png')
        plt.close()       
        
# Note: Compute_levenshtein function should be defined or imported before using the prepare_data_for_plotting.


unbal_LD_null_list=[]
LD_dist=5
for i in hb_percentages:
  unbal_df=create_subset(100000,i,df_hblb_50percent)
  d_res1,d_mean1 =prepare_data_for_plotting( unbal_df,LD_dist, sampled_indices=unbal_df['id'])
  unbal_LD_null_list.append([d_res1,d_mean1] )


unbal_LD_null_list[0][0]
unbal_LD_null_list[0][1]

d_comb, d_sum =prepare_data_for_plotting( unbal_df,2, sampled_indices=unbal_df['id'])

generate_plots(d_comb)



#preliminary plot  --- dati a metà

for idx,hb_perc in enumerate(hb_percentages):
  tmp_df1=moran_res_df[moran_res_df['hb_perc']==hb_perc]
  tmp_ED1 = tmp_df[tmp_df['chosen_dist']=="ED"]
  tmp_LD1 = tmp_df[tmp_df['chosen_dist']=="LD"]
  
  tmp_ed_sum=load_from_pickle(f'vicinity_radius_summary_unbal_{int(hb_perc*100)}.pkl')
  hb_nan=unbal_LD_null_list[idx][1].iloc[0,[1,3,5,7,9]].tolist()  #hb
  hb_nan.extend([ 0,0,0,0])
  hb_nan=np.array(hb_nan)*100
  
  lb_nan=unbal_LD_null_list[idx][1].iloc[1,[1,3,5,7,9]].tolist()  #lb
  lb_nan.extend([ 0,0,0,0])
  lb_nan=np.array(lb_nan)*100
  
    
  plt.figure(figsize=(10, 6))
  plt.plot(tmp_ed_sum['Percentage_Null'][1:13], moran_df_ED['MoranI'], marker='o', color='blue', label=f'ED_{hb_perc}')
  plt.plot(hb_nan, moran_df_LD['MoranI'], marker='o', color='red', label=f'LD_hb_{hb_perc}')
  plt.plot(lb_nan, moran_df_LD['MoranI'], marker='o', color='green', label=f'LD_lb_{hb_perc}')
  
  # Adding details to the plot
  plt.xlabel('Perc of null points')
  plt.ylabel("Moran's I")
  plt.title(f"Moran's I vs. distance --ubal hb perc {hb_perc}")
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.savefig(f'MoranI__unbal_{hb_perc}.png')
  plt.close()



for idx, hb_perc in enumerate(hb_percentages):
    tmp_df1 = moran_res_df[moran_res_df['hb_perc'] == hb_perc]
    tmp_ED1 = tmp_df1[tmp_df1['chosen_dist'] == "ED"]
    tmp_LD1 = tmp_df1[tmp_df1['chosen_dist'] == "LD"]
    
    tmp_ed_sum = load_from_pickle(f'vicinity_radius_summary_unbal_{int(hb_perc * 100)}.pkl')
 
    hb_nan = unbal_LD_null_list[idx][1].iloc[0, [1, 3, 5, 7, 9]].tolist()  # hb
    hb_nan.extend([0, 0, 0, 0])
    hb_nan = np.array(hb_nan) * 100
  
    lb_nan = unbal_LD_null_list[idx][1].iloc[1, [1, 3, 5, 7, 9]].tolist()  # lb
    lb_nan.extend([0, 0, 0, 0])
    lb_nan = np.array(lb_nan) * 100
    
    # Set up the subplot grid
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    # Plotting the first subplot
    ax1.plot(tmp_ed_sum['Percentage_Null'][1:13], tmp_ED1['MoranI'], marker='o', color='blue', label=f'ED_{hb_perc}')
    ax1.plot(hb_nan, tmp_LD1['MoranI'], marker='o', color='red', label=f'LD_hb_{hb_perc}')
    ax1.plot(lb_nan, tmp_LD1['MoranI'], marker='o', color='green', label=f'LD_lb_{hb_perc}')
    
    ax1.set_xlabel('Percentage of Null Points')
    ax1.set_ylabel("Moran's I")
    ax1.set_title(f"Moran's I vs. Null Percentage - Unbalanced HB Perc {hb_perc}")
    ax1.legend()
    ax1.grid(True)
    
    hb_avg = unbal_LD_null_list[idx][1].iloc[0, [0,2,4,6,8]].tolist()  # hb
    # hb_avg.extend([0, 0, 0, 0])
    hb_avg = np.array(hb_avg) * 100
  
    lb_avg = unbal_LD_null_list[idx][1].iloc[1, [0,2,4,6,8]].tolist()  # lb
    # lb_avg.extend([0, 0, 0, 0])
    lb_avg = np.array(lb_avg) * 100
    print(lb_avg)
    # Plotting the second subplot
    ax2.plot(tmp_ed_sum['Mean_Num_Points'][1:13], tmp_ED1['MoranI'], marker='o', color='blue', label=f'ED_{hb_perc}')
    ax2.plot(hb_avg, tmp_LD1['MoranI'][:5], marker='o', color='red', label=f'LD_hb_{hb_perc}')
    ax2.plot(lb_avg, tmp_LD1['MoranI'][:5], marker='o', color='green', label=f'LD_lb_{hb_perc}')
    
    ax2.set_xlabel('Mean Number of Points')
    ax2.set_ylabel("Moran's I")
    ax2.set_title(f"Moran's I vs. Mean Number of Points - Unbalanced HB Perc {hb_perc}")
    ax2.set_xlim([-10,1500] )  # Set x-axis limits for the second plot
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f'MoranI_combined_unbal_{hb_perc}.png')
    plt.close()
    

##

tmp_df1 = moran_res_df[moran_res_df['hb_perc'] == hb_perc]
tmp_ED1 = tmp_df1[tmp_df1['chosen_dist'] == "ED"]
tmp_LD1 = tmp_df1[tmp_df1['chosen_dist'] == "LD"]

tmp_ed_sum = load_from_pickle(f'vicinity_radius_summary_unbal_{int(hb_perc * 100)}.pkl')
 
hb_nan = unbal_LD_null_list[idx][1].iloc[0, [1, 3, 5, 7, 9]].tolist()  # hb
hb_nan.extend([0, 0, 0, 0])
hb_nan = np.array(hb_nan) * 100
  
lb_nan = unbal_LD_null_list[idx][1].iloc[1, [1, 3, 5, 7, 9]].tolist()  # lb
lb_nan.extend([0, 0, 0, 0])
lb_nan = np.array(lb_nan) * 100

# Set up the subplot grid
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

# Plotting the first subplot
ax1.plot(tmp_ed_sum['Percentage_Null'][1:13], tmp_ED1['MoranI'], marker='o', color='blue', label=f'ED_{hb_perc}')
ax1.plot(hb_nan, tmp_LD1['MoranI'], marker='o', color='red', label=f'LD_hb_{hb_perc}')
ax1.plot(lb_nan, tmp_LD1['MoranI'], marker='o', color='green', label=f'LD_lb_{hb_perc}')

ax1.set_xlabel('Percentage of Null Points')
ax1.set_ylabel("Moran's I")
ax1.set_title(f"Moran's I vs. Null Percentage - Unbalanced HB Perc {hb_perc}")
ax1.legend()
ax1.grid(True)

hb_avg = unbal_LD_null_list[idx][1].iloc[0, [0,2,4,6,8]].tolist()  # hb
# hb_avg.extend([0, 0, 0, 0])
hb_avg = np.array(hb_avg) * 100
  
lb_avg = unbal_LD_null_list[idx][1].iloc[1, [0,2,4,6,8]].tolist()  # lb
# lb_avg.extend([0, 0, 0, 0])
lb_avg = np.array(lb_avg) * 100
print(lb_avg)
# Plotting the second subplot
ax2.plot(tmp_ed_sum['Mean_Num_Points'][1:13], tmp_ED1['MoranI'], marker='o', color='blue', label=f'ED_{hb_perc}')
ax2.plot(hb_avg, tmp_LD1['MoranI'][:5], marker='o', color='red', label=f'LD_hb_{hb_perc}')
ax2.plot(lb_avg, tmp_LD1['MoranI'][:5], marker='o', color='green', label=f'LD_lb_{hb_perc}')

ax2.set_xlabel('Mean Number of Points')
ax2.set_ylabel("Moran's I")
ax2.set_title(f"Moran's I vs. Mean Number of Points - Unbalanced HB Perc {hb_perc}")
ax2.legend()
ax2.grid(True)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(f'MoranI_combined_unbal_{hb_perc}.png')
plt.close()



#calculate the statistics
d_comb, d_sum =prepare_data_for_plotting( unbal_df,5, sampled_indices=unbal_df['id'])



  




#------------------------simulated try 

# Set random seed for reproducibility
np.random.seed(42)

# Number of sequences and nearest neighbors
n = 100
nn = 10

# Simulate Euclidean distances for nearest neighbors
NN_dist_sim = np.random.rand(n, nn) * 20  # Distances between 0 and 20
NN_id_sim = np.random.randint(1, 100, size=(n, nn))  # Random IDs between 1 and 1000

# Simulate labels (using 'hb' and 'lb' as categories)
labels = np.random.choice(['hb', 'lb'], size=n)
NN_label_sim = np.random.choice(['hb', 'lb'], size=(n, nn))

# Assume distances to self are the smallest and should not be considered in weights
NN_dist_sim[:, 0] = 0  # setting distances to self as the smallest, unrealistic in actual data but for consistency with your example

# Filter only 'hb' labels and corresponding distances and IDs
hb_indices = np.where(labels == 'hb')[0]
filtered_NN_dist_sim = NN_dist_sim[hb_indices]
filtered_NN_id_sim = NN_id_sim[hb_indices]

# Convert distances to weights with a threshold, only considering distances <= 8
filtered_inverse_distance_weights = (filtered_NN_dist_sim <= 8).astype(int)

# Creating weights and neighbors dictionary for 'hb' labeled points
neighbors = {i: filtered_NN_id_sim[i][filtered_inverse_distance_weights[i] > 0].tolist() for i in range(filtered_NN_dist_sim.shape[0])}
w = W(neighbors)

#gpt correction -------------

# Adjust neighbor IDs to be consistent with filtered indices
# Create a new index map for filtered IDs
index_map = {old_index: new_index for new_index, old_index in enumerate(hb_indices)}
neighbors = {}
for i in range(filtered_NN_dist_sim.shape[0]):
    valid_neighbors = filtered_NN_id_sim[i][filtered_inverse_distance_weights[i] > 0]
    # Remap neighbors to the new indexing system
    neighbors[index_map[hb_indices[i]]] = [index_map[n] for n in valid_neighbors if n in index_map]

# Create the W object using the remapped neighbors
w = W(neighbors)





# All 'hb' labels are encoded as 1, no need for a dictionary since we only consider 'hb'
values = np.ones(len(hb_indices))

# Calculate Moran's I for 'hb' labeled points
mi = Moran(values, w)
print(f"Moran's I for 'hb' labels: {mi.I}, p-value: {mi.p_sim}")



# sim   snjjcsdnvkpèdvjvk
n = 100
nn = 10
NN_dist_sim = np.random.rand(n, nn) * 20
NN_id_sim = np.random.randint(1, 100, size=(n, nn))  # Ensure IDs range from 1 to n
labels = np.random.choice(['hb', 'lb'], size=n)
# Adjustments: Convert 'hb' and 'lb' to numeric
labels_numeric = pd.get_dummies(labels)
hb_labels = labels_numeric['hb'].values  # 'hb' as 1, 'lb' as 0
lb_labels = labels_numeric['lb'].values  # 'lb' as 1, 'hb' as 0

# Create a weights matrix using IDs and distances, manually building weights
W = lps.weights.W(neighbors={})
for i in range(n):
    neighbors = {}
    for j in range(nn):
        if NN_dist_sim[i, j] != 0:  # exclude self-distance, which is 0
            neighbor_id = NN_id_sim[i, j] - 1  # Python is 0-indexed
            if neighbor_id not in neighbors:
                neighbors[neighbor_id] = 1 / NN_dist_sim[i, j]
            else:
                neighbors[neighbor_id] += 1 / NN_dist_sim[i, j]
    W.neighbors[i] = list(neighbors.keys())
    W.weights[i] = list(neighbors.values())

# Normalize the weights
W.transform = 'R'

# Calculate Moran's I for 'hb'
moran_hb = Moran(hb_labels, W)
print("Moran's I for hb:", moran_hb.I, "with p-value:", moran_hb.p_sim)

# Calculate Moran's I for 'lb'
moran_lb = Moran(lb_labels, W)
print("Moran's I for lb:", moran_lb.I, "with p-value:", moran_lb.p_sim





# Number of sequences and nearest neighbors
n = 100
nn = 10

# Simulate Euclidean distances for nearest neighbors
NN_dist_sim = np.random.rand(n, nn) * 20  # Distances between 0 and 20
NN_id_sim = np.random.randint(1, 100, size=(n, nn))  # Random IDs between 1 and 1000

# Simulate labels (using 'hb' and 'lb' as categories)
labels = np.random.choice(['hb', 'lb'], size=n)
NN_label_sim = np.random.choice(['hb', 'lb'], size=(n, nn))

# Assume distances to self are the smallest and should not be considered in weights
NN_dist_sim[:, 0] = 0  # setting distances to self as the smallest, unrealistic in actual data but for consistency with your example


# Convert distances to weights
# inverse_distance_weights = 1 / (NN_dist_sim + 1e-9)  # add small value to avoid division by zero
# inverse_distance_weights[NN_dist_sim > 14] = 0  # apply threshold for maximum distance

inverse_distance_weights = (NN_dist_sim <= 14).astype(int)

# Creating weights and neighbors dictionary
neighbors = {i: NN_id_sim[i][inverse_distance_weights[i] > 0].tolist() for i in range(n)}
weights = {i: inverse_distance_weights[i][inverse_distance_weights[i] > 0].tolist() for i in range(n)}

# Create the W object
w = W(neighbors)

# Print properties to check setup
print("Number of observations:", w.n)
print("Percentage of nonzero weights:", "%.3f" % w.pct_nonzero)

# Encode labels as integers for Moran's I calculation
label_encoding = {'hb': 1, 'lb': 0}
values = np.array([label_encoding[label] for label in labels])

# Calculate Moran's I
mi = Moran(values[:10], w)
print(f"Moran's I: {mi.I}, p-value: {mi.p_sim}")


# Calculate Moran's LOCAL
moran_local = Moran_Local(values, w, permutations=999)
# Accessing Moran's Local results
local_Is = moran_local.Is
p_values = moran_local.p_sim


local_Is[values == 0].mean() # hb
lb_Is = df[df['point_type'] == 'lb']['local_Is']

print(f"Moran's I: {mi.I}, p-value: {mi.p_sim}")









