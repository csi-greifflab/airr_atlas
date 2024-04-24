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
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.stats import mannwhitneyu
from scipy.spatial.distance import jensenshannon
import concurrent.futures
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
import pickle
import sys
sys.path.insert(0, '/doctorai/niccoloc/airr_atlas')

from Vicinity_analysis_class import Vicinity_analysis
from Vicinity_analysis_class import calculate_moran_index
from libpysal.weights import W



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
id_index_mb= df_hblbmb_5000['id'] # indexes of 20k each lb mb hb

"""  ** Run Vicinity analysis ** """

max_neighbors = 1000 # This is the maximum number of neighbors you're interested in
part1 = np.arange(2, 304, 4)  # check in detail first 300 NN
part2 = np.arange(350, max_neighbors+1, 50)  # Second part: numbers from 300 to 1000 with steps of 50
neighbor_numbers = np.concatenate((part1, part2))

id_index_sample= df_hblbmb_5000['id'] # indexes of 20k each lb mb hb

#runnin on the sampled 60k vertex, whole dataset
# fractions_results,NN_id,NN_dist, NN_label, NN_lev, ID_labels = calculate_fractions_for_data(df, neighbor_numbers, id_index_sample)

#runnin on the WHOLE dataset
vicinity_analysis_instance = Vicinity_analysis(df, neighbor_numbers, df['id'])
vicinity_analysis_instance.run_analysis()  # This populates the necessary attributes

ED_radius = range(7, 25)  # Define your Euclidian distance radius to check
percentages_results, res_df, mean_num_points, LD1_res, LD2_res = vicinity_analysis_instance.perc_Euclidian_radius(ED_radius)

# to save -->      vicinity_analysis_instance.save_to_pickle("Vicinity_Tz_AB2_WHOLE.pkl")
# to load -->      ex1=Vicinity_analysis.load_from_pickle("Vicinity_Tz_AB2_WHOLE.pkl")
# SUGGESTED TO LOAD THE ALREADY COMPUTED FILE ( Tz dataset)
# vicinity_analysis_instance=ex1

np.random.seed(123)
rand_100k = np.random.choice(df.index, size=10000, replace=False)
# VERY LONG TO RUN - 6 h FOR LD= 1,2,3,4,5
# TODO ---- Please parallelize this function, it's very slow, at least run each LD threshold on a different core
max_LD=5
d_res1,d_mean1 =prepare_data_for_plotting( df,max_LD, sampled_indices=rand_100k) # to get VICINITY percentages of LD dist
# TODO --- also should save or integrate into the vicinity class this result,
# given that it's the most computationally intensive step of all the analysis ( for i in sampled_indices --> LD calculation  i vs ALL)


""" Moran index evaluation"""
NN_dist_full = vicinity_analysis_instance.NN_dist
NN_id_full = vicinity_analysis_instance.NN_id
NN_label_full = vicinity_analysis_instance.NN_label
NN_lev_full =vicinity_analysis_instance.NN_lev

# compute global Moran I
data=[]
chosen_dist= "ED"
for i in range(25): # ED ranges
  I, p_value, pct_nonzerow = calculate_moran_index(NN_dist_full, NN_id_full, NN_label_full[:,0], distance_threshold=i, weight_distance=True)
  data.append([I, p_value,i,chosen_dist])

moran_df_ED=pd.DataFrame(data,columns=['MoranI', 'pval', 'dist_thr','chosen_dist'])

data=[]
chosen_dist= "LD"
for i in range(1,10): # LD ranges
  I, p_value ,pct_nonzerow = calculate_moran_index(NN_lev_full, NN_id_full, NN_label_full[:,0], distance_threshold=i, weight_distance=True)
  data.append([I, p_value,i,chosen_dist])

moran_df_LD=pd.DataFrame(data,columns=['MoranI', 'pval', 'dist_thr','chosen_dist'])
# Join in one single dataset
moran_df=pd.concat((moran_df_ED,pd.DataFrame(data,columns=['MoranI', 'pval', 'dist_thr','chosen_dist'])))
#moran_df.to_csv("moranI_ED_LD.csv")

# -------------------------Load and save----------------------
# save_to_pickle(moran_df,"moranI_ED_LD.pkl")
# moran_df= load_from_pickle("moranI_ED_LD.pkl")
#-------------------------------------------------------------

# -------------------------------------------Plot analysis main results ----------------------------------

tmp_ed_sum=vicinity_analysis_instance.summary_results   # to get VICINITY percentages of ED radius dist
d_mean1                                                 # to get VICINITY percentages of LD radius dist
tmp_ED1 = moran_df[moran_df['chosen_dist']=="ED"] # to get MORAN values of ED 
tmp_LD1 = moran_df[moran_df['chosen_dist']=="LD"] # to get MORAN values of LD

LD_perc =d_res1.loc[:, [f'Perc_LD_{i}' for i in range(1,max_LD+1)]].mean() # get MEANs across affinities for each LD
LD_mean_perc =d_mean1.loc[:, [f'Avg_Num_LD_{i}' for i in range(1,max_LD+1)]].mean() # get MEANs across affinities for each LD
LD_NULL_perc =d_mean1.loc[:, [f'Avg_NaN_Percentage_LD_{i}' for i in range(1,max_LD+1)]].mean() # get NULL % across  affinities for each LD
# manual annotation of LD mean Percentage [0.97,0.92,0.81,0.70,0.5,0.4,0.35,0.30,0.27]  -- i lost some data didin't have time to recompute


#tmp_ed_sum=load_from_pickle(f'vicinity_radius_summary_unbal_50.pkl')

plt.figure(figsize=(10, 6))

plt.plot(tmp_ed_sum['Mean_Percentage'], tmp_ED1['MoranI'][7:], marker='o', color='blue', label=f'ED') # [7:] to  start from ED 7
plt.plot(LD_mean_perc, tmp_LD1['MoranI'], marker='o', color='red', label=f'LD')

# Annotate ED distance for each point - ED
for i, txt in enumerate(tmp_ED1['dist_thr'][7:]):
    print(i)
    plt.text(tmp_ed_sum['Mean_Percentage'][i]+0.02, tmp_ED1['MoranI'][7:][txt], str(txt), fontsize=8)

  # Adding details to the plot
plt.xlabel('Vicinity')
plt.ylabel("Moran's I")
plt.title(f"Moran's I vs. Vicinity ")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'Vicinity_Vs_MoranI_correct.png')
plt.close()


# Classic Vicinity
plt.figure(figsize=(10, 6))
plt.plot( tmp_ed_sum['Threshold'],tmp_ed_sum['Mean_Percentage'], marker='o', color='blue', label=f'ED')
plt.plot( range(1,max_LD+1),LD_perc, marker='o', color='red', label=f'LD')

# Annotate the Average Num of points for each point for ED
for i  in range(len(tmp_ed_sum)):
    plt.text(tmp_ed_sum['Threshold'][i], tmp_ed_sum['Mean_Percentage'][i]+0.02,
              str(round(tmp_ed_sum['Mean_Num_Points'].iloc[i],ndigits=2)),fontsize=8)

# Annotate Average Num of points for each point for LD
for i  in range(0,max_LD):
    plt.text(int(i+1) , LD_perc[i]+0.02,str(round(LD_mean_perc[i],ndigits=2)),fontsize=8)

# Adding details to the plot
plt.xlabel('Distance THR - ED or LD')
plt.ylabel("Vicinity score")
plt.title(f"Vicinity and Avg NN at Distance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'Vicinity_avgNN_thr.png')
plt.close()

#-----  % Null points
plt.figure(figsize=(10, 6))
plt.plot(tmp_ed_sum['Percentage_Null'], tmp_ed_sum['Mean_Percentage'], marker='o', color='blue', label='ED')
plt.plot(LD_NULL_perc*100,LD_perc, marker='o', color='red', label='LD')
# Annotate each point for ED
for i  in range(len(tmp_ed_sum)):
    plt.text(tmp_ed_sum['Percentage_Null'][i]+2, tmp_ed_sum['Mean_Percentage'][i],
              str(tmp_ed_sum['Threshold'].iloc[i]), fontsize=8)

# Annotate each point for LD, adjust your data accordingly
for i  in range(0,max_LD):
    plt.text((LD_NULL_perc[i]*100)-2, LD_perc[i], str(i+1), fontsize=8)

# Adding details to the plot
plt.xlabel('Perc of Null points')
plt.ylabel("Vicinity Score")
plt.title("Vicinity vs. Null points")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'Vicinity_Null_perc.png')
plt.close()




# --------------------------------------------Classic Moran Plot
plt.plot(tmp_ED1['dist_thr'][7:], tmp_ED1['MoranI'][7:], marker='o', color='blue', label=f'ED')
plt.plot(tmp_LD1['dist_thr'], tmp_LD1['MoranI'], marker='o', color='red', label=f'LD')


# Adding details to the plot
plt.xlabel('Distance THR - ED or LD')
plt.ylabel("Moran's I")
plt.title(f"Moran's I and Distance thr ")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'MoranI_dist_thr.png')
plt.close()

#-----  % Null points
plt.figure(figsize=(10, 6))
plt.plot(tmp_ed_sum['Percentage_Null'], tmp_ED1['MoranI'], marker='o', color='blue', label='ED')
plt.plot(LD_NULL_perc, tmp_LD1['MoranI'], marker='o', color='red', label='LD')
# Annotate each point for ED
#for i, txt in enumerate(tmp_ed_sum):
#    plt.text(tmp_ed_sum[i]+2, moran_df_ED['MoranI'][i], str(moran_df_ED['dist_thr'].iloc[i]), fontsize=8)
# Annotate each point for LD, adjust your data accordingly
#for i, txt in enumerate(LD_NULL_perc):
#    print(i)
#    plt.text(LD_NULL_perc[i]-2, moran_df_LD1['MoranI'].iloc[i], str(moran_df_LD1['dist_thr'].iloc[i]), fontsize=8)

# Adding details to the plot
plt.xlabel('perc of null points')
plt.ylabel("Moran's I")
plt.title("Moran's I vs. Null points")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'MoranI_Null_perc.png')
plt.close()

#-------------- Moran Vs Vicinity

plt.figure(figsize=(10, 6))
plt.plot(tmp_ed_sum['Mean_Percentage'], tmp_ED1['MoranI'][7:], marker='o', color='blue', label=f'ED') # [7:] to  start from ED 7
plt.plot(LD_mean_perc, tmp_LD1['MoranI'], marker='o', color='red', label=f'LD')

# Annotate ED distance for each point - ED
for i, txt in enumerate(tmp_ED1['dist_thr'][7:]):
    print(i)
    plt.text(tmp_ed_sum['Mean_Percentage'][i]+0.02, tmp_ED1['MoranI'][7:][txt], str(txt), fontsize=8)

  # Adding details to the plot
plt.xlabel('Vicinity')
plt.ylabel("Moran's I")
plt.title(f"Moran's I vs. Vicinity ")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'Vicinity_Vs_MoranI_correct.png')
plt.close()


#-----------------------------------------Unbalancing datasets--------------% 70  read to work---------------------------------------
import random
np.random.seed(123)
random.seed(123)
df_hb_50percent = df_hb.sample(n=min(len(df_hb), len(df_lb)), random_state=42)
df_lb_50percent = df_lb.sample(n=min(len(df_hb), len(df_lb)), random_state=42)

df_hblb_50percent = pd.concat([df_hb_50percent, df_lb_50percent], ignore_index=True)

def create_subset(sample_size, hb_percentage, df_hblb_50percent):
    random.seed(123)
    df_new = pd.concat([
        df_hblb_50percent.query("affinity == 'hb'").sample(int(sample_size * hb_percentage), random_state=42),
        df_hblb_50percent.query("affinity == 'lb'").sample(int(sample_size * (1-hb_percentage)), random_state=42)
    ])
    df_new.reset_index(drop=True, inplace=True)  #important! 
    # TODO check discrepancies in the use of .loc and .iloc in Vicinity_class -- this causes problems with indexes
    df_new['id'] = range(0, len(df_new) )
  
    return df_new

hb_percentages= [0.1, 0.3, 0.5, 0.7, 0.9]



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



# -----------------------------------------Local Moran I ----------  work in progress --------------------#


# Run Local Moran I analysis
for hb_percentage in hb_percentages:
    #unbal_df=create_subset(100000,hb_percentage,df_hblb_50percent)
    #vicinity_result = run_vicinity_analysis(unbal_df, neighbor_numbers)
    #percentages_results, res_df, mean_num_points, LD1_res, LD2_res = vicinity_result.perc_Euclidian_radius(range(7, 25))
    
    # Save the vicinity analysis instance
    # LOAD the vicinity analysis instance
    vicinity_result= Vicinity_analysis.load_from_pickle(f'vicinity_analysis_unbal_{int(hb_percentage * 100)}.pkl')
    #save_to_pickle(vicinity_result.summary_results,f'vicinity_radius_summary_unbal_{int(hb_percentage * 100)}.pkl')
    
    # Calculate and save Moran's indices for Euclidean distance (ED) and Lev distance (LD)
    moran_LOC_data_ED = []
    moran_LOC_data_LD = []
    for i in range(8, 20):  # Example range for Euclidean
        moran_hb, moran_lb= calculate_moran_index_V2(vicinity_result.NN_dist, vicinity_result.NN_id, vicinity_result.NN_label[:,0], i, weight_distance=False)
        moran_LOC_data_ED.append([moran_hb, moran_lb, i, "ED"])
    for i in range(1, 10):  # Example range for Lev
        moran_hb, moran_lb = calculate_moran_index_V2(vicinity_result.NN_lev, vicinity_result.NN_id, vicinity_result.NN_label[:,0], i, weight_distance=False)
        moran_LOC_data_LD.append([moran_hb, moran_lb, i, "LD"])
    
    moran_df_ED = pd.DataFrame(moran_LOC_data_ED, columns=['MoranI_loc_hb', 'MoranI_loc_lb', 'dist_thr', 'chosen_dist'])
    moran_df_LD = pd.DataFrame(moran_LOC_data_LD, columns=['MoranI_loc_hb', 'MoranI_loc_lb', 'dist_thr', 'chosen_dist'])
    
    moran_df_ED.to_csv( f'moran_LOC_results_ED_unbal_{int(hb_percentage * 100)}.csv' ,index=False)
    moran_df_LD.to_csv( f'moran_LOC_results_LD_unbal_{int(hb_percentage * 100)}.csv' , index=False)


# Local moran try

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


calculate_moran_index_V2(NN_dist_sim, NN_id_sim, NN_label_sim[:,0], 7, weight_distance=False)

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
mi = Moran(values, w)
print(f"Moran's I: {mi.I}, p-value: {mi.p_sim}")


    
label_encoder = LabelEncoder()
val = label_encoder.fit_transform(values)
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# Calculate Moran's 
Moran_loc = Moran_Local(val, w, permutations=999)
# Accessing Moran's  results
Moran_loc_Is = Moran_loc.Is
p_values = Moran_loc.p_sim


Moran_loc_Is[values == 0].mean(), # hb
Moran_loc_Is[values == 1].mean() # lb
lb_Is = df[df['point_type'] == 'lb']['_Is']

print(f"Moran's I: {mi.I}, p-value: {mi.p_sim}")




def calculate_moran_index_V2(distance_mat, NN_id_mat, label_target, distance_threshold, weight_distance=False):
    # Define who are the neighbors -- EU or LD threshold
    print("calc spatial matrix")
    spatial_NN = (distance_mat <= distance_threshold).astype(int)
    # spatial_NN[distance_mat > distance_threshold] = 0
    
    # Number of observations
    n = distance_mat.shape[0]
    print("calc NN")
    # Create the neighbors and weights dictionaries
    neighbors = {i: NN_id_mat[i][spatial_NN[i] > 0].tolist() for i in range(n)}
    
    if weight_distance == True:
        print("calc Weighted weights")
        distance_mat=distance_mat+ 1e-9
        weights = {i: distance_mat[i][spatial_NN[i] > 0].tolist() for i in range(n)}  # Assume all nonzero are weights
        w = W(neighbors, weights, silence_warnings=True)
    elif weight_distance==False:
        print("calc  weights")
        w = W(neighbors)        
    # Encoding labels into numerical values
    label_encoder = LabelEncoder()
    labels = label_target
    values = label_encoder.fit_transform(labels)
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print("Label to number mapping:", label_mapping)    
    print(f"Calculate Moran's I for distance {distance_threshold}")
    mi = Moran_Local(values, w,permutations=999)
    Moran_loc_Is = mi.Is
    # p_values = Moran_loc_Is.p_simc
    print ( Moran_loc_Is[values == 0].mean(), Moran_loc_Is[values == 1].mean() )
    return Moran_loc_Is[values == 0].mean(), Moran_loc_Is[values == 1].mean() # lb   
    # Print properties to ensure it's set up correctly
    # print("Number of observations:", w.n)
    # print("Percentage of nonzero weights:", "%.3f" % w.pct_nonzero)
    # print(f"Moran's I: {mi.I}, p-value: {mi.p_sim} , threshold = {distance_threshold}")
    # return mi.I, mi.p_sim ,w.pct_nonzero


hb_nan=d_mean1[1].iloc[0,[1,3,5,7,9]].tolist()  #hb
hb_nan.extend([ 0,0,0,0])
hb_nan=np.array(hb_nan)*100
  
lb_nan=d_mean1[1].iloc[1,[1,3,5,7,9]].tolist()  #lb
lb_nan.extend([ 0,0,0,0])
lb_nan=np.array(lb_nan)*100











