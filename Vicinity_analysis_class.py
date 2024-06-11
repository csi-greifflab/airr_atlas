<<<<<<< Updated upstream


=======
from functools import partial
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
from sklearn.preprocessing import LabelEncoder
import pickle
from esda import Moran
from esda import Moran_Local
from libpysal.weights import W
import time

def compute_levenshtein(ref_seq, sequences):
    return [lev.distance(ref_seq, seq) for seq in sequences]

def save_to_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_from_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)        

def analyze_neighbors(instance, indices, n, id_affinity_label):
    result, label_result, knn_perc = instance._calculate_fractions_for_subset(indices, [n], id_affinity_label)
    return result, label_result, knn_perc


>>>>>>> Stashed changes
class Vicinity_analysis:
    # TO DO:
    # - Add Marina and Evgenii types of anaylsis
    # - Add Marina and Evgenii plots
    # - uniform data structures or think of a uniform new one (like the multidim array)
    # - If slow, implement multi threading
    # - sparsity normalization
    # - calling the umap script to get the xy coordinates (if it's not too difficult to implement)
    # GITHUB
    # - whatever you think could  be useful :)
    
    
    
    def __init__( self, df,neighbor_numbers,id_index, colname_affinity='affinity', colname_junction='junction_aa', metric= "euclidean"  ):
        self.df=df
        self.neighbor_numbers= neighbor_numbers
<<<<<<< Updated upstream
        self.neigh = NearestNeighbors()
        self.neigh.fit(list(self.df['embedding']))
=======
        self.colname_affinity=colname_affinity
        self.colname_junction=colname_junction
        self.metric = metric
        # self.neigh = NearestNeighbors()
        # self.neigh.fit(list(self.df['embedding']))
>>>>>>> Stashed changes
        self.id_index=id_index
        #self.parameters=  TO DO , PASTe THE INPUT PARAMETERS to have a record of the anaylsis done
    
    def run_analysis(self):
        self.fractions_results,self.NN_id,self.NN_dist, self.NN_label, self.NN_lev, self.ID_labels = self.calculate_fractions_for_data()
        self.result_dict = dict(zip(self.neighbor_numbers, self.fractions_results))
        print( self.result_dict )
        return self
        
    def save_to_pickle(self,file_name):
        data_to_save = {
        "fractions_results": self.fractions_results,
        "NN_id": self.NN_id,
        "NN_dist": self.NN_dist,
        "NN_label": self.NN_label,
        "NN_lev": self.NN_lev,
<<<<<<< Updated upstream
        "ID_labels": self.ID_labels
=======
        "ID_labels": self.ID_labels,
        "summary_results": self.summary_results,
        "label_results": self.label_results,
>>>>>>> Stashed changes
        #"analysis_info": self.paramters
        }
        with open(file_name, 'wb') as file:
            pickle.dump(data_to_save, file)
<<<<<<< Updated upstream
        
        
    def calculate_fractions_for_data(self):
=======
            
    @classmethod    
    def load_from_pickle(cls, file_name, df_default=None, neighbor_numbers_default=None, id_index_default=None):
        with open(file_name, 'rb') as file:
            data_loaded = pickle.load(file)
        # Initialize object attributes from the loaded data
        instance = cls(  df=data_loaded.get("df", df_default),
            neighbor_numbers=data_loaded.get("neighbor_numbers", neighbor_numbers_default),
            id_index=data_loaded.get("id_index", id_index_default)  )
        instance.neighbor_numbers = data_loaded.get("neighbor_numbers", None)
        instance.neigh = data_loaded.get("neigh", None)
        instance.id_index = data_loaded.get("id_index", None)
        instance.fractions_results = data_loaded.get("fractions_results", None)
        instance.NN_id = data_loaded.get("NN_id", None)
        instance.NN_dist = data_loaded.get("NN_dist", None)
        instance.NN_label = data_loaded.get("NN_label", None)
        instance.NN_lev = data_loaded.get("NN_lev", None)
        instance.ID_labels = data_loaded.get("ID_labels", None)
        instance.summary_results = data_loaded.get("summary_results", None)
        instance.label_results = data_loaded.get("label_results", None)
        return instance
    
       
    def calculate_fractions_for_data(self):
        self.df['affinity']= self.df[self.colname_affinity]
        self.df['junction_aa']= self.df[self.colname_junction]
        self.neigh = NearestNeighbors(metric=self.metric)
        self.neigh.fit(list(self.df['embedding']))
>>>>>>> Stashed changes
        # Compute the nearest neighbors for the maximum number of neighbors needed
        print("Compunting KNN ...")
        distances, indices = self.neigh.kneighbors(self.df.iloc[self.id_index]['embedding'].tolist(), n_neighbors=max(self.neighbor_numbers))
        fractions_results = []
        indices_affinity = self.df.loc[indices.flatten(), 'affinity'].values
        # Redimension affinity values array to corrispond to indices shape
        id_affinty_label=self.df.loc[self.id_index, 'affinity']
        indices_affinity_mat = indices_affinity.reshape(indices.shape)
<<<<<<< Updated upstream
        t_lev= time.time()
        lev_mat=[]
        for idx, seq_index in enumerate(self.id_index):
            NN_index=indices[idx]
            lev_mat.append(compute_levenshtein(self.df.iloc[seq_index]['junction_aa'],self.df.iloc[NN_index]['junction_aa']) )
        lev_mat= np.array(lev_mat)
        print(f' LEV running time {time.time()-t_lev}')
        t_NN= time.time()
        for n in self.neighbor_numbers:
            result = self._calculate_fractions_for_subset(indices, [n])
            fractions_results.append(result)
=======
        t_lev= time.time()        
        print("Computing Levenshtein distances...")
        lev_mat = []
        # # tqdm progress bar for Levenshtein distance computation
        for idx in tqdm(range(len(self.id_index)), desc="Levenshtein distances"):
            seq_index = self.id_index[idx]
            NN_index = indices[idx]
            lev_mat.append(compute_levenshtein(self.df.iloc[seq_index]['junction_aa'], self.df.iloc[NN_index]['junction_aa']))
        lev_mat = np.array(lev_mat)
        print(f' LEV running time {time.time()-t_lev}')
        t_NN= time.time()
        print("Calculating nearest neighbors fractions...")
        tmp_label_res = []
        
        
        knn_vicinity=[]
        # for n in tqdm(self.neighbor_numbers, desc="Neighbors analysis"):
        #     result, label_result , knn_perc = self._calculate_fractions_for_subset(indices, [n],id_affinty_label)
        #     fractions_results.append(result)
        #     tmp_label_res.append(label_result)
        #     knn_vicinity.append(knn_perc)
        # Use joblib.Parallel to parallelize the computation
        results = Parallel(n_jobs= 10)(
            delayed(analyze_neighbors)(self, indices, n, id_affinty_label) 
            for n in tqdm(self.neighbor_numbers, desc="Neighbors analysis")
        )
       # Create a partial function with 'self' bound
        # analyze_neighbors_partial = partial(self.analyze_neighbors, self)
        # # Use joblib.Parallel to parallelize the computation
        # results = Parallel(n_jobs=5)(
        #     delayed(analyze_neighbors_partial)(n) for n in tqdm(self.neighbor_numbers, desc="Neighbors analysis")
        # )
        # results = Parallel(n_jobs=5)(
        #     delayed(self.analyze_neighbors)(n) for n in tqdm(self.neighbor_numbers, desc="Neighbors analysis")
        # )
        fractions_results,tmp_label_res,knn_vicinity = [] ,[],[]
        # Unpack the results
        for result, label_result, knn_perc in results:
            fractions_results.append(result)
            tmp_label_res.append(label_result)
            knn_vicinity.append(knn_perc)
        
>>>>>>> Stashed changes
        print(f' NN running time {time.time()-t_lev}')
        knn_vicinity= np.array(knn_vicinity)
        # We create an empty DataFrame and fill it with the results
        label_idx = sorted(set(key for dic in tmp_label_res for key in dic.keys()))
        result_labels_df = pd.DataFrame(index=label_idx)# Create index from all possible labels
        for i, result in enumerate(tmp_label_res): # i should be the respective NN
            for label, percentage in result.items():
                result_labels_df.loc[label, i] = percentage  # Set the percentage in the correct position
        print(result_labels_df)
        self.label_results = result_labels_df
        self.knn_vicinity = knn_vicinity
        
        return fractions_results,indices,distances,indices_affinity_mat,lev_mat,id_affinty_label
    
    def _calculate_fractions_for_subset(self, indices, neighbor_subset,id_affinty_label):
        percentages = []
        label_percentages={}
        for n in neighbor_subset:       # useless , just loops once, should be corrected
            for idx, given_index in enumerate(self.id_index):
                indices_slice = indices[idx]
                percentage = self._calculate_fraction(indices_slice, n, given_index)
                percentages.append(percentage)  
                id_label = id_affinty_label.iloc[idx]  # get the label of the current id
                # Append the calculated percentage to the corresponding label's list in the dictionary
                if id_label not in label_percentages:
                    label_percentages[id_label] = []
                label_percentages[id_label].append(percentage)    
        label_means = {label: np.mean(percentages) for label, percentages in label_percentages.items()}
        #hb_perc=label_percentages['hb']  
        #print(f'these are the percentages {hb_perc} at Neigh susbet {neighbor_subset} ')
        return np.mean(percentages), label_means, percentages 
    
    def _calculate_fraction(self, indices_slice, n_neighbors, given_index):
        given_affinity = self.df.iloc[given_index]['affinity']
        neighbors_indices = indices_slice[1:n_neighbors+1]
        neighbors_affinity = self.df.iloc[neighbors_indices]['affinity']
        perc = (neighbors_affinity == given_affinity).sum() / len(neighbors_affinity)    
        return perc
        
    def calculate_percentages_with_precomputed_distances(self, distance_thresholds):
        results, LD1_res, LD2_res = [], [], []
        res_df = pd.DataFrame(columns=[f'EU_{i}' for i in distance_thresholds])
        mean_num_points = []
        for threshold in distance_thresholds:
            print(f'Computing radius at thr {threshold}...')
            percentages, LD1_list, LD2_list, num_of_points_within_rad = [], [], [], []
            label_percentages={}
            label_mean_point={}
            label_null_points = {}  # Dictionary to store null points by label
            label_LD_sim = {}
            for i in range(self.NN_dist.shape[0]):
                within_threshold_indices = np.where(self.NN_dist[i] <= threshold)[0][1:]  # Exclude the point itself
                #print(within_threshold_indices)
                LD_within_threshold= self.NN_lev[i,within_threshold_indices]  #extract the NN in the LD matrix
                num_of_points_within_rad.append(len(within_threshold_indices)) 
                ref_label = self.ID_labels.iloc[i]
                if ref_label not in label_null_points:
                        label_null_points[ref_label] = 0  # Initialize null points                               
                if len(within_threshold_indices) != 0:
                    labels_within_threshold = self.NN_label[i, within_threshold_indices]
                    percentage = np.sum(labels_within_threshold == ref_label) / len(within_threshold_indices)
                    percentages.append(percentage)
                    LD_sim=LD_within_threshold.mean() #compute the mean sequence similarity of the NN
                    #print(f"id{i} ld dist are {LD_within_threshold[:4]}")
                    #for each unique label initialize a dictionary key if not already present
                    if ref_label not in label_percentages:
                        label_percentages[ref_label] = []
                        label_mean_point[ref_label] =[]
                        label_null_points[ref_label] = 0  # Initialize null points
                        label_LD_sim[ref_label] = []  # Initialize LD sim
                    #append perc and mean_points
                    label_percentages[ref_label].append(percentage)
                    label_mean_point[ref_label].append( len(within_threshold_indices) )
                    label_LD_sim[ref_label].append(LD_sim)
                    LD1_list.append(np.sum(self.NN_lev[i, within_threshold_indices] == 1) / len(within_threshold_indices))
                    LD2_list.append(np.sum(self.NN_lev[i, within_threshold_indices] == 2) / len(within_threshold_indices))
                else:
                    percentages.append(np.nan)
                    LD1_list.append(np.nan)
                    LD2_list.append(np.nan)
                    label_null_points[ref_label] += 1  # Increment null count for label
            label_perc = {label: np.mean(percentages) for label, percentages in label_percentages.items()}
            label_avg_points = {label: np.mean(percentages) for label, percentages in label_mean_point.items()}
            label_LD=   {label: np.mean(percentages) for label, percentages in label_LD_sim.items()}
            results.append(np.nanmean(percentages))
            LD1_res.append(np.nanmean(LD1_list))
            LD2_res.append(np.nanmean(LD2_list))
            mean_num_points.append(np.mean(num_of_points_within_rad))
<<<<<<< Updated upstream
            res_df[f'EU_{threshold}'] = percentages
=======
            res_df[f'EU_{threshold}'] = percentages          
            #Store results in the summary DataFrame
            summary_row=({
                'Threshold': threshold,
                'Mean_Percentage': results[-1],
                'Mean_LD1': LD1_res[-1],
                'Mean_LD2': LD2_res[-1],
                'Mean_Num_Points': mean_num_points[-1],
                'Percentage_Null': sum(res_df[f'EU_{threshold}'].isna()) / len(res_df[f'EU_{threshold}']) * 100
            })
            # Incorporate label-specific metrics into the summary row
            for label in label_perc:  # loop over all unique keys 
                summary_row[f'Perc_{label}'] = label_perc[label]
                summary_row[f'AvgPoints_{label}'] = label_avg_points[label]
                summary_row[f'NULLPerc_{label}'] = label_null_points[label] / len([x for x in self.ID_labels if x == label]) * 100
                summary_row[f'LD_avgSim_{label}'] = label_LD[label] 
            #Append to the general summmary
            summary_data.append(summary_row)
                   
        # Create and store the DataFrame in the class attribute
        self.summary_results = pd.DataFrame(summary_data)
        #Printing results
>>>>>>> Stashed changes
        for idx,i in enumerate(res_df.columns):
            null_points=sum(res_df[i].isna())
            print(f'{i}:{results[idx]:.4f} ,n_points= {mean_num_points_LV[idx]:.4f}, %null={null_points/len(res_df[i])*100:.4f}, perc_of_LD1= {LD1_res[idx]:.4f}, perc_of_LD2= {LD2_res[idx]:.4f}')
        
        return results, res_df, mean_num_points, LD1_res, LD2_res        


<<<<<<< Updated upstream
vicinity_analysis_instance = Vicinity_analysis(df, neighbor_numbers, id_index_sample)
vicinity_analysis_instance.run_analysis()  # This populates the necessary attributes
distance_thresholds = range(7, 20)  # Define your distance thresholds
=======
def calculate_moran_index(distance_mat, NN_id_mat, label_target, distance_threshold, weight_distance=False):
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
    mi = Moran(values, w)
    
    # Print properties to ensure it's set up correctly
    print("Number of observations:", w.n)
    print("Percentage of nonzero weights:", "%.3f" % w.pct_nonzero)
    print(f"Moran's I: {mi.I}, p-value: {mi.p_sim} , threshold = {distance_threshold}")
    return mi.I, mi.p_sim ,w.pct_nonzero



def prepare_data_for_plotting(df, LD_dist , sampled_indices= None, junction_aa_col='junction_aa', affinity_col='affinity'):
    df['affinity']= df[affinity_col]
    df['junction_aa']= df[junction_aa_col]
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
    grouped = df_combined.groupby('affinity')
    grouped[[f'Perc_LD_{i}' for i in range(1,LD_dist+1)]].count()
    grouped[[f'Perc_LD_{i}' for i in range(1,LD_dist+1)]].mean()
    df_summary[[f'Num_of_LD_{i}' for i in range(1,LD_dist+1)]] = grouped[[f'Perc_LD_{i}' for i in range(1,LD_dist+1)]].count()
    df_summary[[f'Perc_LD_{i}' for i in range(1,LD_dist+1)]] = grouped[[f'Perc_LD_{i}' for i in range(1,LD_dist+1)]].mean()
    
    return df_combined, df_summary 

>>>>>>> Stashed changes

# Now you can call the new method with these precomputed values
percentages_results11, res_df11, mean_num_points11, LD1_res1, LD2_res1 = vicinity_analysis_instance.calculate_percentages_with_precomputed_distances(distance_thresholds)

<<<<<<< Updated upstream
for idx, i in enumerate(res_df11.columns):
    null_points = sum(res_df11[i].isna())
    print(f'{i}:{percentages_results11[idx]:.4f} ,n_points= {mean_num_points11[idx]:.4f}, %null={null_points/len(res_df11[i])*100:.4f}, perc_of_LD1= {LD1_res1[idx]:.4f}, perc_of_LD2= {LD2_res1[idx]:.4f}')


=======
>>>>>>> Stashed changes
class VicinityPlots:
    def __init__(self, vicinity_instance=None):
        if vicinity_instance:
            self.vicinity_instance = vicinity_instance
            self.neighbor_numbers = vicinity_instance.neighbor_numbers
            self.fractions_results = vicinity_instance.fractions_results
            self.NN_lev = vicinity_instance.NN_lev
            self.NN_dist = vicinity_instance.NN_dist
            self.df = vicinity_instance.df
        else:
            self.vicinity_instance = None

    def plot_basic_vicinity_score(self,file_name, fractions_results=None, neighbor_numbers=None):
        if self.vicinity_instance:
            fractions_results = self.fractions_results
            neighbor_numbers = self.neighbor_numbers
        elif fractions_results is None or neighbor_numbers is None:
            raise ValueError("Fraction results and neighbor numbers must be provided")

        plt.figure(figsize=(10, 6))
        plt.scatter(neighbor_numbers, fractions_results, color='blue', label='Data Points')
        plt.plot(neighbor_numbers, fractions_results, color='red', label='Interpolated Line')
        plt.title('Fraction Results by Number of Neighbors')
        plt.xlabel('Number of Neighbors')
        plt.ylabel('Fraction Results')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'Vicinity_score_NN_{file_name}.png')

    def plot_combined_fraction_results_vs_LD(self,file_name,fractions_results=None, NN_lev=None, neighbor_numbers=None):
        if self.vicinity_instance:
            NN_lev = self.NN_lev
            neighbor_numbers = self.neighbor_numbers
            fractions_results=self.fractions_results
        elif neighbor_numbers is None or fractions_results is None or NN_lev is None:
            raise ValueError("Fraction results, NN_lev, neighbor numbers must be provided")
        
        NN_lev_mean=[]
        row_mean=[]
        for n in neighbor_numbers:
          for i in range(len(NN_lev[:,])):
            row_mean.append(np.mean(NN_lev[i,1:n+1]) )
          # print(row_mean)
          # print(len(row_mean))
          NN_lev_mean.append(np.mean(row_mean) )
          row_mean=[]
        corr_pearson_NN_thr, _ = pearsonr(self.fractions_results, NN_lev_mean)
        
        plt.figure(figsize=(10, 6))
        ax1 = plt.gca()
        ax1.scatter( neighbor_numbers, fractions_results, color='blue', label='AB2 embeddings eucl. distance')
        ax1.set_xlabel('Number of Neighbors')
        ax1.set_ylabel('Fraction Results', color='blue')
        ax1.set_xticks(fractions_results)
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True)
        
        ax2 = ax1.twinx()
        ax2.scatter(neighbor_numbers, NN_lev_mean, color='red', label='LD distance')
        ax2.set_ylabel('Mean Lev', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        plt.title(f'AB2 EU vs LD by kNN__ pearsonCorr = {corr_pearson_NN_thr}')
        plt.savefig('combined_graph_EUvsLD_kNN.png')
<<<<<<< Updated upstream
=======
        
    def KNNvsFraction(self,file_name,fractions_results=None, NN_lev=None, neighbor_numbers=None):
        if self.vicinity_instance:
            NN_lev = self.NN_lev
            neighbor_numbers = self.neighbor_numbers
            fractions_results=self.fractions_results
        elif neighbor_numbers is None or fractions_results is None or NN_lev is None:
            raise ValueError("Fraction results, NN_lev, neighbor numbers must be provided")
        
        NN_lev_mean=[]
        row_mean=[]
        for n in neighbor_numbers:
          for i in range(len(NN_lev[:,])):
            row_mean.append(np.mean(NN_lev[i,1:n+1]) )
          # print(row_mean)
          # print(len(row_mean))
          NN_lev_mean.append(np.mean(row_mean) )
          row_mean=[]
        corr_pearson_NN_thr, _ = pearsonr(self.fractions_results, NN_lev_mean)
        
        plt.figure(figsize=(10, 6))
        ax1 = plt.gca()
        ax1.scatter( neighbor_numbers, fractions_results, color='blue', label='AB2 embeddings eucl. distance')
        ax1.set_xlabel('Number of Neighbors')
        ax1.set_ylabel('Fraction Results', color='blue')
        ax1.set_xticks(fractions_results)
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True)
        
        ax2 = ax1.twinx()
        ax2.scatter(neighbor_numbers, NN_lev_mean, color='red', label='LD distance')
        ax2.set_ylabel('Mean Lev', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        plt.title(f'AB2 EU vs LD by kNN__ pearsonCorr = {corr_pearson_NN_thr}')
        plt.savefig('combined_graph_EUvsLD_kNN.png')
        
        


def run_ggplot_vicinity( analysis_name  , input_ED,input_LD, output_path= None):
#    Activates the specified Conda environment and runs the given R script.
    import subprocess
    source_env= "source /opt/anaconda3/bin/activate"  
    #conda_env= "/doctorai/marinafr/progs/miniconda3/envs/airr_atlas/"
    conda_env= "R4.3.3"
    R_script= "/doctorai/niccoloc/Vicinity_ggplot.r"
    if output_path is None:
        output_path=f"./{analysis_name}_plots"
    # Build the command to activate Conda environment and run the R script
    #command = f"{source_env} && conda activate {conda_env} && R --version"
    #command = f"{source_env} && conda activate {conda_env} && echo $CONDA_DEFAULT_ENV && echo $CONDA_PREFIX"
    command = f"{source_env} && conda activate {conda_env} && /opt/anaconda3/envs/{conda_env}/bin/Rscript {R_script} {input_ED} {input_LD} {output_path} "
    command = f"/doctorai/niccoloc/airr_atlas/run_ggplot_vicinity.sh {input_ED} {input_LD} {output_path}"
    # Execute the command
    print(command)
    try:
        # Using shell=True to handle the command chain correctly
        output = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        print("R script output:", output.stdout)
    except subprocess.CalledProcessError as e:
        print("Error running R script:", e.stderr)








import numpy as np
import pandas as pd
from multiprocessing import Pool, Value, Manager
from tqdm import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

def worker_by_threshold(lev_dist, sampled_indices, df):
    num_samples = len(sampled_indices)
    results = np.zeros(num_samples)
    num_of_points = np.zeros(num_samples)
    affinities = []
    
    for i, index in enumerate(sampled_indices):
        initial_affinity = df.loc[index, 'affinity']
        initial_seq = df.loc[index, 'junction_aa']
        affinities.append(initial_affinity)
        
        lev_dists = compute_levenshtein(initial_seq, df.iloc[1:]['junction_aa'])
        indices_at_dist = [j for j, x in enumerate(lev_dists) if x == lev_dist]
        if indices_at_dist:
            affinities_at_dist = df.iloc[indices_at_dist]['affinity']
            percentage = sum(affinities_at_dist == initial_affinity) / len(affinities_at_dist)
            results[i] = percentage
            num_of_points[i] = len(affinities_at_dist)
        else:
            results[i] = np.nan
            num_of_points[i] = 0
    
    return lev_dist, results, num_of_points, affinities

def prepare_data_for_plotting(df, LD_dist, sampled_indices=None, junction_aa_col='junction_aa', affinity_col='affinity'):
    df['affinity'] = df[affinity_col]
    df['junction_aa'] = df[junction_aa_col]
    
    if sampled_indices is None:
        sampled_indices = df.index
    num_samples = len(sampled_indices)
    
    results = np.zeros((num_samples, LD_dist))
    num_of_points = np.zeros((num_samples, LD_dist))
    affinities = np.zeros((num_samples, LD_dist), dtype=object)
    
    tasks = [(lev_dist, sampled_indices, df) for lev_dist in range(1, LD_dist + 1)]
    
    with tqdm_joblib(desc="Processing Levenshtein distances", total=len(tasks)) as progress_bar:
        results_list = Parallel(n_jobs=16)(delayed(worker_by_threshold)(lev_dist, sampled_indices, df) for lev_dist in range(1, LD_dist + 1))
    
    for lev_dist, res, num_pts, aff in results_list:
        results[:, lev_dist - 1] = res
        num_of_points[:, lev_dist - 1] = num_pts
        affinities[:, lev_dist - 1] = aff
    
    columns = [f'LD_{i}' for i in range(1, LD_dist + 1)]
    df_combined = pd.DataFrame({
        **{f'Perc_{col}': results[:, idx] for idx, col in enumerate(columns)},
        **{f'Num_{col}': num_of_points[:, idx] for idx, col in enumerate(columns)},
        'sample_id': sampled_indices,
        'affinity': np.max(affinities, axis=1)  # Assuming same affinity across all thresholds for each sample
    })
    
    df_summary = pd.DataFrame()
    for col in columns:
        df_combined[f'NaN_Count_{col}'] = df_combined[f'Perc_{col}'].isna()
        summary_stats = df_combined.groupby('affinity')[[f'Num_{col}', f'NaN_Count_{col}']].agg({
            f'Num_{col}': 'mean',
            f'NaN_Count_{col}': 'mean'
        }).rename(columns={f'Num_{col}': f'Avg_Num_{col}', f'NaN_Count_{col}': f'Avg_NaN_Percentage_{col}'})
        df_summary = pd.concat([df_summary, summary_stats], axis=1)
    
    grouped = df_combined.groupby('affinity')
    df_summary[[f'Num_of_LD_{i}' for i in range(1, LD_dist + 1)]] = grouped[[f'Perc_LD_{i}' for i in range(1, LD_dist + 1)]].count()
    df_summary[[f'Perc_LD_{i}' for i in range(1, LD_dist + 1)]] = grouped[[f'Perc_LD_{i}' for i in range(1, LD_dist + 1)]].mean()
    
    return df_combined, df_summary

>>>>>>> Stashed changes
