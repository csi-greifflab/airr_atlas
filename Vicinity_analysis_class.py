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
from sklearn.preprocessing import LabelEncoder
import pickle
from esda import Moran
from esda import Moran_Local
from libpysal.weights import W


def compute_levenshtein(ref_seq, sequences):
    return [lev.distance(ref_seq, seq) for seq in sequences]

def save_to_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_from_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)        

class Vicinity_analysis:
    # TO DO:
    #---------!!!!!!!!!!!! add the summary resut  from euclidian radius to the save_pickle and load_from_pickle methods !!!!!!!!
    # - Add Marina and Evgenii types of anaylsis
    # - Add Marina and Evgenii plots
    # - uniform data structures or think of a uniform new one (like the multidim array)
    # - If slow, implement multi threading
    # - sparsity normalization
    # - calling the umap script to get the xy coordinates (if it's not too difficult to implement)
    # GITHUB
    # - whatever you think could  be useful :)
    
    
    
    def __init__( self, df,neighbor_numbers,id_index  ):
        self.df=df
        self.neighbor_numbers= neighbor_numbers
        # self.neigh = NearestNeighbors()
        # self.neigh.fit(list(self.df['embedding']))
        self.id_index=id_index
        print(f"Analysis initialized for index {len(id_index)} with neighbor numbers: {neighbor_numbers}")
        #self.parameters=  TO DO , PASTe THE INPUT PARAMETERS to have a record of the anaylsis done
        
    def run_analysis(self):
        print("Running the analysis...")
        start_time = time.time()
        self.fractions_results, self.NN_id, self.NN_dist, self.NN_label, self.NN_lev, self.ID_labels = self.calculate_fractions_for_data()
        self.result_dict = dict(zip(self.neighbor_numbers, self.fractions_results))
        print(f"Analysis results: {self.result_dict}")
        print(f"Total analysis time: {time.time() - start_time:.2f} seconds")
        return self
        
    def save_to_pickle(self,file_name):
        data_to_save = {
        "neighbor_numbers": self.neighbor_numbers,
        "neigh": self.neigh ,
        "id_index" :self.id_index,
        "fractions_results": self.fractions_results,
        "NN_id": self.NN_id,
        "NN_dist": self.NN_dist,
        "NN_label": self.NN_label,
        "NN_lev": self.NN_lev,
        "ID_labels": self.ID_labels,
        "summary_results": self.summary_results
        #"analysis_info": self.paramters
        }
        with open(file_name, 'wb') as file:
            pickle.dump(data_to_save, file)
            
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
        return instance
        
        
    def calculate_fractions_for_data(self):
        self.neigh = NearestNeighbors()
        self.neigh.fit(list(self.df['embedding']))
        # Compute the nearest neighbors for the maximum number of neighbors needed
        distances, indices = self.neigh.kneighbors(self.df.iloc[self.id_index]['embedding'].tolist(), n_neighbors=max(self.neighbor_numbers))
        fractions_results = []
        indices_affinity = self.df.loc[indices.flatten(), 'affinity'].values
        # Redimension affinity values array to corrispond to indices shape
        id_affinty_label=self.df.loc[self.id_index, 'affinity']
        indices_affinity_mat = indices_affinity.reshape(indices.shape)
        t_lev= time.time()
        
        print("Computing Levenshtein distances...")
        lev_mat = []
        # tqdm progress bar for Levenshtein distance computation
        for idx in tqdm(range(len(self.id_index)), desc="Levenshtein distances"):
            seq_index = self.id_index[idx]
            NN_index = indices[idx]
            lev_mat.append(compute_levenshtein(self.df.iloc[seq_index]['junction_aa'], self.df.iloc[NN_index]['junction_aa']))
        lev_mat = np.array(lev_mat)
        # lev_mat=[]
        # for idx, seq_index in enumerate(self.id_index):
        #     NN_index=indices[idx]
        #   lev_mat.append(compute_levenshtein(self.df.iloc[seq_index]['junction_aa'],self.df.iloc[NN_index]['junction_aa']) )
        # lev_mat= np.array(lev_mat)
        print(f' LEV running time {time.time()-t_lev}')
        t_NN= time.time()
        print("Calculating nearest neighbors fractions...")
        for n in tqdm(self.neighbor_numbers, desc="Neighbors analysis"):
            result = self._calculate_fractions_for_subset(indices, [n])
            fractions_results.append(result)
        # for n in self.neighbor_numbers:
        #     result = self._calculate_fractions_for_subset(indices, [n])
        #     fractions_results.append(result)
        #     print(n)
        print(f' NN running time {time.time()-t_lev}')
        
        return fractions_results,indices,distances,indices_affinity_mat,lev_mat,id_affinty_label
        
    def _calculate_fractions_for_subset(self, indices, neighbor_subset):
        percentages = []
        for n in neighbor_subset:
            for idx, given_index in enumerate(self.id_index):
                indices_slice = indices[idx]
                percentage = self._calculate_fraction(indices_slice, n, given_index)
                percentages.append(percentage)
        return np.mean(percentages)
    
    def _calculate_fraction(self, indices_slice, n_neighbors, given_index):
        given_affinity = self.df.iloc[given_index]['affinity']
        neighbors_indices = indices_slice[1:n_neighbors+1]
        neighbors_affinity = self.df.iloc[neighbors_indices]['affinity']
        perc = (neighbors_affinity == given_affinity).sum() / len(neighbors_affinity)    
        return perc
        
    def perc_Euclidian_radius(self, distance_thresholds):
        results, LD1_res, LD2_res = [], [], []
        res_df = pd.DataFrame(columns=[f'EU_{i}' for i in distance_thresholds])
        mean_num_points = []
        summary_data = []
        for threshold in distance_thresholds:
            percentages, LD1_list, LD2_list, num_of_points_within_rad = [], [], [], []
            for i in range(self.NN_dist.shape[0]):
                within_threshold_indices = np.where(self.NN_dist[i] <= threshold)[0][1:]  # Exclude the point itself
                num_of_points_within_rad.append(len(within_threshold_indices))
                
                if len(within_threshold_indices) != 0:
                    ref_label = self.ID_labels.iloc[i]
                    labels_within_threshold = self.NN_label[i, within_threshold_indices]
                    percentage = np.sum(labels_within_threshold == ref_label) / len(within_threshold_indices)
                    percentages.append(percentage)
                    LD1_list.append(np.sum(self.NN_lev[i, within_threshold_indices] == 1) / len(within_threshold_indices))
                    LD2_list.append(np.sum(self.NN_lev[i, within_threshold_indices] == 2) / len(within_threshold_indices))
                else:
                    percentages.append(np.nan)
                    LD1_list.append(np.nan)
                    LD2_list.append(np.nan)
            
            results.append(np.nanmean(percentages))
            LD1_res.append(np.nanmean(LD1_list))
            LD2_res.append(np.nanmean(LD2_list))
            mean_num_points.append(np.mean(num_of_points_within_rad))
            res_df[f'EU_{threshold}'] = percentages
            
            #Store results in the summary DataFrame
            summary_data.append({
                'Threshold': threshold,
                'Mean_Percentage': results[-1],
                'Mean_LD1': LD1_res[-1],
                'Mean_LD2': LD2_res[-1],
                'Mean_Num_Points': mean_num_points[-1],
                'Percentage_Null': sum(res_df[f'EU_{threshold}'].isna()) / len(res_df[f'EU_{threshold}']) * 100
            })
            
        # Create and store the DataFrame in the class attribute
        self.summary_results = pd.DataFrame(summary_data)
        
        for idx,i in enumerate(res_df.columns):
            null_points=sum(res_df[i].isna())
            print(f'{i}:{results[idx]:.4f} ,n_points= {mean_num_points[idx]:.4f}, %null={null_points/len(res_df[i])*100:.4f}, perc_of_LD1= {LD1_res[idx]:.4f}, perc_of_LD2= {LD2_res[idx]:.4f}')
            
        return results, res_df, mean_num_points, LD1_res, LD2_res        
        

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
        
        
