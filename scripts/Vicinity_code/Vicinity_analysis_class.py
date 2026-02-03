# Standard library imports
import pickle
import statistics
import time
from collections import Counter
from functools import partial
from memory_profiler import profile
import gc 
import os
import argparse
import sys


# Third-party library imports
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Scientific and computational libraries
from scipy import linalg
from scipy.sparse import coo_matrix, lil_matrix
from scipy.sparse import csr_matrix


# Machine learning and data processing
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed

# String distance metrics
from rapidfuzz.distance import Levenshtein as RapidfuzzLevenshtein
from rapidfuzz.process import cdist


# Spatial statistics
from esda import Moran, Moran_Local
from libpysal.weights import W

# Progress tracking
from tqdm.auto import tqdm

# Set seaborn style
sns.set()







def compute_levenshtein(ref_seq, sequences):
    return [RapidfuzzLevenshtein.distance(ref_seq, seq) for seq in sequences]


def save_to_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_from_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)        

def analyze_neighbors(instance, indices, n, id_affinity_label):
    result, label_result, knn_perc = instance._calculate_fractions_for_subset(indices, [n], id_affinity_label)
    return result, label_result, knn_perc


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
    
    
    
    def __init__( self, df,embeddings,neighbor_numbers,id_index, colname_affinity='affinity', colname_junction='junction_aa', metric= "euclidean"  ,parallel=False , skip_KNN = False):
        self.df=df
        self.embedding= embeddings
        self.neighbor_numbers= neighbor_numbers
        self.colname_affinity=colname_affinity
        self.colname_junction=colname_junction
        self.metric = metric
        self.parallel= parallel
        self.skip_neighbors_analysis = skip_KNN
        # self.neigh = NearestNeighbors()
        # self.neigh.fit(list(self.df['embedding']))
        self.id_index=id_index
        print(f"Analysis initialized for index {len(id_index)} with neighbor numbers: {neighbor_numbers}, Parellelization= {parallel}")
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
        "summary_results": self.summary_results,
        "label_results": self.label_results,
        "knn_vicinity": self.knn_vicinity
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
        instance.label_results = data_loaded.get("label_results", None)
        instance.knn_vicinity = data_loaded.get("knn_vicinity", None)
        return instance
    
    def calculate_fractions_for_data_bk(self):
        self.df['affinity']= self.df[self.colname_affinity]
        self.df['junction_aa']= self.df[self.colname_junction]
        # Prepare embeddings and query
        embeddings = np.vstack(self.df['embedding'].values)
        query_embedding = np.array(self.df.iloc[self.id_index]['embedding'], dtype='float32')  # Extract and convert query embedding

        # Reshape query_embedding to match FAISS input requirements
        if query_embedding.ndim == 1:  # Ensure it's a single vector
            query_embedding = query_embedding.reshape(1, -1)
        # Perform FAISS KNN search
        distances, indices = faiss_exact_knn(
            embeddings=embeddings,
            query_embedding=query_embedding,
            metric=self.metric,
            n_neighbors=max(self.neighbor_numbers)
        )

        # Extract affinity values
        indices_affinity = self.df.loc[indices, 'affinity'].values
        indices_affinity_mat = indices_affinity.reshape(-1, max(self.neighbor_numbers))

        id_affinity_label = self.df.loc[self.id_index, 'affinity']
        fractions_results = []
        t_lev = time.time()        
        print("Computing Levenshtein distances...")
        lev_mat = []
        # tqdm progress bar for Levenshtein distance computation
        for idx in tqdm(range(len(self.id_index)), desc="Levenshtein distances"):
            seq_index = self.id_index[idx]
            NN_index = indices[idx]
            lev_mat.append(compute_levenshtein(self.df.iloc[seq_index]['junction_aa'], self.df.iloc[NN_index]['junction_aa']))
        lev_mat = np.array(lev_mat)
        print(f' LEV running time {time.time() - t_lev}')
        
        if self.skip_neighbors_analysis:
            print("Skipping Neighbors analysis...")
            fractions_results = [np.nan] * len(self.neighbor_numbers)
            tmp_label_res = {label: [np.nan] * len(self.neighbor_numbers) for label in id_affinity_label.unique()}
            knn_vicinity = np.zeros((len(self.neighbor_numbers), len(self.id_index)))
            result_labels_df = pd.DataFrame()
        else:
            t_NN = time.time()
            print("Calculating nearest neighbors fractions...")
            tmp_label_res = []
            fractions_results, tmp_label_res, knn_vicinity = [], [], []
            if self.parallel == False:
                for n in tqdm(self.neighbor_numbers, desc="Neighbors analysis"):
                    result, label_result, knn_perc = self._calculate_fractions_for_subset(indices, [n], id_affinity_label)
                    fractions_results.append(result)
                    tmp_label_res.append(label_result)
                    knn_vicinity.append(knn_perc)
            # Use joblib.Parallel to parallelize the computation
            if self.parallel == True:
                results = Parallel(n_jobs=10)(
                    delayed(analyze_neighbors)(self, indices, n, id_affinity_label) 
                    for n in tqdm(self.neighbor_numbers, desc="Neighbors analysis")
                )
                fractions_results, tmp_label_res, knn_vicinity = [], [], []
                # Unpack the results
                for result, label_result, knn_perc in results:
                    fractions_results.append(result)
                    tmp_label_res.append(label_result)
                    knn_vicinity.append(knn_perc)
            print(f' NN running time {time.time() - t_NN}')
            knn_vicinity = np.array(knn_vicinity)
            # We create an empty DataFrame and fill it with the results
            label_idx = sorted(set(key for dic in tmp_label_res for key in dic.keys()))
            result_labels_df = pd.DataFrame(index=label_idx)  # Create index from all possible labels
            for i, result in enumerate(tmp_label_res):  # i should be the respective NN
                for label, percentage in result.items():
                    result_labels_df.loc[label, i] = percentage  # Set the percentage in the correct position
            print(result_labels_df)
        self.label_results = result_labels_df
        self.knn_vicinity = knn_vicinity
        
        return fractions_results, indices, distances, indices_affinity_mat, lev_mat, id_affinity_label
    
    @profile   
    def calculate_fractions_for_data_FAISS(self):
        self.df= self.df.loc[self.id_index]
        self.df['affinity']= self.df[self.colname_affinity]
        self.df['junction_aa']= self.df[self.colname_junction]
        # self.neigh = NearestNeighbors(metric=self.metric, n_jobs=5)
        # self.neigh.fit(list(self.df['embedding']))
        self.neigh.fit(self.embedding)

        self.df['affinity'] = self.df[self.colname_affinity]
        self.df['junction_aa'] = self.df[self.colname_junction]





        # Compute the nearest neighbors for the maximum number of neighbors needed
        print("Computing KNN ...")

        
        xb = self.embedding  
        n_samples, dim = xb.shape
        k = max(self.neighbor_numbers)   # how many neighbors

        # 1) tell FAISS how many threads to use
        #    (this sets OpenMP threads for FlatL2 brute-force)
        faiss.omp_set_num_threads(5)

        # 2) build or reload a disk-backed “FlatL2” index
        idx_file = "faiss_flat_L2.index"
        if not os.path.exists(idx_file):
            flat = faiss.IndexFlatL2(dim)     # exact L2
            flat.add(xb)                      # copies all xb into RAM once
            faiss.write_index(flat, idx_file)

        # now reopen with mmap so the OS page-cache backs it,
        # no second full copy in RAM at search-time
        index = faiss.read_index(idx_file, faiss.IO_FLAG_MMAP)

        # 3) search exactly as before
        distances, indices = index.search(xb, k)

        # 4) cast into the same dtypes sklearn would give you
        #    assuming xb.dtype is float32 → sklearn euclidean distances
        #    on float32 inputs also returns float32
        distances = distances.astype(xb.dtype, copy=False)
        indices   = indices.astype(np.int64, copy=False)
        # analyze_memory_usage("after KNN")
        # if you converted embeddings into a list or array:

        fractions_results = []
        indices_affinity = self.df.iloc[indices.flatten()]['affinity'].values
        # Redimension affinity values array to correspond to indices shape
        id_affinity_label = self.df.loc[self.id_index, 'affinity']
        indices_affinity_mat = indices_affinity.reshape(indices.shape)
        t_lev = time.time()        
        print("Computing Levenshtein distances...")
        lev_mat = []
        # tqdm progress bar for Levenshtein distance computation
        for idx in tqdm(range(len(self.id_index)), desc="Levenshtein distances"):
            seq_index = self.id_index[idx]
            NN_index = indices[idx]
            # Use .loc to access the row by index and get the 'junction_aa' column value
            ref_junction_aa = self.df.loc[seq_index]['junction_aa']
            # Use .iloc to access rows by integer-location based indexing and get the 'junction_aa' column values
            nn_junction_aa = self.df.iloc[NN_index]['junction_aa']
            # Compute Levenshtein distances between the reference sequence and its K nearest neighbors
            lev_mat.append(compute_levenshtein(ref_junction_aa, nn_junction_aa))
        lev_mat = np.array(lev_mat)
        print(f' LEV running time {time.time() - t_lev}')
        
        if self.skip_neighbors_analysis:
            print("Skipping Neighbors analysis...")
            fractions_results = [np.nan] * len(self.neighbor_numbers)
            tmp_label_res = {label: [np.nan] * len(self.neighbor_numbers) for label in id_affinity_label.unique()}
            knn_vicinity = np.zeros((len(self.neighbor_numbers), len(self.id_index)))
            result_labels_df = pd.DataFrame()
        else:
            t_NN = time.time()
            print("Calculating nearest neighbors fractions...")
            tmp_label_res = []
            fractions_results, tmp_label_res, knn_vicinity = [], [], []
            if self.parallel == False:
                for n in tqdm(self.neighbor_numbers, desc="Neighbors analysis"):
                    result, label_result, knn_perc = self._calculate_fractions_for_subset(indices, [n], id_affinity_label)
                    fractions_results.append(result)
                    tmp_label_res.append(label_result)
                    knn_vicinity.append(knn_perc)
            # Use joblib.Parallel to parallelize the computation
            if self.parallel == True:
                results = Parallel(n_jobs=10)(
                    delayed(analyze_neighbors)(self, indices, n, id_affinity_label) 
                    for n in tqdm(self.neighbor_numbers, desc="Neighbors analysis")
                )
                fractions_results, tmp_label_res, knn_vicinity = [], [], []
                # Unpack the results
                for result, label_result, knn_perc in results:
                    fractions_results.append(result)
                    tmp_label_res.append(label_result)
                    knn_vicinity.append(knn_perc)
            print(f' NN running time {time.time() - t_NN}')
            knn_vicinity = np.array(knn_vicinity)
            # We create an empty DataFrame and fill it with the results
            label_idx = sorted(set(key for dic in tmp_label_res for key in dic.keys()))
            result_labels_df = pd.DataFrame(index=label_idx)  # Create index from all possible labels
            for i, result in enumerate(tmp_label_res):  # i should be the respective NN
                for label, percentage in result.items():
                    result_labels_df.loc[label, i] = percentage  # Set the percentage in the correct position
            print(result_labels_df)
        self.label_results = result_labels_df
        self.knn_vicinity = knn_vicinity
        
        return fractions_results, indices, distances, indices_affinity_mat, lev_mat, id_affinity_label
    
    
    

    def calculate_fractions_for_data(self):
        self.df= self.df.loc[self.id_index]
        self.df['affinity']= self.df[self.colname_affinity]
        self.df['junction_aa']= self.df[self.colname_junction]
        self.neigh = NearestNeighbors(metric=self.metric, n_jobs=3)
        # self.neigh.fit(list(self.df['embedding']))
        self.neigh.fit(self.embedding)

        self.df['affinity'] = self.df[self.colname_affinity]
        self.df['junction_aa'] = self.df[self.colname_junction]
        # Compute the nearest neighbors for the maximum number of neighbors needed
        print("Computing KNN ...")

        
        # distances, indices = self.neigh.kneighbors(self.df.loc[self.id_index]['embedding'].tolist(), n_neighbors=max(self.neighbor_numbers))
        distances, indices = self.neigh.kneighbors(self.embedding, n_neighbors=max(self.neighbor_numbers))  #embedding are already sampled in the correct order (filtering step in pipeline)
        # analyze_memory_usage("after KNN")
        # if you converted embeddings into a list or array:

        fractions_results = []
        indices_affinity = self.df.iloc[indices.flatten()]['affinity'].values
        # Redimension affinity values array to correspond to indices shape
        id_affinity_label = self.df.loc[self.id_index, 'affinity']
        indices_affinity_mat = indices_affinity.reshape(indices.shape)

        # lev_mat = []
        # # tqdm progress bar for Levenshtein distance computation
        # for idx in tqdm(range(len(self.id_index)), desc="Levenshtein distances",
        #                 file=sys.stdout,     # force stdout carriage-return updates
        #                 leave=False          # clear bar on completion
        #                 ):
        #     seq_index = self.id_index[idx]
        #     NN_index = indices[idx]
        #     # Use .loc to access the row by index and get the 'junction_aa' column value
        #     ref_junction_aa = self.df.loc[seq_index]['junction_aa']
        #     # Use .iloc to access rows by integer-location based indexing and get the 'junction_aa' column values
        #     nn_junction_aa = self.df.iloc[NN_index]['junction_aa']
        #     # Compute Levenshtein distances between the reference sequence and its K nearest neighbors
        #     lev_mat.append(compute_levenshtein(ref_junction_aa, nn_junction_aa))
        # lev_mat = np.array(lev_mat)
        
        t_lev = time.time()        
        print("Computing Levenshtein distances...")
        # 1) One-time extraction of all junction strings into a plain list
        junctions = self.df['junction_aa'].tolist()
        n_queries = len(self.id_index)
        k = max(self.neighbor_numbers)

        # 3) Compute all Levenshtein distances with a list comprehension
        lev_mat = [
            [
                RapidfuzzLevenshtein.distance(junctions[i], junctions[nn_idx])
                for nn_idx in indices[i]
            ]
            for i, seq_idx in enumerate(self.id_index)
        ]
        lev_mat = np.array(lev_mat)

        print(f' LEV running time {time.time() - t_lev}')


        # #iteration 3
        # junctions = self.df['junction_aa'].tolist()

        # t0 = time.time()
        # lev_mat = np.array([
        #     [RapidfuzzLevenshtein.distance(junctions[seq_idx], junctions[nn_idx])
        #     for nn_idx in indices[i]]
        #     for i, seq_idx in enumerate(self.id_index)
        # ], dtype=int)
        # print(f"LEV running time {time.time() - t0:.2f}s")

  
        if self.skip_neighbors_analysis:
            print("Skipping Neighbors analysis...")
            fractions_results = [np.nan] * len(self.neighbor_numbers)
            tmp_label_res = {label: [np.nan] * len(self.neighbor_numbers) for label in id_affinity_label.unique()}
            knn_vicinity = np.zeros((len(self.neighbor_numbers), len(self.id_index)))
            result_labels_df = pd.DataFrame()
        else:
            t_NN = time.time()
            print("Calculating nearest neighbors fractions...")
            tmp_label_res = []
            fractions_results, tmp_label_res, knn_vicinity = [], [], []
            if self.parallel == False:
                for n in tqdm(self.neighbor_numbers, desc="Neighbors analysis"):
                    result, label_result, knn_perc = self._calculate_fractions_for_subset(indices, [n], id_affinity_label)
                    fractions_results.append(result)
                    tmp_label_res.append(label_result)
                    knn_vicinity.append(knn_perc)
            # Use joblib.Parallel to parallelize the computation
            if self.parallel == True:
                results = Parallel(n_jobs=10)(
                    delayed(analyze_neighbors)(self, indices, n, id_affinity_label) 
                    for n in tqdm(self.neighbor_numbers, desc="Neighbors analysis")
                )
                fractions_results, tmp_label_res, knn_vicinity = [], [], []
                # Unpack the results
                for result, label_result, knn_perc in results:
                    fractions_results.append(result)
                    tmp_label_res.append(label_result)
                    knn_vicinity.append(knn_perc)
            print(f' NN running time {time.time() - t_NN}')
            knn_vicinity = np.array(knn_vicinity)
            # We create an empty DataFrame and fill it with the results
            label_idx = sorted(set(key for dic in tmp_label_res for key in dic.keys()))
            result_labels_df = pd.DataFrame(index=label_idx)  # Create index from all possible labels
            for i, result in enumerate(tmp_label_res):  # i should be the respective NN
                for label, percentage in result.items():
                    result_labels_df.loc[label, i] = percentage  # Set the percentage in the correct position
            print(result_labels_df)
        self.label_results = result_labels_df
        self.knn_vicinity = knn_vicinity
        
        return fractions_results, indices, distances, indices_affinity_mat, lev_mat, id_affinity_label
    
    




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
        
    def perc_Euclidian_radius(self,  distance_thresholds):
        quantiles=(0.01, 0.95)
        all_distances = self.NN_dist.flatten()
        min_dist = np.percentile(all_distances, quantiles[0] * 100)
        max_dist = np.percentile(all_distances, quantiles[1] * 100)  
        # # Create evenly spaced thresholds between min_dist and max_dist
        # if len(self.df) < 2500:
        #     distance_thresholds = np.concatenate([
        #     np.linspace(min_dist, min_dist + (max_dist - min_dist) * 0.1, 10),
        #     np.linspace(min_dist + (max_dist - min_dist) * 0.1, max_dist, 5)
        #     ])
        # else:
        distance_thresholds = np.linspace(min_dist, max_dist, 15)
        self.lin_density_thresholds = np.linspace(min_dist, max_dist, 7)
        print(f"Computed threshold are {distance_thresholds} !")
        results, LD1_res, LD2_res = [], [], []
        res_df = pd.DataFrame(columns=[f'EU_{i}' for i in distance_thresholds])
        mean_num_points = []
        summary_data = []
        perc_list = []
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
            #generate a dataframe with the results
            perc_list.append(percentages)
            label_perc = {label: np.mean(percentages) for label, percentages in label_percentages.items()}
            label_avg_points = {label: np.mean(percentages) for label, percentages in label_mean_point.items()}
            label_LD=   {label: np.mean(percentages) for label, percentages in label_LD_sim.items()}
            results.append(np.nanmean(percentages))
            LD1_res.append(np.nanmean(LD1_list))
            LD2_res.append(np.nanmean(LD2_list))
            mean_num_points.append(np.mean(num_of_points_within_rad))
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
                summary_row[f'NULLpoints_{label}'] = label_null_points[label] # Store null points
                summary_row[f'NULLPerc_{label}'] = label_null_points[label] / len([x for x in self.ID_labels if x == label]) * 100
                summary_row[f'LD_avgSim_{label}'] = label_LD[label] 
            #Append to the general summmary
            summary_data.append(summary_row)
                # Create a DataFrame with the ID and percentages at each threshold
        
        perc_df = pd.DataFrame(perc_list).T
        perc_df.columns = [f'Thr_{thr}' for thr in list(range(1,16))]

        perc_df['ID'] = self.id_index
        

        perc_df['Affinity'] = self.ID_labels.reset_index(drop=True).astype(str)
        self.perc_df = perc_df

        # Transform the data into long format
        perc_df_long = pd.melt(perc_df, id_vars=['ID', 'Affinity'], var_name='Threshold', value_name='Percentage')
        #remove threshold from the name
        perc_df_long['Threshold']=perc_df_long['Threshold'].str.replace('Thr_','').astype(int)
        self.perc_df_long = perc_df_long
        # Create and store the DataFrame in the class attribute
        self.summary_results = pd.DataFrame(summary_data)
        #Printing results
        for idx,i in enumerate(res_df.columns):
            null_points=sum(res_df[i].isna())
            print(f'{i}:{results[idx]:.4f} ,n_points= {mean_num_points[idx]:.4f}, %null={null_points/len(res_df[i])*100:.4f}, perc_of_LD1= {LD1_res[idx]:.4f}, perc_of_LD2= {LD2_res[idx]:.4f}')
            
        return results,perc_df_long, res_df, mean_num_points, LD1_res, LD2_res        
    


    def compute_adjacency_matrices(self, density_thresholds):
        """
        Computes sparse adjacency matrices for each density threshold.

        Parameters:
        -----------
        self : object
            An object that contains the attributes:
                - NN_id: a 2D numpy array where the first column contains point IDs,
                        and subsequent columns contain the IDs of neighboring points.
                - NN_dist: a 2D numpy array of the same shape as NN_id where each row
                        contains the distances from the point (first column) to its neighbors.
        density_thresholds : list or array-like
            A list of threshold values. For each threshold, only the neighbor distances
            less than or equal to the threshold (excluding self-distance) are considered.

        Returns:
        --------
        adj_mat_list : list
            A list of sparse adjacency matrices (in CSR format), one for each threshold.
        """
        # Precompute row identifiers from the first column of NN_id.
        row_ids_all = self.NN_id[:, 0]
        n = len(row_ids_all)
        
        # Create a mapping array to quickly map point IDs to their row indices.
        # We assume point IDs are non-negative integers.
        unique_ids = np.unique(row_ids_all)
        mapping = np.zeros(unique_ids.max() + 1, dtype=int)
        for idx, row_id in enumerate(row_ids_all):
            mapping[row_id] = idx

        # Prepare a list to hold the resulting adjacency matrices.
        adj_mat_list = [None] * len(density_thresholds)
        print("Computing adjacency matrices...")
        start = time.time()
        # For each density threshold, compute the adjacency matrix.
        for i_density, thr_density in tqdm(enumerate(density_thresholds), total=len(density_thresholds)):
            # Create a boolean mask where NN_dist <= threshold.
            mask = self.NN_dist <= thr_density
            # Exclude self-connections: assume first column corresponds to self.
            mask[:, 0] = False

            # Create an array with repeated row indices for each neighbor.
            repeated_rows = np.repeat(np.arange(n), self.NN_dist.shape[1])
            
            # Flatten the NN_id matrix to align with the flattened mask.
            flattened_neighbor_ids = self.NN_id.flatten()
            mask_flat = mask.flatten()
            
            # Select valid row indices and neighbor IDs using the mask.
            valid_row_indices = repeated_rows[mask_flat]
            valid_neighbor_ids = flattened_neighbor_ids[mask_flat]
            
            # Map valid neighbor IDs to row indices using the mapping array.
            valid_neighbor_indices = mapping[valid_neighbor_ids]
            
            # To create a symmetric adjacency matrix, add both (i, j) and (j, i).
            all_rows = np.concatenate([valid_row_indices, valid_neighbor_indices])
            all_cols = np.concatenate([valid_neighbor_indices, valid_row_indices])
            data = np.ones(all_rows.shape[0], dtype=int)
            
            # Build the sparse matrix in COO format and convert to CSR.
            adjacency_matrix = coo_matrix((data, (all_rows, all_cols)), shape=(n, n)).tocsr()
            adj_mat_list[i_density] = adjacency_matrix
        end = time.time()
        print(f"Time elapsed: {end - start} seconds")
        return adj_mat_list ,row_ids_all





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


def prepare_data_for_plotting_LD_MAT(matrix_path,metadata_df,vicinity_df,  id_index_sample , max_LD, junction_aa_col='junction_aa', affinity_col='affinity'):
    # metadata_df['affinity'] = metadata_df[affinity_col]
    # metadata_df['junction_aa'] = metadata_df[junction_aa_col]
    # if id_index_sample is None:
    #     id_index_sample = df.index
    
  
    print( f' number of samples {len(id_index_sample)}')
    
    def extract_subsample(distance_matrix, indexes):
        # if isinstance(distance_matrix, torch.Tensor):
            # indexes_tensor = torch.tensor(indexes, dtype=torch.long)
            # subsample = distance_matrix[indexes_tensor][:, indexes_tensor]
        # Convert indexes to a NumPy array for fancy indexing
        indexes_array = np.array(indexes, dtype=np.int64)
        # Extract the subsample by indexing only the needed parts
        subsample = distance_matrix[indexes_array][:, indexes_array]
        return subsample

    def count_values_greater_than_X(tensor, ld_thr, labels):
        # Convert tensor to numpy array if it's a PyTorch tensor
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.numpy()
        # Create a boolean mask where condition is satisfied
        condition = (0 < tensor) & (tensor <= ld_thr)
        # Retrieve column indices for each row where the condition holds true
        indices_per_row = [np.where(row_mask)[0] for row_mask in condition]
        res = []
        num_points = []
        for i, idx in enumerate(indices_per_row):
            if len(idx) > 0:
                res.append(sum(labels[idx] == labels[i]) / len(idx))
                num_points.append(len(idx))
            else:
                res.append(np.nan)
                num_points.append(0)
        return res, num_points

    def group_results_by_binding_label(results, labels):
        grouped_results = {}
        for result, label in zip(results, labels):
            if label not in grouped_results:
                grouped_results[label] = []
            grouped_results[label].append(result)
        return grouped_results

    # metadata_df=pd.read_csv( metadata_path, sep='\t')
    print("Matrix Loading...")
    # LD_dist_mat=torch.load(matrix_path)
    arr = np.load(matrix_path, mmap_mode='r')
    shape = arr.shape
    dtype = arr.dtype
    LD_dist_mat = np.memmap(matrix_path, dtype=dtype, mode='r', shape=shape)
    if isinstance(vicinity_df, str) and vicinity_df == "all":
        matrix_indexes = metadata_df
        print("All matrix indexes loaded, no vicinity object provided")
    elif isinstance(vicinity_df, pd.DataFrame):
        vicinity_df = vicinity_df.loc[id_index_sample]
        sample_id = vicinity_df['id'].values
        matrix_indexes = metadata_df[metadata_df['sequence_id'].isin(vicinity_df['sequence_id'])]
    else:
        raise ValueError("vicinity_df must be either a DataFrame or the string 'all' ")
    print("Matrix Loaded")
    # Match the matrix indexes with the binding labels
    binding_labels = matrix_indexes['binding_label'].values
    LD_dist_mat_sample = extract_subsample(LD_dist_mat, matrix_indexes.index)

    print("Matrix indexes matched with binding labels")
    print("Calculating vicinity scores...") 
    tensor = LD_dist_mat_sample
    results = []
    for LD_thr in range(1, max_LD + 1):
        vicinity, num_points = count_values_greater_than_X(tensor, LD_thr, binding_labels)
        vicinity_by_class = group_results_by_binding_label(vicinity, binding_labels)
        num_points_by_class = group_results_by_binding_label(num_points, binding_labels)
        results.append((LD_thr, vicinity, vicinity_by_class, num_points, num_points_by_class,binding_labels, matrix_indexes,sample_id))

        # Create a DataFrame for the results
    
    # Create a DataFrame for the results
    results_df = pd.DataFrame(results, columns=['LD_thr', 'vicinity', 'vicinity_by_class', 'num_points', 'num_points_by_class', 'binding_labels', 'matrix_indexes','sample_id'])
    results_df

    # LD_dist_mat=torch.load('/doctorai/niccoloc/tz_LD_dist_mat_HB_LB.pt')
    # np.save('/doctorai/niccoloc/tz_LD_dist_mat_HB_LB.npy', LD_dist_mat.numpy())

    # Initialize lists to store the summary statistics
    mean_vicinity = []
    mean_num_points = []
    perc_nan = []
    mean_num_points_HB = []
    mean_num_points_LB = []
    mean_vicinity_HB = []
    mean_percentage_LB = []
    perc_nan_HB = []
    perc_nan_LB = []
    perc_nan_HB_CORRECT = []
    num_nan_HB=[]

    # Calculate summary statistics for each LD threshold
    # for LD_thr, counts, vicinity_by_class in results[0]:

    # LD_thr= results[0][0]
    # vicinity= results[0][1]
    # vicinity_by_class= results[0][2]
    # num_points= results[0][3]
    # num_points_by_class = results[0][4]

    print("Calculating summary statistics...")
    for LD_thr, vicinity, vicinity_by_class, num_points, num_points_by_class, _,_ ,_ in tqdm(results, desc="Calculating summary statistics"):
        # Calculate mean percentage and number of points
        vicinity_scores = [count for count in vicinity if not np.isnan(count)]
        mean_vicinity.append(np.mean(vicinity_scores))
        mean_num_points.append(np.mean(num_points))
        perc_nan.append(np.isnan(vicinity).mean())

        # Calculate statistics for Hb group
        hb_counts = vicinity_by_class.get('hb', [])
        num_points_HB = num_points_by_class.get('hb', [])

        vicinity_score_HB = [count for count in hb_counts if not np.isnan(count)]
        mean_vicinity_HB.append(np.mean(vicinity_score_HB))
        mean_num_points_HB.append(np.mean(num_points_HB))
        tot_nan_HB = np.isnan(hb_counts).sum()
        num_nan_HB.append( tot_nan_HB)
        perc_nan_HB.append(np.isnan(hb_counts).mean())
        perc_nan_HB_CORRECT.append(tot_nan_HB / len(hb_counts))

        # Calculate statistics for Lb group
        lb_counts = vicinity_by_class.get('lb', [])
        num_points_LB = num_points_by_class.get('lb', [])

        vicinity_score_LB = [count for count in lb_counts if not np.isnan(count)]
        mean_percentage_LB.append(np.mean(vicinity_score_LB))
        mean_num_points_LB.append(np.mean(num_points_LB))
        perc_nan_LB.append(np.isnan(lb_counts).mean())

    # Threshold,Mean_Percentage,Mean_LD1,Mean_LD2,Mean_Num_Points,Percentage_Null,Perc_lb,AvgPoints_lb,NULLPerc_lb,LD_avgSim_lb

    # Create a DataFrame with the summary statistics
    summary_df = pd.DataFrame({
        'Threshold': [LD_thr for LD_thr, _, _, _ , _ ,_ , _ ,_ in results],

        'Mean_Num_Points': mean_num_points,    
        'Mean_Percentage': mean_vicinity,
        'Percentage_Null': perc_nan,

        'Perc_hb': mean_vicinity_HB,
        'AvgPoints_hb':mean_num_points_HB,
        'num_nan_HB': num_nan_HB,
        'NULLPerc_hb': perc_nan_HB,
        'NULLPerc_hb_corrected': perc_nan_HB_CORRECT,


        'Perc_lb': mean_percentage_LB,
        'AvgPoints_lb': mean_num_points_LB,
        'NULLPerc_lb': perc_nan_LB
    })

    # Display the DataFrame
    print(summary_df)

    # Display the DataFrame
    # print(results_df) 


    
    return results_df, summary_df 
    


def prepare_data_for_plotting(df, LD_dist, num_batches=100, sampled_indices=None, junction_aa_col='junction_aa', affinity_col='affinity', num_jobs=32):
    df['affinity'] = df[affinity_col]
    df['junction_aa'] = df[junction_aa_col]
    
    if sampled_indices is None:
        sampled_indices = df.index
    
    num_samples = len(sampled_indices)
    
    # Split the DataFrame into batches for parallel processing
    batches = np.array_split(sampled_indices, num_batches)
    
    # Initialization of final results arrays
    results = np.zeros((num_samples, LD_dist))
    num_of_points = np.zeros((num_samples, LD_dist))

    # Function to process each batch of indices
    def process_batch(sample_indices):
        batch_results = np.zeros((len(sample_indices), LD_dist))
        batch_num_points = np.zeros((len(sample_indices), LD_dist))
        
        for row, index in enumerate(sample_indices):
            initial_affinity = df.loc[index, 'affinity']
            initial_seq = df.loc[index, 'junction_aa']
            lev_dists = compute_levenshtein(initial_seq, df.iloc[1:]['junction_aa'])  # Compute Levenshtein distances
            results_row = np.zeros(LD_dist)
            num_points_row = np.zeros(LD_dist)
            for lev_dist in range(1, LD_dist + 1):
                indices_at_dist = [i for i, x in enumerate(lev_dists) if 0 < x <= lev_dist]
                if indices_at_dist:
                    affinities_at_dist = df.iloc[indices_at_dist]['affinity']
                    percentage = sum(affinities_at_dist == initial_affinity) / len(affinities_at_dist)
                    results_row[lev_dist - 1] = percentage
                    num_points_row[lev_dist - 1] = len(affinities_at_dist)
                else:
                    results_row[lev_dist - 1] = np.nan
                    num_points_row[lev_dist - 1] = 0
            batch_results[row, :] = results_row
            batch_num_points[row, :] = num_points_row
        return batch_results, batch_num_points
    start_time = time.time()
    # Parallel processing of batches
    from tqdm_joblib import tqdm_joblib
    with tqdm_joblib(desc="Parallel LD", total=len(batches)) as progress_bar:
        processed_batches = Parallel(n_jobs= num_jobs )(delayed(process_batch)(batch) for batch in batches)
    seq_duration = time.time() - start_time
    print(f"Par execution time: {seq_duration:.2f} seconds")
    # Combine the results from each batch
    start_idx = 0
    for batch_results, batch_num_points in processed_batches:
        batch_size = batch_results.shape[0]
        results[start_idx:start_idx + batch_size, :] = batch_results
        num_of_points[start_idx:start_idx + batch_size, :] = batch_num_points
        start_idx += batch_size
    # Combine results and num_of_points into a single DataFrame
    columns = [f'LD_{i}' for i in range(1, LD_dist + 1)]
    df_results = pd.DataFrame(results, columns=[f'Perc_{col}' for col in columns]) # PERCENTAGE OF Points with SAME LABLE (vicinity score)
    df_num_points = pd.DataFrame(num_of_points, columns=[f'Num_{col}' for col in columns])
    print(df_results)
    print(df_num_points)
    # Merge into one DataFrame
    affinities = df.loc[sampled_indices, 'affinity']
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
    command = f"/doctorai/niccoloc/airr_atlas/scripts/Vicinity_code/run_ggplot_vicinity.sh {input_ED} {input_LD} {output_path}"
    # Execute the command
    print(command)
    try:
        # Using shell=True to handle the command chain correctly
        output = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        print("R script output:", output.stdout)
    except subprocess.CalledProcessError as e:
        print("Error running R script:", e.stderr)


#previous code

def prepare_data_for_plotting_debug(df, LD_dist , sampled_indices= None, junction_aa_col='junction_aa', affinity_col='affinity'):
    df['affinity']= df[affinity_col]
    df['junction_aa']= df[junction_aa_col]
    num_samples = len(sampled_indices)
    if sampled_indices is None:
      sampled_indices =df['id']
      num_samples = len(sampled_indices)
    # sampled_indices = np.random.choice(df.index, size=num_samples, replace=False)
    # Inizializzazione degli array per i risultati
    results = np.zeros((num_samples, LD_dist))
    results_nan = np.zeros((num_samples, LD_dist))
    num_of_points = np.zeros((num_samples, LD_dist))
    affinities = []
    # Iterazione su ciascun indice campionato
    start_time = time.time()
    for row, index in tqdm(enumerate(sampled_indices), total=num_samples, desc="Processing samples"):
    # for row, index in enumerate(sampled_indices):
        initial_affinity = df.loc[index, 'affinity']
        initial_seq = df.loc[index, 'junction_aa']
        affinities.append(initial_affinity)
        # Calcolo delle distanze di Levenshtein
        lev_dists = compute_levenshtein(initial_seq, df.iloc[1:]['junction_aa'])
        for lev_dist in range(1, LD_dist + 1):
            indices_at_dist = [i for i, x in enumerate(lev_dists) if  x== lev_dist]
            if row < 3:
                print(f'current lev thr {lev_dist}')
                print(f'mean lev thr {statistics.mean([x for i, x in enumerate(lev_dists) if  x == lev_dist])}')
                print(f'indices_at_dist are {len(indices_at_dist)}')
                value_counts = Counter(indices_at_dist)
                # print(f'distribution of {value_counts}')
            if indices_at_dist:
                affinities_at_dist = df.iloc[indices_at_dist]['affinity']
                if row < 3:
                    print(f'same label at LD {sum(affinities_at_dist == initial_affinity)}')
                percentage = sum(affinities_at_dist == initial_affinity) / len(affinities_at_dist)
                results[row, lev_dist - 1] = percentage
                num_of_points[row, lev_dist - 1] = len(affinities_at_dist)
            else:
                if lev_dist==1:
                    results_nan[row,lev_dist-1] =np.nan
                results[row, lev_dist - 1] = np.nan  # Uso NaN per le distanze senza sequenze
                num_of_points[row, lev_dist - 1] = 0
    seq_duration = time.time() - start_time
    print(f"Seq execution time: {seq_duration:.2f} seconds")
    nan_count = np.isnan(results).sum()
    nan_count2 = np.isnan(results_nan).sum()
    print(f'Number of total NaNs in results: {nan_count}')            
    print(f'Number of LD 1 NaNs in results: {nan_count2}')            
    # Combine results and num_of_points into a single DataFrame
    columns = [f'LD_{i}' for i in range(1, LD_dist + 1)]
    df_results = pd.DataFrame(results, columns=[f'Perc_{col}' for col in columns]) # PERCENTAGE OF Points with SAME LABLE (vicinity score)
    df_num_points = pd.DataFrame(num_of_points, columns=[f'Num_{col}' for col in columns])
    print(df_results)
    print(df_num_points)
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
  

def prepare_data_for_plotting_sequential(df, LD_dist , sampled_indices= None, junction_aa_col='junction_aa', affinity_col='affinity'):
    df['affinity']= df[affinity_col]
    df['junction_aa']= df[junction_aa_col]
    num_samples = len(sampled_indices)
    if sampled_indices is None:
      sampled_indices =df['id']
      num_samples = len(sampled_indices)
    # sampled_indices = np.random.choice(df.index, size=num_samples, replace=False)
    # array initialization
    results = np.zeros((num_samples, LD_dist))
    results_nan = np.zeros((num_samples, LD_dist))
    num_of_points = np.zeros((num_samples, LD_dist))
    affinities = []
    # iteration on each sampled index
    for row, index in tqdm(enumerate(sampled_indices), total=num_samples, desc="Processing samples"):
    # for row, index in enumerate(sampled_indices):
        initial_affinity = df.loc[index, 'affinity']
        initial_seq = df.loc[index, 'junction_aa']
        affinities.append(initial_affinity)
        # Calcolo delle distanze di Levenshtein
        lev_dists = compute_levenshtein(initial_seq, df.iloc[1:]['junction_aa'])
        for lev_dist in range(1, LD_dist + 1):
            indices_at_dist = [i for i, x in enumerate(lev_dists) if  0< x <= lev_dist]
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
    print(df_results)
    print(df_num_points)
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
    



