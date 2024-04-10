

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
    
    
    
    def __init__( self, df,neighbor_numbers,id_index  ):
        self.df=df
        self.neighbor_numbers= neighbor_numbers
        self.neigh = NearestNeighbors()
        self.neigh.fit(list(self.df['embedding']))
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
        "ID_labels": self.ID_labels
        #"analysis_info": self.paramters
        }
        with open(file_name, 'wb') as file:
            pickle.dump(data_to_save, file)
        
        
    def calculate_fractions_for_data(self):
        # Compute the nearest neighbors for the maximum number of neighbors needed
        distances, indices = self.neigh.kneighbors(self.df.iloc[self.id_index]['embedding'].tolist(), n_neighbors=max(self.neighbor_numbers))
        fractions_results = []
        indices_affinity = self.df.loc[indices.flatten(), 'affinity'].values
        # Redimension affinity values array to corrispond to indices shape
        id_affinty_label=self.df.loc[self.id_index, 'affinity']
        indices_affinity_mat = indices_affinity.reshape(indices.shape)
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
        
    def calculate_percentages_with_precomputed_distances(self, distance_thresholds):
        results, LD1_res, LD2_res = [], [], []
        res_df = pd.DataFrame(columns=[f'EU_{i}' for i in distance_thresholds])
        mean_num_points = []
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
        for idx,i in enumerate(res_df.columns):
            null_points=sum(res_df[i].isna())
            print(f'{i}:{results[idx]:.4f} ,n_points= {mean_num_points_LV[idx]:.4f}, %null={null_points/len(res_df[i])*100:.4f}, perc_of_LD1= {LD1_res[idx]:.4f}, perc_of_LD2= {LD2_res[idx]:.4f}')
        
        return results, res_df, mean_num_points, LD1_res, LD2_res        
        

vicinity_analysis_instance = Vicinity_analysis(df, neighbor_numbers, id_index_sample)
vicinity_analysis_instance.run_analysis()  # This populates the necessary attributes
distance_thresholds = range(7, 20)  # Define your distance thresholds

# Now you can call the new method with these precomputed values
percentages_results11, res_df11, mean_num_points11, LD1_res1, LD2_res1 = vicinity_analysis_instance.calculate_percentages_with_precomputed_distances(distance_thresholds)

for idx, i in enumerate(res_df11.columns):
    null_points = sum(res_df11[i].isna())
    print(f'{i}:{percentages_results11[idx]:.4f} ,n_points= {mean_num_points11[idx]:.4f}, %null={null_points/len(res_df11[i])*100:.4f}, perc_of_LD1= {LD1_res1[idx]:.4f}, perc_of_LD2= {LD2_res1[idx]:.4f}')


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
