# Fig 2A:
### Qualitative visualization of global structure in PLM embedding distributions across increasing sample sizes of the AB2, ESM2, OHE representations for HCDR3 BCR sequences

## Code used:
### 1/ Calculation of the embeddings:
**AB2:**
figures/main/fig_2_w2_distance_matrices/scripts/antiberta2_cdr3.py
figures/main/fig_2_w2_distance_matrices/drivers/run_antiberta2_cdr3.bash

**ESM2:**
figures/main/fig_2_w2_distance_matrices/scripts/esm2.py
figures/main/fig_2_w2_distance_matrices/drivers/run_esm2.bash

**OHE:**
figures/main/fig_2_w2_distance_matrices/scripts/OHE_script.py
figures/main/fig_2_w2_distance_matrices/drivers/run_ohe.bash

### 2/ tSNE dimentionality reduction:
figures/main/fig_2_w2_distance_matrices/scripts/build_tsne.py
figures/main/fig_2_w2_distance_matrices/run_tsne.bash

### 3/ Visualization:
figures/main/fig_2_w2_distance_matrices/notebooks/tsne.ipynb

---

# Fig 2B:
### Quantitative assessment of global similarity in PLM embedding distributions across increasing sample sizes of the AB2, ESM2, and OHE sequence representations

### 1/ Calculation of the embeddings:
Same as above

### 2/ Completeness and w2 calculation:
figures/main/fig_2_w2_distance_matrices/notebooks/fig_2b_completeness.ipynb

---

# Fig 2C:
### Evaluating the quality of clustering replicates (18'000'000 sequences in total) of different donors based on different pipeline modifications

### 1/ Dataset construction:
figures/main/fig_2_w2_distance_matrices/notebooks/Briney_dataset_construction.ipynb

### 2/ W2 calculation:
figures/main/fig_2_w2_distance_matrices/scripts/Briney_experiment_W2_calculation.py

### 3/ Comparison:
figures/main/fig_2_w2_distance_matrices/notebooks/fig_2_comparison.ipynb

---

# Supplementary figure 1:
### PLM embeddings of HCDR3 BCR sequences reveal distinct high-dimensional structure starting from 30,000 sequences

### 1/ Calculation of the embeddings:
Same as above

### 2/ tSNE dimentionality reduction:
Same as above

### 3/ Visualization:
Same as above

---

# Supplementary figure 2:
### Accurate estimation of mean and covariance for TCRβ embeddings using AB2, ESM2, and OHE requires larger sample sizes compared to BCRs

### 1/ Calculation of the embeddings:
Same as above

### 2/ Completeness and w2 calculation:
Same as above

---

# Supplementary figure 3:
### Accurate estimation of mean and covariance for Trastuzumab variant HCDR3 targeting HER2 embeddings using AB2, ESM2, and OHE requires larger sample sizes compared to iReceptor BCRs

### 1/ Calculation of the embeddings:
Same as above

### 2/ Completeness and w2 calculation:
figures/supplementary/fig_s3/notebooks/fig_s3_completeness_tz.ipynb

---

# Supplementary figure 4:
### W₂ distance applied to PLM embeddings reveals a clear separation between experimental immune repertoires (Wang and iReceptor) and randomly generated controls, including those matched for sequence length and positional amino acid frequencies

### 1/ Calculation of the embeddings:
Same as above

### 2/ Pairwise W2 calculations and plots:
figures/supplementary/fig_s4/notebooks/w2_random_data.ipynb

---

# Supplementary figure 11:
### Evaluating the performance of different pipeline modifications in revealing donor-specific repertoire structure

figures/supplementary/fig_s11/notebooks/fig_s11_data_overlap.ipynb

---

# Supplementary figure 12:
### Length bias in PLM embeddings leads to distinct density patterns for sequences of different HCDR3 lengths

### 1/ Calculation of the embeddings:
Same as above

### 2/ Visualization:
figures/supplementary/fig_s12/notebooks/fig_s12_length_vdj_bias.ipynb

---

# Supplementary figure 13:
### Residual V(D)J gene bias persists in PLM embeddings of HCDR3 sequences

### 1/ Calculation of the embeddings:
Same as above

### 2/ Visualization:
notebooks/length_vdj_bias.ipynb

