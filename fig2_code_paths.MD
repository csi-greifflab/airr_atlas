# Fig 2A:
### Qualitative visualization of global structure in PLM embedding distributions across increasing sample sizes of the AB2, ESM2, OHE representations for HCDR3 BCR sequences

## Code used:
### 1/ Calculation of the embeddings:
**AB2:**
scripts/antiberta2_cdr3.py
drivers/run_antiberta2_cdr3.bash

**ESM2:**
scripts/esm2.py
drivers/run_esm2.bash

**OHE:**
scripts/OHE_script.py
drivers/run_ohe.bash

### 2/ tSNE dimentionality reduction:
scripts/build_tsne.py
run_tsne.bash

### 3/ Visualization:
notebooks/tsne.ipynb

---

# Fig 2B:
### Quantitative assessment of global similarity in PLM embedding distributions across increasing sample sizes of the AB2, ESM2, and OHE sequence representations

### 1/ Calculation of the embeddings:
Same as above

### 2/ Completeness and w2 calculation:
notebooks/completeness.ipynb

---

# Fig 2C:
### Evaluating the quality of clustering replicates (18'000'000 sequences in total) of different donors based on different pipeline modifications

### 1/ Dataset construction:
notebooks/Briney_dataset_construction.ipynb

### 2/ W2 calculation:
scripts/Briney_experiment_W2_calculation.py

### 3/ Comparison:
notebooks/Fig_2D.ipynb

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
notebooks/completeness_tz.ipynb

---

# Supplementary figure 4:
### W₂ distance applied to PLM embeddings reveals a clear separation between experimental immune repertoires (Wang and iReceptor) and randomly generated controls, including those matched for sequence length and positional amino acid frequencies

### 1/ Calculation of the embeddings:
Same as above

### 2/ Pairwise W2 calculations and plots:
notebooks/w2_random_data.ipynb

---

# Supplementary figure 11:
### Evaluating the performance of different pipeline modifications in revealing donor-specific repertoire structure

notebooks/Data_overlap.ipynb

---

# Supplementary figure 12:
### Length bias in PLM embeddings leads to distinct density patterns for sequences of different HCDR3 lengths

### 1/ Calculation of the embeddings:
Same as above

### 2/ Visualization:
notebooks/length_vdj_bias.ipynb

---

# Supplementary figure 13:
### Residual V(D)J gene bias persists in PLM embeddings of HCDR3 sequences

### 1/ Calculation of the embeddings:
Same as above

### 2/ Visualization:
notebooks/length_vdj_bias.ipynb

