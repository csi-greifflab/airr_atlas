import torch
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import gc
import numpy as np

def load_tensor(file):
    return torch.load(file)




def save_reduced_tensor(tensor, filename):
    torch.save(tensor, filename)

# if __name__ == "__main__":


    # List of .pt files
path = '/doctorai/userdata/airr_atlas/data/embeddings/giulio/esm2/embeddings_unpooled/prepost_15aa_esm2_embeddings_unpooled_layer_32.pt'
import re
models = ["ab2", "esm2"]
models = ["ab2"]
models = ["esm2"]
n_pca=1500
# N = 10000
# input_dim = 1000
# X = torch.randn(N, input_dim)
bg_tensor = torch.randn(N, input_dim)

# Load bg_tensor
bg_tensor = load_tensor(path).to(torch.float16)

# If you haven't already installed PyTorch, do that first (e.g., pip install torch).
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Suppose you have data: X with shape [N, input_dim] (like N rows, each row is a 1000-dim vector)
# Here we'll just make random data for demonstration

# Create a DataLoader
dataset = TensorDataset(bg_tensor.to(torch.float32))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
input_dim = bg_tensor.shape[1]
hidden_dim = 700
latent_dim = 128



bg_tensor.shape


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        # A small feed-forward network to learn mean and log-variance (logvar)
        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # We'll split the output into mean and logvar
        )
    
    def forward(self, x):
        """
        x is the input to the encoder.
        x shape = [batch_size, input_dim] in this example.

        We pass x through our linear layers to get a single vector of size 2*latent_dim.
        Then we split that vector into 'mu' (the mean) and 'logvar' (the log of the variance).
        """
        out = self.linear(x)
        # `chunk(2, dim=-1)` splits out into 2 parts along the last dimension
        mu, logvar = out.chunk(2, dim=-1)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        # Another small feed-forward network that reconstructs data from latent vectors
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, z):
        """
        z is the latent variable sampled from the encoder's distribution.
        shape = [batch_size, latent_dim].

        We pass z through our linear layers to get an output of shape [batch_size, output_dim].
        """
        out = self.linear(z)
        return out

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * eps
        where eps ~ N(0, 1), std = exp(0.5*logvar).
        This allows gradients to flow back through mu/logvar.
        """
        std = torch.exp(0.5 * logvar)     # exponentiate half of the log variance
        eps = torch.randn_like(std)       # sample from standard normal
        return mu + eps * std

    def forward(self, x):
        """
        1) Get mu and logvar from the encoder.
        2) Sample z using reparameterization trick.
        3) Pass z to the decoder to get reconstruction.
        4) Return both the reconstruction and the mu, logvar (needed for the loss).
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

def vae_loss(recon, x, mu, logvar):
    """
    1) Reconstruction loss measures how far recon is from x.
    2) KL divergence measures how different the learned distribution is
       from the standard normal distribution.
    """
    # We can use MSE or BCE or something else. Here we use MSE for simplicity.
    recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')

    # KL Divergence part of the VAE loss
    # -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_div

def train_vae(model, dataloader, epochs=10, lr=1e-3):
    """
    A simple training loop:
    1) Loops over epochs
    2) For each batch, forward pass, compute loss, backward pass
    3) Update parameters, print out training loss
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            x = batch[0].cuda()  # because batch is (bg_tensor,) as a tuple
            recon, mu, logvar = model(x)
            optimizer.zero_grad()               # Reset gradients
            recon, mu, logvar = model(x)        # Forward pass
            loss = vae_loss(recon, x, mu, logvar)  # Calculate VAE loss
            loss.backward()                     # Backpropagation
            optimizer.step()                    # Update weights
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(name, param.grad.data.norm())


input_dim = bg_tensor.shape[1]
hidden_dim = 700
latent_dim = 128
model = VAE(input_dim, hidden_dim, latent_dim).cuda()

# Train model
train_vae(model, dataloader, epochs=1000, lr=1e-4)







import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )

    def forward(self, x):
        out = self.linear(x)
        mu, logvar = out.chunk(2, dim=-1)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z):
        out = self.linear(z)
        return out


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


# 4) Updated loss function to also return components
def vae_loss(recon, x, mu, logvar):
    mse = F.mse_loss(recon, x, reduction='sum')        # sum over all features
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = mse + kl_div
    return total_loss, mse, kl_div


# 5) Training loop with gradient clipping and separate logs
def train_vae(model, dataloader, epochs=10, lr=1e-4):  # smaller LR
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        total_loss = 0.0
        total_mse = 0.0
        total_kl = 0.0
        print(f"Starting Epoch {epoch+1}/{epochs}...")
        for batch in dataloader:
            x = batch[0].to(device)
            # print(x.dtype)                         # should be torch.float16
            # print(next(model.parameters()).dtype)  # should be torch.float16

            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():  # Enable autocast for mixed precision
                recon, mu, logvar = model(x)
                loss, mse_val, kl_val = vae_loss(recon, x, mu, logvar)
            
            scaler.scale(loss).backward()  # Scale the loss and call backward
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)  # Update weights
            scaler.update()  # Update the scale for next iteration

            total_loss += loss.item()
            total_mse += mse_val.item()
            total_kl += kl_val.item()

        # Averages across the entire data
        avg_loss = total_loss / len(dataloader)
        avg_mse = total_mse / len(dataloader)
        avg_kl = total_kl / len(dataloader)
        
        # Optionally compute per-dimension MSE if you want to see that too
        # Suppose each batch is of size B, with 19000 features
        # We would do total_mse / (N * 19000) across the entire dataset
        # but here we simplify with dataloader-based sums.
        avg_mse2=total_mse / (64 * input_dim)

        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Loss: {avg_loss:.2f}, MSE: {avg_mse:.2f}, KL: {avg_kl:.2f}"
              f"MSE averaged {avg_mse2}")


# Load bg_tensor
bg_tensor = load_tensor(path).to(torch.float16)
# 6) Instantiate and train
input_dim = bg_tensor.shape[1]  # e.g. 19000
hidden_dim = 1000
latent_dim = 256


dataset = TensorDataset(bg_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = VAE(input_dim, hidden_dim, latent_dim).to(device)
print(input_dim, hidden_dim, latent_dim)

train_vae(model, dataloader, epochs=100, lr=1e-4)
train_vae_cpu(model, dataloader, epochs=100, lr=1e-4)








def train_vae_cpu(model, dataloader, epochs=10, lr=1e-4):  # smaller LR
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        total_mse = 0.0
        total_kl = 0.0

        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            
            recon, mu, logvar = model(x)
            loss, mse_val, kl_val = vae_loss(recon, x, mu, logvar)
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()
            total_mse += mse_val.item()
            total_kl += kl_val.item()

        # Averages across the entire data
        avg_loss = total_loss / len(dataloader)
        avg_mse = total_mse / len(dataloader)
        avg_kl = total_kl / len(dataloader)
        
        # Optionally compute per-dimension MSE if you want to see that too
        # Suppose each batch is of size B, with 19000 features
        # We would do total_mse / (N * 19000) across the entire dataset
        # but here we simplify with dataloader-based sums.

        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Loss: {avg_loss:.2f}, MSE: {avg_mse:.2f}, KL: {avg_kl:.2f}")






# Fit UMAP
umap_model = umap.UMAP(n_components=1500, init = 'pca')
bg_tensor_np = bg_tensor.cpu().numpy()  # Convert tensor to numpy array
umap_model.fit(bg_tensor_np)


input_tensor = torch.load('/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis2/esm2/embeddings_unpooled/tz_cdr3_only_100k_esm2_embeddings_unpooled_layer_32.pt')
# Transform tensor_2 using the PCA model fitted on tensor_1
input_tensor.shape
tensor_2_transformed = umap_model.transform(input_tensor)
# Save the transformed tensor
transformed_tensor_filename = '/doctorai/niccoloc/tz_cdr3_only_100k_esm2_embeddings_unpooled_bgUMAP_layer_32.pt'
save_reduced_tensor(torch.tensor(tensor_2_transformed).to(torch.float16), transformed_tensor_filename)
print(f'Saved transformed tensor with shape: {torch.tensor(tensor_2_transformed).shape} to {transformed_tensor_filename}')


joblib.dump(umap_model, 'pca_model.pkl')








import torch
from torch.utils.data import DataLoader, TensorDataset
import umap

# Suppose you have data: X with shape [N, input_dim] (like N rows, each row is a 1000-dim vector)
# Here we'll just make random data for demonstration

# Create a DataLoader

dataset = TensorDataset(bg_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
input_dim = bg_tensor.shape[1]
hidden_dim = 700
latent_dim = 128
model = VAE(input_dim, hidden_dim, latent_dim)

# Train model
train_vae(model, dataloader, epochs=1000, lr=1e-4)
