import torch
import torch.nn as nn
import torch.nn.functional as F # We will use this
from torch.utils.data import Dataset, DataLoader
import requests
import math

# --- 0. Configuration (Identical to before) ---
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_TRAIN_EPOCHS = 10 # Increase for better results
TOKEN_EMBED_DIM = 32
CHAR_EMBED_DIM = 16
NUM_TIMESTEPS = 200

# --- 1. Load and Process Data (Identical to before) ---
print("Step 1: Loading and processing the Tiny Shakespeare dataset...")
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)
text = response.text
chars = sorted(list(set(text)))
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}
VOCAB_SIZE = len(chars)
all_tokens = [text[i:i+4] for i in range(len(text) - 4)]
unique_tokens = sorted(list(set(all_tokens)))
token_to_idx = {tok: i for i, tok in enumerate(unique_tokens)}
NUM_UNIQUE_TOKENS = len(unique_tokens)
token_indices = []
char_indices = []
for token_str in unique_tokens:
    token_indices.append(token_to_idx[token_str])
    chars_as_ints = [char_to_int[ch] for ch in token_str]
    char_indices.append(chars_as_ints)

class CharTokenDataset(Dataset):
    def __init__(self, token_indices, char_indices):
        self.token_indices = torch.tensor(token_indices, dtype=torch.long)
        self.char_indices = torch.tensor(char_indices, dtype=torch.long)
    def __len__(self): return len(self.token_indices)
    def __getitem__(self, idx): return self.token_indices[idx], self.char_indices[idx]

dataset = CharTokenDataset(token_indices, char_indices)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print("Data preparation complete.")


# --- 2. Model Architecture (Identical to before) ---
print("\nStep 2: Building the model architecture...")
class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embed = math.log(10000) / (half_dim - 1)
        embed = torch.exp(torch.arange(half_dim, device=device) * -embed)
        embed = t[:, None] * embed[None, :]
        return torch.cat((embed.sin(), embed.cos()), dim=-1)

class Block(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        self.proj = nn.Linear(size_in, size_out)
        self.act = nn.SiLU()
    def forward(self, x): return self.act(self.proj(x))

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed = TimestepEmbedding(CHAR_EMBED_DIM * 4)
        self.token_embed = nn.Linear(TOKEN_EMBED_DIM, CHAR_EMBED_DIM * 4)
        self.block1 = Block(CHAR_EMBED_DIM * 4, 128)
        self.block2 = Block(128, 256)
        self.block3 = Block(256, 128)
        self.output_proj = Block(128, CHAR_EMBED_DIM * 4)
    def forward(self, x, t, condition_embedding):
        x_flat = x.flatten(1)
        t_emb = self.time_embed(t)
        tok_emb = self.token_embed(condition_embedding)
        h = x_flat + t_emb + tok_emb
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        output_flat = self.output_proj(h)
        return output_flat.view_as(x)


# --- 3. Diffusion Logic and Trainer (MODIFIED) ---
print("Step 3: Setting up the diffusion trainer...")

class DiffusionTrainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(NUM_UNIQUE_TOKENS, TOKEN_EMBED_DIM)
        self.char_embedding = nn.Embedding(VOCAB_SIZE, CHAR_EMBED_DIM)
        self.model = SimpleUNet()
        betas = torch.linspace(0.0001, 0.02, NUM_TIMESTEPS)
        alphas = 1. - betas
        self.alpha_cumprod = torch.cumprod(alphas, dim=0)

    def get_noisy_sequence(self, char_embed_seq, t):
        sqrt_alpha_cumprod_t = torch.sqrt(self.alpha_cumprod[t])[:, None, None]
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1. - self.alpha_cumprod[t])[:, None, None]
        noise = torch.randn_like(char_embed_seq)
        noisy_seq = sqrt_alpha_cumprod_t * char_embed_seq + sqrt_one_minus_alpha_cumprod_t * noise
        return noisy_seq, noise

    def forward(self, token_idx, char_idx):
        tok_emb = self.token_embedding(token_idx)
        char_emb = self.char_embedding(char_idx)
        t = torch.randint(0, NUM_TIMESTEPS, (tok_emb.shape[0],), device=tok_emb.device)
        noisy_char_embeddings, actual_noise = self.get_noisy_sequence(char_emb, t)
        predicted_noise = self.model(noisy_char_embeddings, t, tok_emb)
        loss = nn.functional.mse_loss(predicted_noise, actual_noise)
        return loss

    # --- NEW DECODING METHOD ---
    @torch.no_grad()
    def decode(self, token_string):
        """Generates a 4-character sequence from a token string."""
        self.eval() # Set the entire module to evaluation mode
        device = next(self.parameters()).device # Get the device the model is on

        # Get the token's ID and its learned embedding
        token_idx_int = token_to_idx.get(token_string)
        if token_idx_int is None: return "Token not in vocab"
        
        token_idx = torch.tensor([token_idx_int], device=device)
        token_emb = self.token_embedding(token_idx)

        # Start with pure random noise
        char_seq = torch.randn(1, 4, CHAR_EMBED_DIM, device=device)

        # Iteratively denoise
        for t_val in reversed(range(NUM_TIMESTEPS)):
            t = torch.full((1,), t_val, device=device)
            predicted_noise = self.model(char_seq, t, token_emb)
            
            # Denoising formula
            alpha_t = (1. - torch.linspace(0.0001, 0.02, NUM_TIMESTEPS)[t]).to(device)
            alpha_cumprod_t = self.alpha_cumprod[t].to(device)
            coeff1 = 1 / torch.sqrt(alpha_t)
            coeff2 = (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)
            char_seq = coeff1 * (char_seq - coeff2 * predicted_noise)

            if t_val > 0:
                beta_t = torch.linspace(0.0001, 0.02, NUM_TIMESTEPS)[t].to(device)
                noise = torch.randn_like(char_seq)
                char_seq += torch.sqrt(beta_t) * noise
        
        # --- THE FIX IS HERE ---
        # Project the final clean embeddings into character logits using the
        # *trained* embedding layer's weights.
        # F.linear is a functional way to do a linear layer.
        logits = F.linear(char_seq, self.char_embedding.weight)
        
        predicted_char_indices = torch.argmax(logits, dim=-1)
        predicted_chars = [int_to_char[idx.item()] for idx in predicted_char_indices[0]]
        return "".join(predicted_chars)


# --- 4. Training Loop (Identical to before) ---
print("\nStep 4: Starting the training loop...")
trainer = DiffusionTrainer()
optimizer = torch.optim.Adam(trainer.parameters(), lr=LEARNING_RATE)
device = "cuda" if torch.cuda.is_available() else "cpu"
trainer.to(device)
print(f"Using device: {device}")
for epoch in range(NUM_TRAIN_EPOCHS):
    total_loss = 0
    for step, (token_idx, char_idx) in enumerate(dataloader):
        optimizer.zero_grad()
        token_idx, char_idx = token_idx.to(device), char_idx.to(device)
        loss = trainer(token_idx, char_idx)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"--- Epoch {epoch+1}/{NUM_TRAIN_EPOCHS} | Average Loss: {avg_loss:.6f} ---")
print("\nTraining complete.")


# --- 5. Inference (MODIFIED AND SIMPLIFIED) ---
print("\nStep 5: Running inference to test the model...")
test_tokens = ["Firs", "good", "surf", "the ", "feit ", "auth"]
for token in test_tokens:
    # Now we just call the method on our trained model instance
    predicted = trainer.decode(token)
    print(f"Input token: '{token}' -> Predicted characters: '{predicted}'")


