import torch
import torch.nn as nn
import torch.optim as optim
import random


class TokenToCharTransformer(nn.Module):
    def __init__(self, token_vocab_size, char_vocab_size, token_embedding_dim, char_embedding_dim, hidden_dim, num_layers, nhead):
        super().__init__()

        self.token_embedding = nn.Embedding(token_vocab_size, token_embedding_dim)
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim)

        self.input_dim = token_embedding_dim  # Match token embedding and char embedding sizes
        assert token_embedding_dim == char_embedding_dim, "Token and char embedding dims must match for this design."

        self.positional_encoding = PositionalEncoding(self.input_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            batch_first=False
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.input_projection = nn.Linear(self.input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, char_vocab_size)

    def forward(self, token_ids, char_input_ids):
        """
        token_ids: (batch_size,)
        char_input_ids: (seq_len, batch_size)
        """
        batch_size = token_ids.size(0)
        seq_len = char_input_ids.size(0)

        # Token embedding (1, batch_size, embed_dim)
        token_emb = self.token_embedding(token_ids).unsqueeze(0)

        # Character input embeddings (seq_len, batch_size, embed_dim)
        char_emb = self.char_embedding(char_input_ids)

        # Combine token embedding and char embeddings into input sequence
        decoder_input = torch.cat([token_emb, char_emb], dim=0)  # (seq_len + 1, batch_size, embed_dim)

        decoder_input = self.input_projection(decoder_input)  # Project to hidden_dim
        decoder_input = self.positional_encoding(decoder_input)  # Add positional encoding

        # Create causal mask for autoregressive behavior (excluding token embedding)
        tgt_seq_len = seq_len + 1  # token + chars
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(decoder_input.device)

        memory = None  # No encoder memory; token info is in input

        transformer_output = self.transformer_decoder(
            tgt=decoder_input,
            memory=decoder_input,  # Self-conditioning decoder
            tgt_mask=tgt_mask
        )  # (seq_len + 1, batch_size, hidden_dim)

        logits = self.output_layer(transformer_output[1:])  # Skip the token embedding position for output

        return logits  # (seq_len, batch_size, char_vocab_size)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (seq_len, batch_size, d_model)
        """
        return x + self.pe[:x.size(0)]


# ===================
# Dataset Preparation
# ===================

with open("input.txt", "r") as file:
    file_content = file.read()

words = file_content.split()
token_vocab_size = len(words)

# Build token vocabulary
token2id = {word: idx for idx, word in enumerate(words)}
id2token = {idx: token for token, idx in token2id.items()}

# Build character vocabulary
chars = set("".join(words))
char_list = ["<EOS>", "<SOS>"] + sorted(chars)
char2id = {ch: idx for idx, ch in enumerate(char_list)}
id2char = {idx: ch for ch, idx in char2id.items()}

char_vocab_size = len(char2id)


def word_to_char_ids(word):
    return [char2id[c] for c in word]


dataset = []
for word in words:
    token_id = token2id[word]
    char_ids = word_to_char_ids(word)

    char_input = [char2id["<SOS>"]] + char_ids
    char_target = char_ids + [char2id["<EOS>"]]

    dataset.append((token_id, char_input, char_target))


# ==========================
# Hyperparameters and Model
# ==========================

token_embedding_dim = 32
char_embedding_dim = 32  # Needs to match token_embedding_dim
hidden_dim = 64
nhead = 4
num_layers = 2
learning_rate = 0.001
num_epochs = 100

model = TokenToCharTransformer(
    token_vocab_size=token_vocab_size,
    char_vocab_size=char_vocab_size,
    token_embedding_dim=token_embedding_dim,
    char_embedding_dim=char_embedding_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    nhead=nhead
)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


# =====================
# Training Loop
# =====================

for epoch in range(num_epochs):
    total_loss = 0.0
    random.shuffle(dataset)

    for token_id, char_input, char_target in dataset:
        token_ids = torch.tensor([token_id])  # (batch_size=1,)
        char_input_ids = torch.tensor(char_input).unsqueeze(1)  # (seq_len, batch_size=1)
        char_target_ids = torch.tensor(char_target).unsqueeze(1)  # (seq_len, batch_size=1)

        optimizer.zero_grad()

        logits = model(token_ids, char_input_ids)  # (seq_len, batch_size, char_vocab_size)

        logits = logits.squeeze(1)  # (seq_len, vocab_size)
        char_target_ids = char_target_ids.squeeze(1)  # (seq_len,)

        loss = criterion(logits, char_target_ids)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")


# ========================
# Inference Function
# ========================

def decode_token(model, token_id, max_length=20):
    model.eval()

    token_ids = torch.tensor([token_id])
    input_char_ids = torch.tensor([char2id["<SOS>"]]).unsqueeze(1)  # (1, batch_size=1)

    output_chars = []

    for _ in range(max_length):
        logits = model(token_ids, input_char_ids)  # (seq_len, batch_size, char_vocab_size)

        next_logits = logits[-1, 0]  # Last step
        probs = torch.softmax(next_logits, dim=-1)
        next_char_id = torch.argmax(probs).item()

        if next_char_id == char2id["<EOS>"]:
            break

        output_chars.append(id2char[next_char_id])

        # Append next char for next step
        next_char_tensor = torch.tensor([[next_char_id]])  # (1, batch_size=1)
        input_char_ids = torch.cat([input_char_ids, next_char_tensor], dim=0)

    return "".join(output_chars)


# ========================
# Test Decoding
# ========================

test_token = token2id["distinguishability"]  # Replace with any word from your dataset
output = decode_token(model, test_token)
print("Decoded:", output)