import torch
import torch.nn as nn
import random
import torch.optim as optim


# TODO:
# There was something we learned about traveling different paths in the LSTM
# based on the possible samples, so maybe take the highest 4 probable outputs
# and then go down 4 different paths. Maybe this could then be used to sample
# different tokens

# Maybe for the fixed decoder we sample the top k from each position
# and then use that to determine the possible characters given those tokens
# 

# === Model ===

class TokenToCharLSTM(nn.Module):
    def __init__(self, token_vocab_size, char_vocab_size, token_embedding_dim, char_embedding_dim, hidden_dim):
        super().__init__()

        self.token_embedding = nn.Embedding(token_vocab_size, token_embedding_dim)
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim)

        self.lstm = nn.LSTM(token_embedding_dim + char_embedding_dim, hidden_dim)

        self.output_layer = nn.Linear(hidden_dim, char_vocab_size)

    def forward(self, token_ids, char_input_ids):
        """
        token_ids: (batch_size,)
        char_input_ids: (seq_len, batch_size)
        """
        batch_size = token_ids.size(0)
        seq_len = char_input_ids.size(0)

        token_emb = self.token_embedding(token_ids)  # (batch_size, token_embedding_dim)
        token_emb = token_emb.unsqueeze(0).repeat(seq_len, 1, 1)  # (seq_len, batch_size, token_emb_dim)

        char_emb = self.char_embedding(char_input_ids)  # (seq_len, batch_size, char_emb_dim)

        lstm_input = torch.cat([token_emb, char_emb], dim=2)  # (seq_len, batch_size, input_dim)

        h0 = torch.zeros(1, batch_size, self.lstm.hidden_size)
        c0 = torch.zeros(1, batch_size, self.lstm.hidden_size)

        lstm_out, _ = self.lstm(lstm_input, (h0, c0))  # (seq_len, batch_size, hidden_dim)

        logits = self.output_layer(lstm_out)  # (seq_len, batch_size, char_vocab_size)

        return logits
    




with open("test_dataset.txt", "r") as file:
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


print("Token vocab:", token2id)
print("Char vocab:", char2id)


def word_to_char_ids(word):
    return [char2id[c] for c in word]

dataset = []
for word in words:
    token_id = token2id[word]
    char_ids = word_to_char_ids(word)  # ['c', 'a', 't']
    
    char_input = [char2id["<SOS>"]] + char_ids  # Input starts with SOS
    char_target = char_ids + [char2id["<EOS>"]]  # Target shifts forward with EOS
    
    dataset.append((token_id, char_input, char_target))
    

# === Hyperparameters ===

token_embedding_dim = 32
char_embedding_dim = 16
hidden_dim = 64
learning_rate = 0.01
num_epochs = 100

model = TokenToCharLSTM(
    token_vocab_size=token_vocab_size,
    char_vocab_size=char_vocab_size,
    token_embedding_dim=token_embedding_dim,
    char_embedding_dim=char_embedding_dim,
    hidden_dim=hidden_dim
)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    total_loss = 0.0

    # Shuffle dataset
    random.shuffle(dataset)

    for token_id, char_input, char_target in dataset:
        token_ids = torch.tensor([token_id])  # (batch_size=1,)
        char_input_ids = torch.tensor(char_input).unsqueeze(1)  # (seq_len, batch_size=1)
        char_target_ids = torch.tensor(char_target).unsqueeze(1)  # (seq_len, batch_size=1)

        optimizer.zero_grad()

        logits = model(token_ids, char_input_ids)  # (seq_len, batch_size, char_vocab_size)

        logits = logits.squeeze(1)  # (seq_len, char_vocab_size)
        char_target_ids = char_target_ids.squeeze(1)  # (seq_len,)

        loss = criterion(logits, char_target_ids)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

def decode_token(model, token_id, max_length=20):
    model.eval()  # Set to eval mode
    
    token_embed = model.token_embedding(torch.tensor([token_id]))  # (1, token_embedding_dim)
    
    # Start with SOS token if you used it, else use the first char or a dummy zero vector
    input_char_id = torch.tensor([char2id["<SOS>"]])  # (1,)
    char_embed = model.char_embedding(input_char_id)  # (1, char_embedding_dim)
    
    # Initialize hidden state to zeros
    hidden = None  # (PyTorch lets LSTM init to zeros if hidden=None)

    output_chars = []

    for _ in range(max_length):
        # Concatenate token embedding and current char embedding
        lstm_input = torch.cat([token_embed, char_embed], dim=-1).unsqueeze(0)  # (1, 1, token+char dim)

        output, hidden = model.lstm(lstm_input, hidden)  # output: (1, 1, hidden_dim)

        logits = model.output_layer(output.squeeze(0))  # (1, char_vocab_size)
        probs = torch.softmax(logits, dim=-1)
        next_char_id = torch.argmax(probs, dim=-1)

        if next_char_id.item() == char2id["<EOS>"]:
            break

        output_chars.append(id2char[next_char_id.item()])

        # Prepare next input: embed the predicted char
        char_embed = model.char_embedding(next_char_id)

    return "".join(output_chars)

test_token = token2id["distinguishability"]  # or any token from your vocab
output = decode_token(model, test_token)
print("Decoded:", output)