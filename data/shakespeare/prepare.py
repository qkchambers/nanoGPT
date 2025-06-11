import os
import requests
import tiktoken
import numpy as np
import json
import pickle


# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
#enc = tiktoken.get_encoding("gpt2")
#train_ids = enc.encode_ordinary(train_data)
#val_ids = enc.encode_ordinary(val_data)

def encode_fixed_chunks(text, k):
    chunks = []
    for i in range(0, len(text), k):
        chunk = text[i:i+k]
        if len(chunk) < k:
            chunk += '\0' * (k - len(chunk))  # pad with null characters
        chunks.append(chunk)
    return chunks

k = 4
train_chunks = encode_fixed_chunks(train_data, k)
val_chunks = encode_fixed_chunks(val_data, k)

# Optionally, map each chunk to an integer ID (simple vocabulary-based tokenizer)
# Build vocabulary
vocab = sorted(set(train_chunks + val_chunks))
chunk_to_id = {chunk: i for i, chunk in enumerate(vocab)}

# Convert chunks to token IDs
train_ids = [chunk_to_id[chunk] for chunk in train_chunks]
val_ids = [chunk_to_id[chunk] for chunk in val_chunks]

all_chars = sorted(set(train_data + val_data) | {'\0', '<SOS>', '<EOS>'})  
char_to_id = {ch: i for i, ch in enumerate(all_chars)}  # \0 will be ID 0
id_to_char = {i: ch for ch, i in char_to_id.items()}

# Save vocab and mappings
with open("vocab.json", "w") as f:
    json.dump(vocab, f)  # list of string chunks

# Optionally save mappings directly if you don't want to rebuild them
with open("chunk_to_id.pkl", "wb") as f:
    pickle.dump(chunk_to_id, f)

with open("char_vocab.json", "w") as f:
    json.dump(char_to_id, f)



print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
