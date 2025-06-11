import time

out_dir = 'out-shakespeare'
wandb_log = False # feel free to turn on
eval_interval = 20 # keep frequent because we'll overfit
eval_iters = 100
#log_interval = 10 # don't print too too often

dataset = 'shakespeare'

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 32
gradient_accumulation_steps = 32
max_iters = 500

# finetune at constant LR
learning_rate = 3e-4
decay_lr = False
device = 'cpu'
max_token_length = 16 + 2 # For <SOS> and <EOS> tokens

n_layer = 6
n_head = 6
n_embd = 384
block_size = 128
dropout = 0.1
charset = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
char_to_id = {ch: idx + 1 for idx, ch in enumerate(charset)}  # start indexing at 1
char_to_id['<pad>'] = 0
char_to_id['<SOS>'] = len(char_to_id)
char_to_id['<EOS>'] = len(char_to_id)
id_to_char = {idx: ch for ch, idx in char_to_id.items()}


charset = sorted(set(charset))
charset_size = len(charset) + 3