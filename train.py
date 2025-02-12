import gzip
import random
import tqdm
import numpy as np
import argparse

import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from accelerate import Accelerator
from block_recurrent_transformer_pytorch import BlockRecurrentTransformer, RecurrentTrainerWrapper
from accelerate.utils import DistributedDataParallelKwargs

# parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train Block Recurrent Transformer')
    parser.add_argument('--device', type=str, default='auto',
                      choices=['cpu', 'cuda', 'mps', 'auto'],
                      help='Device to use (auto, cpu, cuda, or mps)')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--num_batches', type=int, default=int(1e5),
                      help='Number of batches to train')
    parser.add_argument('--gradient_accumulate_every', type=int, default=4,
                      help='Number of steps to accumulate gradients')
    return parser.parse_args()

args = parse_args()

# constants
NUM_BATCHES = args.num_batches
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATE_EVERY = args.gradient_accumulate_every
LEARNING_RATE = args.learning_rate
VALIDATE_EVERY = 100
PRIME_LENGTH = 128
GENERATE_EVERY = 250
GENERATE_LENGTH = 2048
SEQ_LEN = 2048

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

# Initialize accelerator with device preference
accelerator = Accelerator(
    gradient_accumulation_steps=GRADIENT_ACCUMULATE_EVERY,
    cpu=(args.device == 'cpu'),
    device_placement=(args.device != 'auto'),
    mixed_precision='fp16' if torch.cuda.is_available() else 'no',
    kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
)
device = accelerator.device
acc_print = accelerator.print

# Print device information
acc_print(f"Using device: {device}")
acc_print(f"Training arguments: {args}")

# instantiate palm with optimized settings
model = BlockRecurrentTransformer(
    num_tokens = 256,
    dim = 512,
    depth = 6,
    dim_head = 64,
    heads = 8,
    max_seq_len = 1024,
    block_width = 512,
    num_state_vectors = 512,
    recurrent_layers = (4,),
    use_flash_attn = True if torch.cuda.is_available() else False,
    use_compressed_mem = True,
    compressed_mem_factor = 4,
    all_layers_qk_rmsnorm = True  # Enable rmsnorm for all layers for better performance
)

# Enable memory efficient attention if not using A100
if torch.cuda.is_available() and not (torch.cuda.get_device_properties(0).major == 8 and torch.cuda.get_device_properties(0).minor == 0):
    for module in model.modules():
        if hasattr(module, 'use_flash_attn'):
            module.use_flash_attn = False

train_wrapper = RecurrentTrainerWrapper(
    model,
    xl_memories_dropout = 0.1,
    state_dropout = 0.1,
)

# Optimize memory usage
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere
torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
torch.backends.cudnn.allow_tf32 = True  # Allow TF32 for cudnn

# prepare enwik8 data

with gzip.open("./data/enwik8.gz") as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    np_train, np_valid = np.split(data, [int(90e6)])
    data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.total_len = self.data.size(0) - self.seq_len - 1
        
    def __getitem__(self, index):
        # Use vectorized operations instead of random sampling
        rand_start = torch.randint(0, self.total_len, (1,))
        full_seq = self.data[rand_start:rand_start + self.seq_len + 1].long()
        return full_seq.to(device, non_blocking=True)  # Enable async transfer

    def __len__(self):
        return self.total_len // self.seq_len

def create_dataloaders(batch_size):
    train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
    val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
    
    # Use multiple workers and pin memory for faster data loading
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    
    return train_loader, val_loader

# Create optimized dataloaders
train_loader, val_loader = create_dataloaders(BATCH_SIZE)

# optimizer
optim = Adam(model.parameters(), lr=LEARNING_RATE)

# prepare for training using accelerator
model, train_wrapper, optim, train_loader, val_loader = accelerator.prepare(
    model, train_wrapper, optim, train_loader, val_loader
)

# Create cyclic iterators for the dataloaders
train_iter = cycle(train_loader)
val_iter = cycle(val_loader)

# training
pbar = tqdm.tqdm(range(NUM_BATCHES), mininterval=10.0, desc="training")
for i in pbar:
    model.train()

    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = train_wrapper(next(train_iter))
        accelerator.backward(loss / GRADIENT_ACCUMULATE_EVERY)

    metrics = {'train_loss': f"{loss.item():.4f}"}

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            val_loss = train_wrapper(next(val_iter))
            metrics['val_loss'] = f"{val_loss.item():.4f}"

    pbar.set_postfix(**metrics)
    accelerator.clip_grad_norm_(model.parameters(), 0.5)

    optim.step()
    optim.zero_grad()

    if i % GENERATE_EVERY == 0:
        model.eval()
        inp = next(val_iter)[:PRIME_LENGTH]  # Use val_iter instead of random.choice
        prime = decode_tokens(inp)
        acc_print(f"%s \n\n %s", (prime, "*" * 100))

        # Move input to the same device as model
        inp = accelerator.prepare(inp[None, ...])
        sample = train_wrapper.generate(inp, length=GENERATE_LENGTH)
        output_str = decode_tokens(accelerator.gather(sample)[0])
        acc_print(output_str, "\n")
