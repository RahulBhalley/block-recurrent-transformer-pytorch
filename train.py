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

def init_accelerator(device_str: str = None, mixed_precision: str = 'no'):
    """Initialize accelerator with optional device override and mixed precision"""
    if device_str and device_str != 'auto':
        # Manual device override
        if device_str == 'cpu':
            device = 'cpu'
            mixed_precision = 'no'  # Force no mixed precision for CPU
        elif device_str.startswith('cuda'):
            device = 'cuda'
            multi_gpu = torch.cuda.device_count() > 1
        elif device_str == 'mps':
            device = 'mps'
            mixed_precision = 'no'  # Force no mixed precision for MPS
            multi_gpu = False
        else:
            raise ValueError(f"Unsupported device: {device_str}")
        
        accelerator = Accelerator(
            device=device,
            mixed_precision=mixed_precision,
            split_batches=multi_gpu if 'multi_gpu' in locals() else False,
            gradient_accumulation_steps=GRADIENT_ACCUMULATE_EVERY
        )
    else:
        # Let Accelerate automatically choose the best device
        multi_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 1
        accelerator = Accelerator(
            mixed_precision=mixed_precision,
            split_batches=multi_gpu,
            gradient_accumulation_steps=GRADIENT_ACCUMULATE_EVERY
        )
    
    return accelerator, accelerator.device

# accelerator
accelerator, device = init_accelerator(args.device)
acc_print = accelerator.print

# Print device information and arguments
acc_print(f"Using device: {device}")
acc_print(f"Training arguments: {args}")

# instantiate palm

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
    use_flash_attn = True
)

train_wrapper = RecurrentTrainerWrapper(
    model,
    xl_memories_dropout = 0.1,
    state_dropout = 0.1,
)

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

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))
val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE))

# optimizer
optim = Adam(model.parameters(), lr=LEARNING_RATE)

# prepare for training using accelerator
model, train_wrapper, optim, train_loader, val_loader = accelerator.prepare(
    model, train_wrapper, optim, train_loader, val_loader
)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10.0, desc="training"):
    model.train()

    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = train_wrapper(next(train_loader))
        accelerator.backward(loss / GRADIENT_ACCUMULATE_EVERY)

    acc_print(f"training loss: {loss.item()}")
    accelerator.clip_grad_norm_(model.parameters(), 0.5)

    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            loss = train_wrapper(next(val_loader))
            acc_print(f"validation loss: {loss.item()}")

    if i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        prime = decode_tokens(inp)
        acc_print(f"%s \n\n %s", (prime, "*" * 100))

        # Move input to the same device as model
        inp = accelerator.prepare(inp[None, ...])
        sample = train_wrapper.generate(inp, length=GENERATE_LENGTH)
        output_str = decode_tokens(accelerator.gather(sample)[0])
        acc_print(output_str, "\n")
