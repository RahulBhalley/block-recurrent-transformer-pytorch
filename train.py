import gzip
import random
import tqdm
import numpy as np
import argparse
import torch.multiprocessing as mp
from torch.multiprocessing import freeze_support

import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from accelerate import Accelerator
from block_recurrent_transformer_pytorch import BlockRecurrentTransformer, RecurrentTrainerWrapper
from accelerate.utils import DistributedDataParallelKwargs

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

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

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.total_len = self.data.size(0) - self.seq_len - 1
        
    def __getitem__(self, index):
        rand_start = torch.randint(0, self.total_len, (1,))
        full_seq = self.data[rand_start:rand_start + self.seq_len + 1].long()
        return full_seq

    def __len__(self):
        return self.total_len // self.seq_len

def create_dataloaders(batch_size, data_train, data_val, seq_len):
    train_dataset = TextSamplerDataset(data_train, seq_len)
    val_dataset = TextSamplerDataset(data_val, seq_len)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader

def main():
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
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

    # Initialize accelerator
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

    # Load data
    with gzip.open("./data/enwik8.gz") as file:
        data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
        np_train, np_valid = np.split(data, [int(90e6)])
        data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

    # Create model
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
        all_layers_qk_rmsnorm = True
    )

    if torch.cuda.is_available() and not (torch.cuda.get_device_properties(0).major == 8 and torch.cuda.get_device_properties(0).minor == 0):
        for module in model.modules():
            if hasattr(module, 'use_flash_attn'):
                module.use_flash_attn = False

    # Optimize CUDA settings
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True

    # Create trainer wrapper
    train_wrapper = RecurrentTrainerWrapper(
        model,
        xl_memories_dropout = 0.1,
        state_dropout = 0.1,
    )

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(BATCH_SIZE, data_train, data_val, SEQ_LEN)

    # Create optimizer
    optim = Adam(model.parameters(), lr=LEARNING_RATE)

    # Prepare for training using accelerator
    model, train_wrapper, optim, train_loader, val_loader = accelerator.prepare(
        model, train_wrapper, optim, train_loader, val_loader
    )

    # Create iterators
    train_iter = cycle(train_loader)
    val_iter = cycle(val_loader)

    # Training loop
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
            inp = next(val_iter)[:PRIME_LENGTH]
            prime = decode_tokens(inp)
            acc_print(f"%s \n\n %s", (prime, "*" * 100))

            inp = accelerator.prepare(inp[None, ...])
            sample = train_wrapper.generate(inp, length=GENERATE_LENGTH)
            output_str = decode_tokens(accelerator.gather(sample)[0])
            acc_print(output_str, "\n")

if __name__ == '__main__':
    freeze_support()  # For Windows support
    main()
