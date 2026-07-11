import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# must run before importing torch

import torch
from torch import nn
import math
import gc
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from model import Transformer
from dataset import (
    train_data, val_data, src_vocab, tgt_vocab, 
    Multi30kDataset, collate_function
)

def train_batch(model, src, tgt, optimizer, criterion, clip):
    # Shift Targets
    tgt_input = tgt[:, :-1] # Remove <EOS> at the end
    tgt_output = tgt[:, 1:] # Remove <SOS> at the start

    optimizer.zero_grad()
    logits = model(src, tgt_input)
    loss = criterion(logits.permute(0, 2, 1), tgt_output)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip) # Clip Gradients
    optimizer.step()

    return loss.item()


def evaluate_batch(model, src, tgt, criterion):
    # Shift Targets
    tgt_input = tgt[:, :-1] # Remove <EOS> at the end
    tgt_output = tgt[:, 1:] # Remove <SOS> at the start

    with torch.inference_mode():
        logits = model(src, tgt_input)
        loss = criterion(logits.permute(0, 2, 1), tgt_output)

    return loss.item()

def plot_loss_curves(train_loss, test_loss):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(len(train_loss)), y=train_loss, label='Training Loss')
    sns.lineplot(x=range(len(test_loss)), y=test_loss, label='Test Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def run_training():
    transformer = Transformer(
        model_dim=MODEL_DIM,
        num_layers=NUM_LAYERS,
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        padding_idx=PADDING_IDX,
        dropout=DROPOUT
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=PADDING_IDX, label_smoothing=0.1) # NOTE -> Label smoothing is important to add here, it makes our loss Super-CrossEntropy !
    optimizer = torch.optim.Adam(params=transformer.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    scheduler = StepLR(optimizer=optimizer, step_size=7, gamma=0.8)

    train_loss_list = []
    val_loss_list = []
    best_val_loss = float('inf')
    try:
        for epoch in range(EPOCH):
            # --- Train --- #
            transformer.train()
            total_train_loss = 0
            for src, tgt in train_dataloader:
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                total_train_loss += train_batch(transformer, src, tgt, optimizer, criterion, GRAD_CLIP_VALUE)

            # --- Validation --- #
            total_val_loss = 0
            transformer.eval()
            for src, tgt in val_dataloader:
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                total_val_loss += evaluate_batch(transformer, src, tgt, criterion)

            train_loss_list.append(total_train_loss / len(train_dataloader))
            val_loss_list.append(total_val_loss / len(val_dataloader))

            # --- Log Output --- #
            print(f'Epoch: {epoch+1:02}')
            print(f'\tTrain Loss: {train_loss_list[-1]:.4f} | Train PPL: {math.exp(train_loss_list[-1]):.4f}')
            print(f'\t Val. Loss: {val_loss_list[-1]:.4f} |  Val. PPL: {math.exp(val_loss_list[-1]):.4f}')
            print(f'\t Learning Rate: {scheduler.get_last_lr()}')

            scheduler.step()

            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                torch.save(transformer.state_dict(), 'best_model.pth')
                # print(f"    -> Saved best model (Loss: {best_val_loss:.4f})")

    except KeyboardInterrupt:
        print("Training interrupted by user.")

    except Exception as e:
        print(f"Training crashed with error: {e}")

    finally:
        print("Cleaning up GPU memory...")
        del optimizer
        del scheduler
        gc.collect() # <--- Force Python garbage collection
        torch.cuda.empty_cache() # <--- Clear GPU cache
        torch.cuda.ipc_collect()

    return transformer, train_loss_list, val_loss_list


if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'


BATCH_SIZE = 512
EPOCH = 25
NUM_LAYERS = 3
LEARNING_RATE = 0.0005
PADDING_IDX = src_vocab['[PAD]'] # ! the src and tgt vocab must have the same padding index ( could be fixed in the Transformer class)
MODEL_DIM = 256
GRAD_CLIP_VALUE = 1
DROPOUT = 0.25

if __name__ == "__main__":
    train_dataset = Multi30kDataset(train_data, src_vocab, tgt_vocab)
    val_dataset = Multi30kDataset(val_data, src_vocab, tgt_vocab)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_function,
        shuffle=True,
        num_workers=2,            # PARALLEL LOADING (Try 2 or 4 in Colab)
        pin_memory=True           # FASTER TRANSFER (CPU -> GPU)
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_function,
        shuffle=True,
        num_workers=2,            # PARALLEL LOADING (Try 2 or 4 in Colab)
        pin_memory=True           # FASTER TRANSFER (CPU -> GPU)
    )

    transformer, train_loss_list, val_loss_list = run_training()
    plot_loss_curves(train_loss_list, val_loss_list)
