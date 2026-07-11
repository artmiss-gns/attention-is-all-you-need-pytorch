import torch
from torch import nn
from torchmetrics.text import BLEUScore

from model import Transformer
from dataset import (
    test_data, src_vocab, tgt_vocab, reversed_tgt_vocab,
    tokenize_en, word2idx, idx2word
)

def translate(input_src: str, model: nn.Module, max_len=200):
    # --- Preprocessing & Tokenization --- #
    tokenized = tokenize_en(input_src)
    tokenized = [src_vocab['[SOS]']] + word2idx(tokenized, src_vocab) + [src_vocab['[EOS]']]
    src_tensor = torch.tensor(tokenized).unsqueeze(0).to(DEVICE)
    tgt_tensor = torch.tensor([tgt_vocab['[SOS]']]).unsqueeze(0).to(DEVICE)
    
    model = model.to(DEVICE)
    model.eval()
    
    # Store result indices here
    output_indices = []
    
    with torch.inference_mode():
        for _ in range(max_len):
            logits = model(src_tensor, tgt_tensor)
            last_token_logits = logits[:, -1, :] 
            next_token_index = last_token_logits.argmax(dim=-1).item()

            if next_token_index == tgt_vocab['[EOS]']:
                break
            
            output_indices.append(next_token_index)
            
            next_token_tensor = torch.tensor([[next_token_index]]).to(DEVICE)
            tgt_tensor = torch.cat([tgt_tensor, next_token_tensor], dim=1)
        
    return idx2word(output_indices, reversed_tgt_vocab)


if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

NUM_LAYERS = 3
PADDING_IDX = src_vocab['[PAD]']
MODEL_DIM = 256
DROPOUT = 0.25


if __name__ == "__main__":
    model = Transformer(
        model_dim=MODEL_DIM,
        num_layers=NUM_LAYERS,
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        padding_idx=PADDING_IDX,
        dropout=DROPOUT
    ).to(DEVICE)

    model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))

    bleu = BLEUScore()

    random_indexes = torch.randint(0, len(test_data), (5,)).tolist()
    for index in random_indexes:
        src, tgt = test_data[index]['en'], test_data[index]['de']
        preds = translate(src, model)
        score = bleu([preds], [[tgt.lower()]]) 
        
        print(f"Source: {src}")
        print(f"Target: {tgt}")
        print(f"Prediction: {preds}")
        print(f"BLEU Score: {score.item() * 100:.2f}")
        print("\n")
