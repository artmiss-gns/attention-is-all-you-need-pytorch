import torch
import spacy
from datasets import load_dataset
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

# 2. Tokenizers (Spacy)
spacy_en = spacy.load("en_core_web_sm")
spacy_de = spacy.load("de_core_news_sm")

def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

def tokenize_de(text):
    return [tok.text.lower() for tok in spacy_de.tokenizer(text)]

# 3. Build Vocabulary
def build_vocab(sentences, tokenizer, min_freq=2):
    counter = Counter()
    for s in sentences:
        counter.update(tokenizer(s))

    # Start with special tokens
    vocab = {'[PAD]': 0, '[SOS]': 1, '[EOS]': 2, '[UNK]': 3}
    idx = 4

    # filtering the vocab based on a minimum threshold
    for word, count in counter.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1

    # to make this deterministic, we sort by word
    vocab = dict(sorted(vocab.items()))
    # ! This doesn't guarantee that the reproducibility is perfect across different runs/environments, but it's enough for our test

    return vocab

def word2idx(tokens, vocab):
    return [vocab.get(w, vocab['[UNK]']) for w in tokens]

def idx2word(idx_tokens: list, reversed_vocab):
    # idx_tokens = idx_tokens.tolist()
    return " ".join([reversed_vocab.get(w) for w in idx_tokens])

class Multi30kDataset(Dataset):
    def __init__(self, dataset, src_vocab, tgt_vocab):
        super().__init__()
        self.dataset = dataset
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __getitem__(self, index):
        src_tensor = torch.tensor(
            [self.src_vocab['[SOS]']] + word2idx(tokenize_en(self.dataset[index]['en']), self.src_vocab) + [self.src_vocab['[EOS]']]
        )
        tgt_tensor = torch.tensor(
            [self.src_vocab['[SOS]']] + word2idx(tokenize_de(self.dataset[index]['de']), self.tgt_vocab) + [self.src_vocab['[EOS]']]
        )
        return src_tensor, tgt_tensor

    def __len__(self):
        return len(self.dataset)

def collate_function(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)

    src_batch = pad_sequence(src_batch, padding_value=src_vocab['[PAD]'], batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=tgt_vocab['[PAD]'], batch_first=True)
    return src_batch, tgt_batch


# We use the 'bentrevett/multi30k' mirror as the official one is often down
print("Downloading Dataset...")
dataset = load_dataset("bentrevett/multi30k")

train_data = dataset['train']
val_data = dataset['validation']
test_data  = dataset['test']

print(f"Train size: {len(train_data)}")
print(f"Example: {train_data[0]}")
# Output: {'en': 'Two young, White males are outside near many bushes.', 'de': 'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.'}

print("Building Vocabularies... (this takes a moment)")
src_vocab = build_vocab([x['en'] for x in train_data], tokenize_en)
tgt_vocab = build_vocab([x['de'] for x in train_data], tokenize_de)

print(f"Source (EN) Vocab Size: {len(src_vocab)}")
print(f"Target (DE) Vocab Size: {len(tgt_vocab)}")

reversed_src_vocab = {k: w for w, k in src_vocab.items()}
reversed_tgt_vocab = {k: w for w, k in tgt_vocab.items()}
