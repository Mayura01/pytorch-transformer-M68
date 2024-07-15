import torch
from torch.utils.data import Dataset
import random

class MLMDataset(Dataset):
    def __init__(self, config, ds, tokenizer):
        super().__init__()
        self.seq_len = config['seq_len']
        self.mask_probability = config.get('mask_probability', 0.15)

        self.ds = ds
        self.tokenizer = tokenizer

        self.sos_token = torch.tensor([tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer.token_to_id("[PAD]")], dtype=torch.int64)
        self.mask_token = torch.tensor(tokenizer.token_to_id("[MASK]"), dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_text = self.ds[idx]['text']
        input_tokens = self.tokenizer.encode(src_text).ids
        num_padding_tokens = self.seq_len - len(input_tokens) - 2

        if num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        tokenized_text = torch.cat(
            [
                self.sos_token,
                torch.tensor(input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        assert tokenized_text.size(0) == self.seq_len

        original_tokens = tokenized_text.clone()
        tokenized_text = self.apply_mlm(tokenized_text)

        return {
            "tokenized_text": tokenized_text,  # (seq_len)
            "original_tokens": original_tokens,  # (seq_len)
        }

    def apply_mlm(self, tokens):
        for i in range(1, len(tokens) - 1):  # Exclude [SOS] and [EOS]
            if random.random() < self.mask_probability:
                tokens[i] = self.mask_token
        return tokens

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
